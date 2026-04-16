"""
MongoDBSemanticStore tests
==========================
Uses mongomock to simulate a MongoDB server in-process — no real MongoDB needed.

Covers:
  - upsert: insert new, dedup merge at cosine >= 0.92, content concatenation
  - search: linear fallback (no Atlas), threshold filter, top_k limit, touch
  - tombstone / set_stale_flag: persist flags to MongoDB docs
  - all_active: excludes tombstoned records
  - __len__: counts only active records
  - insert_qc_result / find_similar_defects: Robo QC helpers
  - ConsolidationEngine.staleness_sweep: stale_flagged persists via set_stale_flag

All tests are offline — no network calls.

Run: python -m pytest tests/test_mongodb_store.py -v
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

pytest.importorskip("mongomock", reason="mongomock not installed; run: pip install mongomock")

import mongomock

from core.memory import ConsolidationEngine, SweepResult, DEFAULT_TTL_DAYS, STALENESS_GRACE_DAYS

# ── Helpers ───────────────────────────────────────────────────────────────────

DIM = 8


def _unit(v: list) -> list:
    arr = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 1e-9 else v


@pytest.fixture()
def store(monkeypatch):
    """MongoDBSemanticStore backed by mongomock."""
    from core.mongodb_store import MongoDBSemanticStore

    # Patch MongoClient → mongomock.MongoClient so no real server needed
    monkeypatch.setattr("core.mongodb_store.MongoClient", mongomock.MongoClient)

    s = MongoDBSemanticStore.__new__(MongoDBSemanticStore)
    client = mongomock.MongoClient()
    db = client["romeoflexvision"]
    s._client = client
    s._db = db
    s._col = db["semantic_memory"]
    s._qc_col = db["quality_control"]
    s._atlas_search_available = False
    return s


# ── upsert: insert ────────────────────────────────────────────────────────────

def test_insert_returns_id(store):
    emb = _unit([1.0] + [0.0] * 7)
    record_id, was_merged = store.upsert("revenue Q3", emb)
    assert record_id
    assert was_merged is False
    assert len(store) == 1


def test_insert_multiple_distinct(store):
    for i in range(4):
        v = [0.0] * DIM
        v[i] = 1.0
        store.upsert(f"fact {i}", _unit(v))
    assert len(store) == 4


def test_insert_persists_fields(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("hello", emb, metadata={"session_id": "s1"}, ttl_days=30)
    doc = store._col.find_one({"record_id": rid})
    assert doc["content"] == "hello"
    assert doc["ttl_days"] == 30
    assert doc["metadata"]["session_id"] == "s1"
    assert doc["tombstoned"] is False
    assert doc["stale_flagged"] is False


# ── upsert: dedup / merge ─────────────────────────────────────────────────────

def test_dedup_identical_embedding(store):
    emb = _unit([1.0] + [0.0] * 7)
    id1, _ = store.upsert("fact A", emb)
    id2, was_merged = store.upsert("fact B", emb)
    assert was_merged is True
    assert id2 == id1
    assert len(store) == 1


def test_dedup_merges_content(store):
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("revenue is up", emb)
    store.upsert("costs are down", emb)  # same embedding → merge
    records = store.all_active()
    assert len(records) == 1
    assert "costs are down" in records[0].content


def test_dedup_skips_duplicate_content(store):
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("same content", emb)
    store.upsert("same content", emb)
    records = store.all_active()
    assert records[0].content.count("same content") == 1


def test_dedup_orthogonal_inserts_new(store):
    emb_a = _unit([1.0, 0.0] + [0.0] * 6)
    emb_b = _unit([0.0, 1.0] + [0.0] * 6)  # orthogonal → cosine = 0
    store.upsert("A", emb_a)
    _, was_merged = store.upsert("B", emb_b)
    assert was_merged is False
    assert len(store) == 2


# ── search ────────────────────────────────────────────────────────────────────

def test_search_top_k(store):
    for i in range(4):
        v = [0.0] * DIM
        v[i] = 1.0
        store.upsert(f"fact {i}", _unit(v))
    query = _unit([1.0] + [0.0] * 7)  # closest to fact 0
    results = store.search(query, top_k=2, threshold=0.0)
    assert len(results) <= 2
    assert results[0].content == "fact 0"


def test_search_threshold_filters(store):
    emb = _unit([0.0, 1.0] + [0.0] * 6)  # orthogonal to query
    store.upsert("irrelevant", emb)
    query = _unit([1.0] + [0.0] * 7)
    results = store.search(query, top_k=5, threshold=0.9)
    assert results == []


def test_search_excludes_tombstoned(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("deleted", emb)
    store.tombstone(rid)
    results = store.search(emb, top_k=5, threshold=0.0)
    assert results == []


def test_search_updates_last_accessed_at(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("touchable", emb)
    old_ts = store._col.find_one({"record_id": rid})["last_accessed_at"]
    time.sleep(0.02)
    store.search(emb, top_k=1, threshold=0.0)
    new_ts = store._col.find_one({"record_id": rid})["last_accessed_at"]
    assert new_ts >= old_ts


# ── tombstone ─────────────────────────────────────────────────────────────────

def test_tombstone_removes_from_active(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("will be deleted", emb)
    assert len(store) == 1
    store.tombstone(rid)
    assert len(store) == 0
    assert store.all_active() == []


def test_tombstone_persists_in_mongodb(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("x", emb)
    store.tombstone(rid)
    doc = store._col.find_one({"record_id": rid})
    assert doc["tombstoned"] is True


def test_tombstone_nonexistent_is_noop(store):
    store.tombstone("does-not-exist")  # must not raise


# ── set_stale_flag ────────────────────────────────────────────────────────────

def test_set_stale_flag_persists_in_mongodb(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("x", emb)
    store.set_stale_flag(rid, True)
    doc = store._col.find_one({"record_id": rid})
    assert doc["stale_flagged"] is True


def test_set_stale_flag_clear(store):
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("x", emb)
    store.set_stale_flag(rid, True)
    store.set_stale_flag(rid, False)
    doc = store._col.find_one({"record_id": rid})
    assert doc["stale_flagged"] is False


# ── staleness_sweep via ConsolidationEngine ───────────────────────────────────

def test_sweep_flags_idle_record(store):
    """staleness_sweep should persist stale_flagged=True via set_stale_flag."""
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("old analysis", emb)

    # Back-date to exceed TTL
    old_time = time.time() - (100 * 86400)
    store._col.update_one(
        {"record_id": rid},
        {"$set": {"created_at": old_time, "last_accessed_at": old_time}},
    )

    engine = ConsolidationEngine(store=store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)

    assert result.newly_flagged == 1
    assert result.tombstoned == 0
    # Flag must have been written to MongoDB, not just to the local SemanticRecord copy
    doc = store._col.find_one({"record_id": rid})
    assert doc["stale_flagged"] is True


def test_sweep_tombstones_after_grace(store):
    """A record with stale_flagged=True and idle > ttl+grace should be tombstoned."""
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("stale", emb)

    old_time = time.time() - (100 * 86400)
    store._col.update_one(
        {"record_id": rid},
        {"$set": {
            "created_at": old_time,
            "last_accessed_at": old_time,
            "stale_flagged": True,
        }},
    )

    engine = ConsolidationEngine(store=store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)

    assert result.tombstoned == 1
    assert len(store) == 0
    doc = store._col.find_one({"record_id": rid})
    assert doc["tombstoned"] is True


def test_sweep_skips_fresh_records(store):
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("fresh", emb)  # just created — idle ≈ 0
    engine = ConsolidationEngine(store=store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)
    assert result.newly_flagged == 0
    assert result.tombstoned == 0


# ── Robo QC helpers ───────────────────────────────────────────────────────────

def test_insert_qc_result(store):
    emb = _unit([0.5] * DIM)
    qc_id = store.insert_qc_result(
        line_id="line_07",
        defect_code="SCRATCH_A",
        confidence=0.92,
        image_path="/data/img/001.jpg",
        embedding=emb,
        metadata={"batch_id": "B001"},
    )
    assert qc_id
    doc = store._qc_col.find_one({"qc_id": qc_id})
    assert doc["line_id"] == "line_07"
    assert doc["defect_code"] == "SCRATCH_A"
    assert doc["confidence"] == pytest.approx(0.92)
    assert doc["metadata"]["batch_id"] == "B001"


def test_find_similar_defects_linear(store):
    # Insert two defects with distinct embeddings
    emb_a = _unit([1.0, 0.0] + [0.0] * 6)
    emb_b = _unit([0.0, 1.0] + [0.0] * 6)
    store.insert_qc_result("line_1", "TYPE_A", 0.9, "/img/a.jpg", embedding=emb_a)
    store.insert_qc_result("line_1", "TYPE_B", 0.8, "/img/b.jpg", embedding=emb_b)

    # Query similar to A
    results = store.find_similar_defects(emb_a, top_k=5, threshold=0.9)
    assert len(results) == 1
    assert results[0]["defect_code"] == "TYPE_A"


def test_find_similar_defects_no_embedding(store):
    """Records without embedding should not crash find_similar_defects."""
    store.insert_qc_result("line_2", "NO_EMB", 0.7, "/img/c.jpg", embedding=None)
    results = store.find_similar_defects(_unit([1.0] + [0.0] * 7), top_k=5, threshold=0.0)
    # Should return 0 — None embeddings are filtered out
    assert results == []


def test_find_similar_defects_threshold_filters(store):
    emb = _unit([0.0, 1.0] + [0.0] * 6)
    store.insert_qc_result("line_3", "DENT", 0.85, "/img/d.jpg", embedding=emb)
    query = _unit([1.0] + [0.0] * 7)  # orthogonal
    results = store.find_similar_defects(query, top_k=5, threshold=0.9)
    assert results == []
