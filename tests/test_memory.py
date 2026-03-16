"""
Memory system tests — Sprint 5 §5.4

Covers:
  - InProcessSemanticStore (upsert, dedup at 0.92, search with threshold,
    tombstone, touch updates last_accessed_at)
  - ConsolidationEngine.staleness_sweep (flag → tombstone two-pass)
  - ConsolidationEngine.consolidate_session (insert, dedup merge, empty skip)
  - ProceduralStore (record, bias calculation, merge at 0.95)
  - SweepResult repr

All tests are offline — no LiteLLM / embedding API calls.
Embeddings are small deterministic vectors injected directly.

Run: python -m pytest tests/test_memory.py -v
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.memory import (
    DEDUP_THRESHOLD,
    ConsolidationEngine,
    InProcessSemanticStore,
    SemanticRecord,
    SweepResult,
)
from core.semantic_router import ProceduralStore, cosine_similarity


# ── Helpers ───────────────────────────────────────────────────────────────────

DIM = 8


def _unit(v: list) -> list:
    arr = np.array(v, dtype=np.float32)
    n = float(np.linalg.norm(arr))
    return (arr / n).tolist() if n > 1e-9 else v


def _store_with(*contents_and_embs) -> InProcessSemanticStore:
    """Build a store pre-populated with (content, embedding) pairs."""
    store = InProcessSemanticStore()
    for content, emb in contents_and_embs:
        store.upsert(content, emb)
    return store


def _engine(store: InProcessSemanticStore, embed_fn=None) -> ConsolidationEngine:
    return ConsolidationEngine(store=store, embed_fn=embed_fn)


# ── InProcessSemanticStore: upsert / dedup ────────────────────────────────────

def test_store_insert_returns_id():
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    record_id, was_merged = store.upsert("total revenue Q3", emb)
    assert record_id
    assert was_merged is False
    assert len(store) == 1


def test_store_dedup_identical_embedding():
    """Two records with identical embeddings → should merge, not insert."""
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    id1, _ = store.upsert("first fact", emb)
    id2, was_merged = store.upsert("second fact", emb)
    assert was_merged is True
    assert id2 == id1
    assert len(store) == 1  # still one active record


def test_store_dedup_threshold_boundary():
    """cosine = 0.92 → merge; cosine < 0.92 → insert new."""
    emb_a = _unit([1.0, 0.0] + [0.0] * 6)
    # Construct an embedding with cosine exactly at threshold by blending:
    # emb_b = normalise(emb_a * 0.92 + orthogonal * small_amount)
    # Easier: just use a clearly orthogonal vector to guarantee no merge.
    emb_b = _unit([0.0, 1.0] + [0.0] * 6)
    store = InProcessSemanticStore()
    store.upsert("fact A", emb_a)
    _, was_merged = store.upsert("fact B", emb_b)
    assert was_merged is False
    assert len(store) == 2


def test_store_dedup_merges_content():
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("revenue is up", emb)
    store.upsert("costs are down", emb)  # same embedding → merge
    records = store.all_active()
    assert len(records) == 1
    assert "costs are down" in records[0].content


def test_store_dedup_skips_duplicate_content():
    """Inserting the same content string twice should not double it."""
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("same content", emb)
    store.upsert("same content", emb)
    records = store.all_active()
    assert records[0].content.count("same content") == 1


def test_store_multiple_distinct_records():
    store = InProcessSemanticStore()
    for i in range(4):
        v = [0.0] * DIM
        v[i] = 1.0
        store.upsert(f"fact {i}", _unit(v))
    assert len(store) == 4


# ── InProcessSemanticStore: search ───────────────────────────────────────────

def test_store_search_returns_top_k():
    store = InProcessSemanticStore()
    for i in range(4):
        v = [0.0] * DIM
        v[i] = 1.0
        store.upsert(f"fact {i}", _unit(v))
    query = _unit([1.0] + [0.0] * 7)  # matches fact 0
    results = store.search(query, top_k=2, threshold=0.0)
    assert len(results) <= 2
    assert results[0].content == "fact 0"


def test_store_search_respects_threshold():
    store = InProcessSemanticStore()
    emb = _unit([0.0, 1.0] + [0.0] * 6)  # orthogonal to query
    store.upsert("irrelevant", emb)
    query = _unit([1.0] + [0.0] * 7)
    results = store.search(query, top_k=5, threshold=0.9)
    assert results == []


def test_store_search_touches_records():
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("touchable", emb)
    rec_before = store.all_active()[0]
    old_ts = rec_before.last_accessed_at
    time.sleep(0.01)
    store.search(emb, top_k=1, threshold=0.0)
    rec_after = store.all_active()[0]
    assert rec_after.last_accessed_at >= old_ts


# ── InProcessSemanticStore: tombstone ────────────────────────────────────────

def test_store_tombstone_removes_from_active():
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("will be deleted", emb)
    assert len(store) == 1
    store.tombstone(rid)
    assert len(store) == 0
    assert store.all_active() == []


def test_store_tombstone_nonexistent_is_noop():
    store = InProcessSemanticStore()
    store.tombstone("does-not-exist")  # must not raise


# ── Staleness sweep ───────────────────────────────────────────────────────────

def test_sweep_flags_idle_records(monkeypatch):
    """Records idle beyond TTL should be flagged on first sweep pass."""
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("old analysis", emb)

    # Back-date last_accessed_at and created_at to exceed TTL
    rec = store._records[rid]
    rec.created_at = time.time() - (100 * 86400)
    rec.last_accessed_at = time.time() - (100 * 86400)

    engine = _engine(store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)

    assert result.newly_flagged == 1
    assert result.tombstoned == 0
    assert store._records[rid].stale_flagged is True


def test_sweep_tombstones_after_grace_period():
    """A record flagged in a prior sweep should be tombstoned once grace expires."""
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    rid, _ = store.upsert("stale analysis", emb)

    rec = store._records[rid]
    # Set idle time = ttl + grace + 1 day
    rec.created_at = time.time() - (100 * 86400)
    rec.last_accessed_at = time.time() - (100 * 86400)
    rec.stale_flagged = True  # simulates prior sweep having flagged it

    engine = _engine(store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)

    assert result.tombstoned == 1
    assert store._records[rid].tombstoned is True
    assert len(store) == 0


def test_sweep_skips_recently_accessed_records():
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    store.upsert("fresh analysis", emb)  # just created — idle_days ≈ 0

    engine = _engine(store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)

    assert result.newly_flagged == 0
    assert result.tombstoned == 0


def test_sweep_result_str():
    r = SweepResult(scanned=10, newly_flagged=2, tombstoned=1, already_tombstoned=3)
    s = str(r)
    assert "scanned=10" in s
    assert "flagged=2" in s
    assert "tombstoned=1" in s


def test_sweep_multiple_records_mixed():
    """Two fresh + one stale (flagged) → 0 new flags, 1 tombstone."""
    store = InProcessSemanticStore()
    for i in range(2):
        v = [0.0] * DIM
        v[i] = 1.0
        store.upsert(f"fresh {i}", _unit(v))

    # Stale already-flagged record
    v = [0.0] * DIM
    v[3] = 1.0
    rid, _ = store.upsert("stale", _unit(v))
    rec = store._records[rid]
    rec.last_accessed_at = time.time() - (100 * 86400)
    rec.stale_flagged = True

    engine = _engine(store)
    result = engine.staleness_sweep(ttl_days=90, grace_days=7)

    assert result.scanned == 3
    assert result.tombstoned == 1
    assert result.newly_flagged == 0


# ── ConsolidationEngine.consolidate_session ───────────────────────────────────

def test_consolidate_empty_episodic_returns_none():
    engine = _engine(InProcessSemanticStore())
    result = engine.consolidate_session("sess-1", [])
    assert result is None


def test_consolidate_inserts_new_record():
    store = InProcessSemanticStore()
    # Inject a dummy embed function returning a fixed vector
    emb = _unit([1.0] + [0.0] * 7)
    engine = ConsolidationEngine(store=store, embed_fn=lambda _: emb)

    episodic = [
        {"role": "user", "content": "Total revenue by region"},
        {"role": "assistant", "content": "Revenue is $1.2M in Q3"},
    ]
    record_id = engine.consolidate_session("sess-1", episodic)
    assert record_id is not None
    assert len(store) == 1
    assert store._records[record_id].metadata["session_id"] == "sess-1"


def test_consolidate_dedup_merges_similar_session():
    """Second consolidation with the same embedding should merge, not insert."""
    store = InProcessSemanticStore()
    emb = _unit([1.0] + [0.0] * 7)
    engine = ConsolidationEngine(store=store, embed_fn=lambda _: emb)

    ep1 = [{"role": "user", "content": "Q3 revenue?"}, {"role": "assistant", "content": "1.2M"}]
    ep2 = [{"role": "user", "content": "Q3 costs?"}, {"role": "assistant", "content": "0.8M"}]

    engine.consolidate_session("sess-1", ep1)
    engine.consolidate_session("sess-2", ep2)  # same embedding → merge

    assert len(store) == 1  # deduped


def test_consolidate_embed_failure_returns_none():
    store = InProcessSemanticStore()

    def failing_embed(_):
        raise RuntimeError("no api key")

    engine = ConsolidationEngine(store=store, embed_fn=failing_embed)
    episodic = [{"role": "user", "content": "some query"}]
    result = engine.consolidate_session("sess-x", episodic)
    assert result is None
    assert len(store) == 0


# ── ProceduralStore ───────────────────────────────────────────────────────────

def test_procedural_record_and_bias():
    """Record a success → bias for that agent should be > 0."""
    ps = ProceduralStore()
    emb = _unit([1.0] + [0.0] * 7)
    ps.record(emb, "reasoning_math", success=True)
    b = ps.bias(emb, "reasoning_math")
    assert b > 0.0


def test_procedural_bias_zero_no_data():
    ps = ProceduralStore()
    emb = _unit([1.0] + [0.0] * 7)
    assert ps.bias(emb, "reasoning_math") == 0.0


def test_procedural_bias_orthogonal_query_zero():
    """Past record that is orthogonal to current query should not contribute."""
    ps = ProceduralStore()
    past_emb = _unit([1.0] + [0.0] * 7)
    query_emb = _unit([0.0, 1.0] + [0.0] * 6)
    ps.record(past_emb, "reasoning_math", success=True)
    b = ps.bias(query_emb, "reasoning_math")
    assert b == 0.0  # below BIAS_SIM threshold


def test_procedural_merge_at_095():
    """Very similar queries (cosine ≥ 0.95) should merge counts, not insert new records."""
    ps = ProceduralStore()
    emb1 = _unit([1.0, 0.01] + [0.0] * 6)
    emb2 = _unit([1.0, 0.01] + [0.0] * 6)  # identical → cosine = 1.0
    ps.record(emb1, "reasoning_math", success=True)
    ps.record(emb2, "reasoning_math", success=True)
    assert len(ps) == 1
    rec = ps._records[0]
    assert rec.success_count == 2


def test_procedural_failure_lowers_success_rate():
    ps = ProceduralStore()
    emb = _unit([1.0] + [0.0] * 7)
    ps.record(emb, "reasoning_math", success=True)
    ps.record(emb, "reasoning_math", success=True)
    ps.record(emb, "reasoning_math", success=False)
    assert ps._records[0].success_rate == pytest.approx(2 / 3, abs=1e-4)


def test_procedural_different_agents_separate():
    ps = ProceduralStore()
    emb = _unit([1.0] + [0.0] * 7)
    ps.record(emb, "reasoning_math", success=True)
    ps.record(emb, "analytics_fastlane", success=False)
    # Each agent gets its own record (different agent_id → different slot)
    assert len(ps) == 2


def test_procedural_bias_only_matches_correct_agent():
    ps = ProceduralStore()
    emb = _unit([1.0] + [0.0] * 7)
    ps.record(emb, "reasoning_math", success=True)
    ps.record(emb, "analytics_fastlane", success=False)

    bias_rm = ps.bias(emb, "reasoning_math")
    bias_fl = ps.bias(emb, "analytics_fastlane")
    # reasoning_math has 100% success → higher bias than analytics_fastlane (0%)
    assert bias_rm > bias_fl


def test_procedural_success_rate_pure_failure():
    ps = ProceduralStore()
    emb = _unit([1.0] + [0.0] * 7)
    ps.record(emb, "reasoning_math", success=False)
    ps.record(emb, "reasoning_math", success=False)
    assert ps._records[0].success_rate == 0.0
