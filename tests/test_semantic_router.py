"""
Semantic Router tests — Sprint 5 §3 + §4.

All tests run fully offline: no LiteLLM / OpenAI calls.
Embeddings are injected directly into CapabilityRecord instances so we can
test the scoring formula and routing logic deterministically.

Run: python -m pytest tests/test_semantic_router.py -v
"""
import math
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from core.semantic_router import (
    CapabilityMeta,
    CapabilityRecord,
    CapabilityRegistry,
    RoutingLog,
    SemanticRouter,
    _AGENT_PROFILES,
    cosine_similarity,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

DIM = 8  # small embedding dimension for tests


def _unit(v: list) -> list:
    """Return L2-normalised version of v."""
    arr = np.array(v, dtype=np.float32)
    norm = np.linalg.norm(arr)
    return (arr / norm).tolist() if norm > 1e-9 else v


def _meta(cost: float = 0.5) -> CapabilityMeta:
    return CapabilityMeta(domain="analytics", cost_weight=cost,
                          latency_hint="medium", requires_sandbox=False)


def _record(agent_id: str, desc_emb: list, example_embs: list,
            cost: float = 0.5, version: int = 1) -> CapabilityRecord:
    return CapabilityRecord(
        agent_id=agent_id,
        version=version,
        description=f"Description for {agent_id}",
        description_embedding=desc_emb,
        example_texts=["example"],
        example_embeddings=example_embs,
        metadata=_meta(cost),
    )


def _router_with_records(*records: CapabilityRecord, threshold: float = 0.72) -> SemanticRouter:
    registry = CapabilityRegistry()
    for r in records:
        registry.register(r)
    router = SemanticRouter(registry=registry, threshold=threshold)
    return router


# ── cosine_similarity ─────────────────────────────────────────────────────────

def test_cosine_identical_vectors():
    v = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-6


def test_cosine_orthogonal_vectors():
    a = _unit([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    b = _unit([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert abs(cosine_similarity(a, b)) < 1e-6


def test_cosine_opposite_vectors():
    a = _unit([1.0] + [0.0] * 7)
    b = _unit([-1.0] + [0.0] * 7)
    assert abs(cosine_similarity(a, b) + 1.0) < 1e-6


def test_cosine_zero_vector():
    a = [0.0] * 8
    b = _unit([1.0] + [0.0] * 7)
    assert cosine_similarity(a, b) == 0.0


# ── CapabilityRegistry ────────────────────────────────────────────────────────

def test_registry_register_and_get():
    reg = CapabilityRegistry()
    emb = _unit([1.0] + [0.0] * 7)
    rec = _record("agent_a", emb, [emb])
    reg.register(rec)
    assert reg.get("agent_a") is rec


def test_registry_version_increments_on_re_register():
    reg = CapabilityRegistry()
    emb = _unit([1.0] + [0.0] * 7)
    rec1 = _record("agent_a", emb, [emb], version=1)
    reg.register(rec1)
    rec2 = _record("agent_a", emb, [emb], version=1)  # same start version
    reg.register(rec2)
    assert reg.get("agent_a").version == 2  # auto-incremented


def test_registry_versions_map():
    reg = CapabilityRegistry()
    emb = _unit([1.0] + [0.0] * 7)
    reg.register(_record("a", emb, [emb]))
    reg.register(_record("b", emb, [emb]))
    versions = reg.versions()
    assert set(versions.keys()) == {"a", "b"}
    assert all(v == 1 for v in versions.values())


def test_registry_get_missing_returns_none():
    reg = CapabilityRegistry()
    assert reg.get("nonexistent") is None


def test_registry_all_records_returns_copy():
    reg = CapabilityRegistry()
    emb = _unit([1.0] + [0.0] * 7)
    reg.register(_record("a", emb, [emb]))
    records = reg.all_records()
    assert len(records) == 1
    records.clear()  # mutating the returned list must not affect the registry
    assert len(reg.all_records()) == 1


def test_registry_len():
    reg = CapabilityRegistry()
    assert len(reg) == 0
    emb = _unit([1.0] + [0.0] * 7)
    reg.register(_record("a", emb, [emb]))
    assert len(reg) == 1


# ── SemanticRouter.score (§4.1 formula) ──────────────────────────────────────

def test_score_identical_embeddings():
    """When query == description == example, score should be high."""
    emb = _unit([1.0] + [0.0] * 7)
    rec = _record("agent", emb, [emb], cost=0.0)
    router = _router_with_records(rec)
    s = router.score(emb, rec)
    # α×1 + β×1 + γ×0 + δ×0 = 0.30 + 0.60 = 0.90
    assert abs(s - (0.30 + 0.60)) < 1e-4


def test_score_orthogonal_embeddings():
    """Orthogonal query → agent should yield only the cost term."""
    query_emb = _unit([1.0] + [0.0] * 7)
    agent_emb = _unit([0.0, 1.0] + [0.0] * 6)
    rec = _record("agent", agent_emb, [agent_emb], cost=0.5)
    router = _router_with_records(rec)
    s = router.score(query_emb, rec)
    # α×0 + β×0 + γ×0.5 = 0.05
    assert abs(s - 0.05) < 1e-4


def test_score_session_term_adds_when_provided():
    """Session embedding that matches the description adds the δ term."""
    emb = _unit([1.0] + [0.0] * 7)
    rec = _record("agent", emb, [emb], cost=0.0)
    router = _router_with_records(rec)
    s_without = router.score(emb, rec, session_emb=None)
    s_with = router.score(emb, rec, session_emb=emb)
    # δ = 0.15 added when session_emb matches description
    assert abs(s_with - s_without - 0.15) < 1e-4


def test_score_uses_max_of_example_embeddings():
    """β term uses the best-matching example, not the average."""
    query_emb = _unit([1.0] + [0.0] * 7)
    good_ex = _unit([1.0] + [0.0] * 7)    # cosine = 1.0
    bad_ex = _unit([0.0, 1.0] + [0.0] * 6)  # cosine = 0.0
    desc_emb = _unit([0.0] * 7 + [1.0])     # orthogonal to query
    rec = _record("agent", desc_emb, [bad_ex, good_ex], cost=0.0)
    router = _router_with_records(rec)
    s = router.score(query_emb, rec)
    # α×0 + β×max(0, 1) + γ×0 = 0.60
    assert abs(s - 0.60) < 1e-4


def test_score_cost_weight_contributes_gamma():
    emb_a = _unit([1.0] + [0.0] * 7)
    emb_b = _unit([0.0, 1.0] + [0.0] * 6)
    rec_low  = _record("cheap", emb_b, [emb_b], cost=0.0)
    rec_high = _record("pricey", emb_b, [emb_b], cost=1.0)
    router = _router_with_records(rec_low, rec_high)
    s_low  = router.score(emb_a, rec_low)
    s_high = router.score(emb_a, rec_high)
    # γ=0.10 difference from cost_weight
    assert abs(s_high - s_low - 0.10) < 1e-4


# ── RoutingLog ────────────────────────────────────────────────────────────────

def test_routing_log_fields_present():
    log = RoutingLog(
        timestamp="2026-03-16T00:00:00+00:00",
        query_hash="abc123",
        selected_agent="standard_analytics",
        top_score=0.85,
        runner_up_score=0.70,
        registry_versions={"standard_analytics": 1},
        fallback_used=False,
        session_length=3,
        memory_records_retrieved=2,
    )
    d = log.as_dict()
    for key in ("timestamp", "query_hash", "selected_agent", "top_score",
                "runner_up_score", "registry_versions", "fallback_used",
                "session_length", "memory_records_retrieved", "memory_precision"):
        assert key in d, f"Missing key: {key}"


def test_routing_log_memory_precision_defaults_none():
    log = RoutingLog(
        timestamp="t", query_hash="h", selected_agent="a",
        top_score=0.8, runner_up_score=0.5,
        registry_versions={}, fallback_used=False,
        session_length=0, memory_records_retrieved=0,
    )
    assert log.memory_precision is None


# ── SemanticRouter.route — embedding path ─────────────────────────────────────

def test_route_selects_highest_scoring_agent():
    """Agent whose embeddings best match the query should win."""
    query_emb = _unit([1.0] + [0.0] * 7)
    emb_a = _unit([1.0] + [0.0] * 7)    # high cosine with query
    emb_b = _unit([0.0, 1.0] + [0.0] * 6)  # low cosine with query

    rec_a = _record("agent_a", emb_a, [emb_a], cost=0.0)
    rec_b = _record("agent_b", emb_b, [emb_b], cost=0.0)
    router = _router_with_records(rec_a, rec_b, threshold=0.0)

    # Inject a fixed embed function that returns the right vector per text
    def fake_embed(text):
        return query_emb

    router.embed = fake_embed
    log = router.route("some analytics query")
    assert log.selected_agent == "agent_a"
    assert not log.fallback_used


def test_route_populates_full_log_schema():
    """RoutingLog from .route() should have all required fields populated."""
    emb = _unit([1.0] + [0.0] * 7)
    rec = _record("agent_a", emb, [emb])
    router = _router_with_records(rec, threshold=0.0)
    router.embed = lambda _: emb

    log = router.route("test query", session_length=5, memory_records_retrieved=2)

    assert log.query_hash
    assert log.timestamp
    assert log.selected_agent == "agent_a"
    assert isinstance(log.registry_versions, dict)
    assert log.session_length == 5
    assert log.memory_records_retrieved == 2
    assert log.runner_up_score == 0.0   # only one candidate


def test_route_runner_up_score_set():
    query_emb = _unit([1.0] + [0.0] * 7)
    emb_a = _unit([1.0] + [0.0] * 7)
    emb_b = _unit([0.5, 0.5] + [0.0] * 6)
    rec_a = _record("agent_a", emb_a, [emb_a])
    rec_b = _record("agent_b", emb_b, [emb_b])
    router = _router_with_records(rec_a, rec_b, threshold=0.0)
    router.embed = lambda _: query_emb
    log = router.route("query")
    assert log.runner_up_score > 0.0
    assert log.top_score >= log.runner_up_score


def test_route_registry_versions_in_log():
    emb = _unit([1.0] + [0.0] * 7)
    rec = _record("agent_a", emb, [emb])
    router = _router_with_records(rec, threshold=0.0)
    router.embed = lambda _: emb
    log = router.route("q")
    assert "agent_a" in log.registry_versions
    assert log.registry_versions["agent_a"] == 1


# ── Fallback paths ────────────────────────────────────────────────────────────

def test_keyword_fallback_on_empty_registry():
    """Empty registry → keyword fallback, no exception."""
    router = SemanticRouter(registry=CapabilityRegistry())
    log = router.route("forecast next quarter arima")
    assert log.selected_agent == "reasoning_math"
    assert log.fallback_used


def test_keyword_fallback_when_embed_returns_none():
    emb = _unit([1.0] + [0.0] * 7)
    rec = _record("agent_a", emb, [emb])
    router = _router_with_records(rec)
    router.embed = lambda _: None   # simulate embedding failure
    log = router.route("run monte carlo simulation")
    assert log.fallback_used
    assert log.selected_agent in {"reasoning_math", "analytics_fastlane", "standard_analytics"}


def test_keyword_fallback_reasoning_math():
    router = SemanticRouter(registry=CapabilityRegistry())
    for query in ["arima forecast", "monte carlo simulation", "neural network predict"]:
        log = router.route(query)
        assert log.selected_agent == "reasoning_math", f"Failed for: {query}"


def test_keyword_fallback_fastlane():
    router = SemanticRouter(registry=CapabilityRegistry())
    for query in ["bar chart of revenue", "how many orders", "pie chart by region"]:
        log = router.route(query)
        assert log.selected_agent == "analytics_fastlane", f"Failed for: {query}"


def test_keyword_fallback_standard_analytics_default():
    router = SemanticRouter(registry=CapabilityRegistry())
    log = router.route("show me cohort retention by signup month")
    assert log.selected_agent == "standard_analytics"


def test_fallback_triggered_when_score_below_threshold():
    """Score just below threshold should trigger fallback path."""
    emb = _unit([1.0] + [0.0] * 7)
    orth = _unit([0.0, 1.0] + [0.0] * 6)  # orthogonal → score ~= γ×cost
    rec = _record("agent_a", orth, [orth], cost=0.5)  # score ≈ 0.05
    router = _router_with_records(rec, threshold=0.72)
    router.embed = lambda _: emb
    # LLM fallback will also fail (no API key) → keyword fallback
    # Point: fallback_used must be True
    log = router.route("some query")
    assert log.fallback_used


# ── Agent profiles completeness ────────────────────────────────────────────────

def test_all_profiles_have_required_keys():
    required = {"description", "examples", "domain", "cost_weight", "latency_hint"}
    for agent_id, profile in _AGENT_PROFILES.items():
        missing = required - set(profile.keys())
        assert not missing, f"{agent_id} profile missing keys: {missing}"


def test_all_profiles_have_at_least_5_examples():
    for agent_id, profile in _AGENT_PROFILES.items():
        assert len(profile["examples"]) >= 5, \
            f"{agent_id} has only {len(profile['examples'])} examples (need ≥ 5)"


def test_cost_weights_in_valid_range():
    for agent_id, profile in _AGENT_PROFILES.items():
        cw = profile["cost_weight"]
        assert 0.0 <= cw <= 1.0, f"{agent_id} cost_weight {cw} out of range [0, 1]"


def test_reasoning_math_most_expensive():
    assert _AGENT_PROFILES["reasoning_math"]["cost_weight"] > \
           _AGENT_PROFILES["standard_analytics"]["cost_weight"] > \
           _AGENT_PROFILES["analytics_fastlane"]["cost_weight"]
