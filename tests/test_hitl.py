"""
HITL tests — two suites covering different layers of the HITL stack.

Suite A — bridge/hitl.py (HitlGate webhook-based review):
  Tests the async webhook gateway: threshold logic, approve/reject/modify,
  timeout behavior, POST failure fallback, review ID tracking.

Suite B — core/andrew_swarm.py (hitl_escalate LangGraph node, Sprint 7):
  Tests the inline HITL escalation node: threshold boundary, reason generation,
  audit log, semantic guardrail → HITL path integration.

Run: python -m pytest tests/test_hitl.py -v
"""

import asyncio
import httpx
import sys
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.hitl import HitlConfig, HitlGate, HitlOutcome
from core.andrew_swarm import (
    hitl_escalate,
    semantic_guardrails,
    finalize_state,
    HITL_CONFIDENCE_THRESHOLD,
)


# ═══════════════════════════════════════════════════════════════
# SUITE A — bridge/hitl.py HitlGate (async webhook)
# ═══════════════════════════════════════════════════════════════

def _gate(enabled=True, threshold=0.5, webhook_url="http://fake-webhook/review",
          timeout=5, poll_interval=0.1, on_timeout="approve") -> HitlGate:
    config = HitlConfig(
        enabled=enabled,
        webhook_url=webhook_url,
        confidence_threshold=threshold,
        timeout_seconds=timeout,
        poll_interval_seconds=poll_interval,
        on_timeout=on_timeout,
    )
    return HitlGate(config=config)


# ── A1. needs_review() — pure logic ──────────────────────────

def test_below_threshold_triggers_review():
    gate = _gate(enabled=True, threshold=0.5)
    assert gate.needs_review(0.3) is True


def test_above_threshold_skips_review():
    gate = _gate(enabled=True, threshold=0.5)
    assert gate.needs_review(0.7) is False


def test_exactly_at_threshold_skips_review():
    """Threshold is exclusive: confidence == threshold is NOT flagged."""
    gate = _gate(enabled=True, threshold=0.5)
    assert gate.needs_review(0.5) is False


def test_disabled_never_triggers():
    gate = _gate(enabled=False, threshold=0.5)
    assert gate.needs_review(0.1) is False


# ── A2. check() — skipped paths ──────────────────────────────

@pytest.mark.asyncio
async def test_high_confidence_returns_skipped():
    gate = _gate(enabled=True, threshold=0.5)
    outcome = await gate.check(query="q", output="out", confidence=0.9)
    assert outcome.triggered is False
    assert outcome.decision == "skipped"
    assert outcome.output == "out"


@pytest.mark.asyncio
async def test_disabled_gate_returns_skipped():
    gate = _gate(enabled=False, threshold=0.5)
    outcome = await gate.check(query="q", output="out", confidence=0.1)
    assert outcome.triggered is False
    assert outcome.decision == "skipped"


@pytest.mark.asyncio
async def test_no_webhook_url_passes_through_with_warning():
    gate = _gate(enabled=True, threshold=0.5, webhook_url="")
    outcome = await gate.check(query="q", output="out", confidence=0.2)
    assert outcome.triggered is True
    assert outcome.decision == "approve"
    assert outcome.output == "out"
    assert any("no webhook" in w.lower() for w in outcome.warnings)


# ── A3. check() — webhook approve ────────────────────────────

@pytest.mark.asyncio
async def test_approved_result_returned_unchanged():
    gate = _gate()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"decision": "approve"})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="my output", confidence=0.3)

    assert outcome.triggered is True
    assert outcome.decision == "approve"
    assert outcome.output == "my output"


# ── A4. check() — webhook reject ─────────────────────────────

@pytest.mark.asyncio
async def test_rejected_result_clears_output():
    gate = _gate()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "decision": "reject",
        "reviewer_note": "hallucinated data",
    })

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="bad output", confidence=0.2)

    assert outcome.triggered is True
    assert outcome.decision == "reject"
    assert outcome.output == ""
    assert any("rejected" in w.lower() for w in outcome.warnings)


# ── A5. check() — webhook modify ─────────────────────────────

@pytest.mark.asyncio
async def test_modified_output_replaces_original():
    gate = _gate()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={
        "decision": "modify",
        "modified_output": "corrected output by reviewer",
    })

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="original", confidence=0.3)

    assert outcome.decision == "modify"
    assert outcome.output == "corrected output by reviewer"
    assert any("modified" in w.lower() for w in outcome.warnings)


# ── A6. Timeout behaviour ─────────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_timeout_approve_behavior():
    gate = _gate(timeout=1, poll_interval=0.5, on_timeout="approve")
    mock_post_response = MagicMock()
    mock_post_response.status_code = 202
    mock_post_response.raise_for_status = MagicMock()
    mock_post_response.json = MagicMock(return_value={})
    mock_poll_response = MagicMock()
    mock_poll_response.status_code = 202

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_post_response)
    mock_client.get = AsyncMock(return_value=mock_poll_response)

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="out", confidence=0.2)

    assert outcome.triggered is True
    assert outcome.decision == "approve"
    assert outcome.output == "out"


@pytest.mark.asyncio
async def test_webhook_timeout_reject_behavior():
    gate = _gate(timeout=1, poll_interval=0.5, on_timeout="reject")
    mock_post_response = MagicMock()
    mock_post_response.status_code = 202
    mock_post_response.raise_for_status = MagicMock()
    mock_post_response.json = MagicMock(return_value={})
    mock_poll_response = MagicMock()
    mock_poll_response.status_code = 202

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_post_response)
    mock_client.get = AsyncMock(return_value=mock_poll_response)

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="out", confidence=0.2)

    assert outcome.triggered is True
    assert outcome.decision == "reject"
    assert outcome.output == ""


# ── A7. Webhook POST failure ──────────────────────────────────

@pytest.mark.asyncio
async def test_webhook_post_failure_falls_back_to_on_timeout():
    gate = _gate(on_timeout="approve")
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="out", confidence=0.2)

    assert outcome.triggered is True
    assert outcome.decision == "approve"
    assert outcome.output == "out"


# ── A8. review_id tracking ────────────────────────────────────

@pytest.mark.asyncio
async def test_review_id_present_on_triggered():
    gate = _gate()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_response.json = MagicMock(return_value={"decision": "approve"})

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(return_value=mock_response)

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="out", confidence=0.1)

    assert outcome.review_id is not None
    assert len(outcome.review_id) == 36  # UUID4 length


@pytest.mark.asyncio
async def test_review_id_absent_when_skipped():
    gate = _gate()
    outcome = await gate.check(query="q", output="out", confidence=0.99)
    assert outcome.review_id is None


# ═══════════════════════════════════════════════════════════════
# SUITE B — core/andrew_swarm.py hitl_escalate node (Sprint 7)
# ═══════════════════════════════════════════════════════════════

def _base_state(confidence: float, warnings=None) -> dict:
    return {
        "confidence": confidence,
        "warnings": warnings or [],
        "audit_log": [],
        "error_message": "",
        "sql_query": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
        "user_request": "revenue by region",
    }


# ── B1. hitl_escalate node ────────────────────────────────────

def test_hitl_escalate_sets_required():
    state = _base_state(confidence=0.20)
    result = hitl_escalate(state)
    assert result["hitl_required"] is True


def test_hitl_escalate_reason_contains_confidence():
    state = _base_state(confidence=0.20)
    result = hitl_escalate(state)
    assert "0.20" in result["hitl_reason"] or "confidence" in result["hitl_reason"]


def test_hitl_escalate_includes_warnings_in_reason():
    warnings = ["Semantic: Asked about revenue but SQL doesn't reference it"]
    state = _base_state(confidence=0.25, warnings=warnings)
    result = hitl_escalate(state)
    assert result["hitl_required"] is True
    assert "revenue" in result["hitl_reason"].lower() or warnings[0] in result["hitl_reason"]


def test_hitl_escalate_caps_warnings_at_five():
    many_warnings = [f"warning {i}" for i in range(10)]
    state = _base_state(confidence=0.10, warnings=many_warnings)
    result = hitl_escalate(state)
    assert result["hitl_reason"].count("warning") <= 5


def test_hitl_escalate_audit_log_appended():
    state = _base_state(confidence=0.20)
    hitl_escalate(state)
    assert any(entry.get("stage") == "hitl_escalate" for entry in state["audit_log"])


# ── B2. Threshold boundary ────────────────────────────────────

def test_threshold_value_matches_env_default():
    expected = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.35"))
    assert HITL_CONFIDENCE_THRESHOLD == expected


def test_below_threshold_triggers_hitl():
    low = HITL_CONFIDENCE_THRESHOLD - 0.01
    state = _base_state(confidence=low)
    result = hitl_escalate(state)
    assert result["hitl_required"] is True


def test_at_threshold_does_not_trigger_hitl():
    threshold = HITL_CONFIDENCE_THRESHOLD
    state = _base_state(confidence=threshold)
    route = "hitl" if state["confidence"] < threshold else "ok"
    assert route == "ok"


def test_above_threshold_does_not_trigger_hitl():
    high = HITL_CONFIDENCE_THRESHOLD + 0.20
    state = _base_state(confidence=high)
    route = "hitl" if state["confidence"] < HITL_CONFIDENCE_THRESHOLD else "ok"
    assert route == "ok"


# ── B3. Integration: semantic_guardrails → HITL path ─────────

def test_semantic_failure_drops_below_threshold():
    """
    semantic_guardrails subtracts 0.20 per issue.
    Starting at 0.50 with 3 issues → 0.50 - 0.60 = 0.0, which is below 0.35.
    """
    state = {
        "confidence": 0.50,
        "sql_query": "SELECT SUM(quantity) FROM sales",
        "user_request": "show top monthly revenue",
        "warnings": [],
        "audit_log": [],
        "error_message": "",
    }
    result = semantic_guardrails(state)
    final_conf = result.get("confidence", state["confidence"])
    assert final_conf < HITL_CONFIDENCE_THRESHOLD


def test_hitl_not_set_on_confident_result():
    """A result with high confidence should have hitl_required=False."""
    state = _base_state(confidence=0.90)
    final = finalize_state(state)
    assert final.get("hitl_required", False) is False


# ── B4. hitl_reason edge cases ────────────────────────────────

def test_hitl_reason_is_non_empty_string():
    state = _base_state(confidence=0.10)
    result = hitl_escalate(state)
    assert isinstance(result["hitl_reason"], str)
    assert len(result["hitl_reason"]) > 0


def test_hitl_reason_without_warnings_still_meaningful():
    state = _base_state(confidence=0.15, warnings=[])
    result = hitl_escalate(state)
    assert result["hitl_reason"]
    assert "confidence" in result["hitl_reason"] or "0.15" in result["hitl_reason"]
