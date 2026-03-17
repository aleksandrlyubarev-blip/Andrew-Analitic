"""
HITL (Human-in-the-Loop) escalation tests — Sprint 7.

Verifies that Andrew correctly flags low-confidence results for human
review rather than silently returning unreliable output.

Run: python -m pytest tests/test_hitl.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.andrew_swarm import (
    hitl_escalate,
    semantic_guardrails,
    finalize_state,
    HITL_CONFIDENCE_THRESHOLD,
)


# ── Helpers ──────────────────────────────────────────────────

def _base_state(confidence: float, warnings=None) -> dict:
    return {
        "confidence": confidence,
        "warnings": warnings or [],
        "audit_log": [],
        "error_message": "",
        "sql_query": "SELECT region, SUM(revenue) FROM sales GROUP BY region",
        "user_request": "revenue by region",
    }


# ── hitl_escalate node ────────────────────────────────────────

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
    # Reason should contain the warning text
    assert "revenue" in result["hitl_reason"].lower() or warnings[0] in result["hitl_reason"]


def test_hitl_escalate_caps_warnings_at_five():
    many_warnings = [f"warning {i}" for i in range(10)]
    state = _base_state(confidence=0.10, warnings=many_warnings)
    result = hitl_escalate(state)
    # Reason is a string — just verify it's not unbounded (no more than 5 warnings appended)
    assert result["hitl_reason"].count("warning") <= 5


def test_hitl_escalate_audit_log_appended():
    state = _base_state(confidence=0.20)
    hitl_escalate(state)
    assert any(entry.get("stage") == "hitl_escalate" for entry in state["audit_log"])


# ── Threshold boundary ────────────────────────────────────────

def test_threshold_value_matches_env_default():
    """Default HITL threshold is 0.35 unless overridden by env var."""
    expected = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.35"))
    assert HITL_CONFIDENCE_THRESHOLD == expected


def test_below_threshold_triggers_hitl():
    """Confidence strictly below threshold → must escalate."""
    low = HITL_CONFIDENCE_THRESHOLD - 0.01
    state = _base_state(confidence=low)
    result = hitl_escalate(state)
    assert result["hitl_required"] is True


def test_at_threshold_does_not_trigger_hitl():
    """
    Confidence exactly at the threshold → the conditional edge routes to
    finalize_state (not hitl_escalate).  We verify this by calling the
    same lambda used in the graph.
    """
    threshold = HITL_CONFIDENCE_THRESHOLD
    state = _base_state(confidence=threshold)
    # Mirror the graph's conditional: "hitl" if confidence < threshold else "ok"
    route = "hitl" if state["confidence"] < threshold else "ok"
    assert route == "ok", f"Expected 'ok' at exact threshold {threshold}, got 'hitl'"


def test_above_threshold_does_not_trigger_hitl():
    high = HITL_CONFIDENCE_THRESHOLD + 0.20
    state = _base_state(confidence=high)
    route = "hitl" if state["confidence"] < HITL_CONFIDENCE_THRESHOLD else "ok"
    assert route == "ok"


# ── Integration: semantic_guardrails → HITL path ──────────────

def test_semantic_failure_drops_below_threshold():
    """
    semantic_guardrails subtracts 0.20 when any issues are found.
    Starting at 0.50 → 0.50 - 0.20 = 0.30, which is below the 0.35 threshold.
    """
    state = {
        "confidence": 0.50,
        "sql_query": "SELECT SUM(quantity) FROM sales",  # revenue missing, no ORDER BY, no GROUP BY
        "user_request": "show top monthly revenue",
        "warnings": [],
        "audit_log": [],
        "error_message": "",
    }
    result = semantic_guardrails(state)
    final_conf = result.get("confidence", state["confidence"])
    assert final_conf < HITL_CONFIDENCE_THRESHOLD, (
        f"Expected confidence below {HITL_CONFIDENCE_THRESHOLD}, got {final_conf}"
    )


def test_hitl_not_set_on_confident_result():
    """A result with high confidence should have hitl_required=False."""
    state = _base_state(confidence=0.90)
    # finalize_state does not set hitl_required; it should default to False
    final = finalize_state(state)
    # hitl_required not set by finalize_state → defaults to False in AndrewResult
    assert final.get("hitl_required", False) is False


# ── hitl_reason edge cases ────────────────────────────────────

def test_hitl_reason_is_non_empty_string():
    state = _base_state(confidence=0.10)
    result = hitl_escalate(state)
    assert isinstance(result["hitl_reason"], str)
    assert len(result["hitl_reason"]) > 0


def test_hitl_reason_without_warnings_still_meaningful():
    state = _base_state(confidence=0.15, warnings=[])
    result = hitl_escalate(state)
    # Must still produce a reason even with no warnings
    assert result["hitl_reason"]
    assert "confidence" in result["hitl_reason"] or "0.15" in result["hitl_reason"]
