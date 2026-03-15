"""
HITL gate tests — no live webhook required.

All tests mock httpx.AsyncClient to control webhook responses.
They run without any external services or API keys.
"""

import asyncio
import httpx
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from bridge.hitl import HitlConfig, HitlGate, HitlOutcome


# ============================================================
# Helpers
# ============================================================

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


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ============================================================
# 1. needs_review() — pure logic, no I/O
# ============================================================

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


# ============================================================
# 2. check() — skipped paths
# ============================================================

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


# ============================================================
# 3. check() — webhook approve (synchronous decision)
# ============================================================

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


# ============================================================
# 4. check() — webhook reject
# ============================================================

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


# ============================================================
# 5. check() — webhook modify
# ============================================================

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


# ============================================================
# 6. Timeout behaviour
# ============================================================

@pytest.mark.asyncio
async def test_webhook_timeout_approve_behavior():
    """When webhook times out and on_timeout=approve, result passes through."""
    gate = _gate(timeout=1, poll_interval=0.5, on_timeout="approve")

    mock_post_response = MagicMock()
    mock_post_response.status_code = 202   # Accepted, not yet decided
    mock_post_response.raise_for_status = MagicMock()
    mock_post_response.json = MagicMock(return_value={})  # No decision field

    mock_poll_response = MagicMock()
    mock_poll_response.status_code = 202  # Still pending

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
    """When webhook times out and on_timeout=reject, output is cleared."""
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


# ============================================================
# 7. Webhook POST failure
# ============================================================

@pytest.mark.asyncio
async def test_webhook_post_failure_falls_back_to_on_timeout():
    """If the POST itself raises an exception, use on_timeout behaviour."""
    gate = _gate(on_timeout="approve")

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)
    mock_client.post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))

    with patch("bridge.hitl.httpx.AsyncClient", return_value=mock_client):
        outcome = await gate.check(query="q", output="out", confidence=0.2)

    # on_timeout=approve → should pass through
    assert outcome.triggered is True
    assert outcome.decision == "approve"
    assert outcome.output == "out"


# ============================================================
# 8. review_id is set when triggered
# ============================================================

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
