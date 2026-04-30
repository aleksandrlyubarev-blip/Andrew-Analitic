"""
tests/test_llm_telemetry.py
===========================
Tests for the LiteLLM success/failure callback that emits per-call telemetry.

Strategy: stub the ``litellm`` module via ``sys.modules`` to expose mutable
``success_callback`` / ``failure_callback`` lists. Then either (a) verify
install() pushes our handler onto those lists, or (b) call our handlers
directly with synthetic LiteLLM responses and assert on the captured log
records.

Run: python -m pytest tests/test_llm_telemetry.py -v
"""

from __future__ import annotations

import datetime as dt
import logging
import sys
import types

import pytest


@pytest.fixture(autouse=True)
def fake_litellm(monkeypatch):
    """Inject a fake ``litellm`` module with empty callback lists."""
    fake = types.ModuleType("litellm")
    fake.success_callback = []
    fake.failure_callback = []
    monkeypatch.setitem(sys.modules, "litellm", fake)

    # Force a clean state on the telemetry module: drop and re-import so
    # _INSTALLED resets between tests.
    monkeypatch.delitem(sys.modules, "bridge.llm_telemetry", raising=False)
    yield fake
    # Best-effort clean up after the test.
    monkeypatch.delitem(sys.modules, "bridge.llm_telemetry", raising=False)


@pytest.fixture
def telemetry(fake_litellm):
    import bridge.llm_telemetry as m
    yield m


# ── install() ────────────────────────────────────────────────────────────────

def test_install_appends_callbacks(telemetry, fake_litellm):
    assert telemetry.install() is True
    assert telemetry._on_success in fake_litellm.success_callback
    assert telemetry._on_failure in fake_litellm.failure_callback


def test_install_is_idempotent(telemetry, fake_litellm):
    telemetry.install()
    telemetry.install()
    telemetry.install()
    # Each callback registered exactly once.
    assert fake_litellm.success_callback.count(telemetry._on_success) == 1
    assert fake_litellm.failure_callback.count(telemetry._on_failure) == 1


def test_install_returns_false_when_disabled(telemetry, fake_litellm, monkeypatch):
    monkeypatch.setenv("LLM_TELEMETRY_ENABLED", "false")
    # Re-import to pick up env-driven kill switch on a fresh module.
    sys.modules.pop("bridge.llm_telemetry", None)
    import bridge.llm_telemetry as m2
    assert m2.install() is False
    assert m2._on_success not in fake_litellm.success_callback


def test_install_returns_false_when_litellm_missing(monkeypatch, caplog):
    # Drop the fake litellm so import will fail.
    monkeypatch.setitem(sys.modules, "litellm", None)
    sys.modules.pop("bridge.llm_telemetry", None)
    import bridge.llm_telemetry as m
    with caplog.at_level(logging.WARNING, logger="bridge.llm_telemetry"):
        assert m.install() is False
    assert any("not installed" in r.getMessage() for r in caplog.records)


def test_install_does_not_clobber_other_callbacks(telemetry, fake_litellm):
    sentinel = lambda *a, **k: None  # noqa: E731
    fake_litellm.success_callback.append(sentinel)
    telemetry.install()
    # Other observability hooks survive.
    assert sentinel in fake_litellm.success_callback
    assert telemetry._on_success in fake_litellm.success_callback


def test_uninstall_removes_callbacks(telemetry, fake_litellm):
    telemetry.install()
    assert telemetry._on_success in fake_litellm.success_callback
    telemetry.uninstall()
    assert telemetry._on_success not in fake_litellm.success_callback
    assert telemetry._on_failure not in fake_litellm.failure_callback


# ── _on_success: structured log content ──────────────────────────────────────

def _records_for(caplog) -> list[logging.LogRecord]:
    """Return only telemetry records, in order."""
    return [r for r in caplog.records if r.name == "bridge.llm_telemetry"]


def test_on_success_logs_full_payload(telemetry, caplog):
    response = {
        "id": "chatcmpl-xyz",
        "model": "gpt-4o-mini",
        "response_cost": 0.0123,
        "usage": {"prompt_tokens": 250, "completion_tokens": 80},
    }
    start = dt.datetime(2026, 4, 29, 12, 0, 0)
    end = dt.datetime(2026, 4, 29, 12, 0, 1, 250000)  # +1.25s

    with caplog.at_level(logging.INFO, logger="bridge.llm_telemetry"):
        telemetry._on_success({}, response, start, end)
    records = _records_for(caplog)
    assert len(records) == 1
    r = records[0]
    assert r.event == "llm_call"
    assert r.model == "gpt-4o-mini"
    assert r.tokens_in == 250
    assert r.tokens_out == 80
    assert r.cost_usd == 0.0123
    assert r.latency_ms == 1250
    assert r.request_id == "chatcmpl-xyz"


def test_on_success_handles_object_response(telemetry, caplog):
    """LiteLLM sometimes hands back a ModelResponse-shaped object, not a dict."""
    class _Usage:
        prompt_tokens = 100
        completion_tokens = 50
    class _Resp:
        id = "x"
        model = "claude-sonnet"
        response_cost = 0.005
        usage = _Usage()

    with caplog.at_level(logging.INFO, logger="bridge.llm_telemetry"):
        telemetry._on_success({}, _Resp(), 0.0, 0.5)
    records = _records_for(caplog)
    assert len(records) == 1
    r = records[0]
    assert r.model == "claude-sonnet"
    assert r.tokens_in == 100
    assert r.tokens_out == 50
    assert r.cost_usd == 0.005
    assert r.latency_ms == 500


def test_on_success_handles_missing_usage(telemetry, caplog):
    response = {"model": "grok-4", "response_cost": 0.0}
    with caplog.at_level(logging.INFO, logger="bridge.llm_telemetry"):
        telemetry._on_success({}, response, None, None)
    records = _records_for(caplog)
    assert len(records) == 1
    r = records[0]
    assert r.tokens_in == 0
    assert r.tokens_out == 0
    assert r.cost_usd == 0.0
    assert r.latency_ms == 0


def test_on_success_falls_back_to_kwargs_cost(telemetry, caplog):
    """Older LiteLLM passes response_cost via kwargs not the response."""
    kwargs = {"model": "gpt-4o-mini", "response_cost": 0.07}
    response = {"model": "gpt-4o-mini", "usage": {"prompt_tokens": 1, "completion_tokens": 1}}
    with caplog.at_level(logging.INFO, logger="bridge.llm_telemetry"):
        telemetry._on_success(kwargs, response, 0.0, 0.0)
    records = _records_for(caplog)
    assert records[0].cost_usd == 0.07


def test_on_success_swallows_callback_errors(telemetry):
    """A malformed response must not raise (LiteLLM would suppress us anyway,
    but we don't want to pollute caller traces with telemetry bugs)."""
    class _Boom:
        def __getattr__(self, name):
            raise RuntimeError("synthetic")
    telemetry._on_success({}, _Boom(), 0.0, 0.0)  # must not raise


# ── _on_failure ──────────────────────────────────────────────────────────────

def test_on_failure_logs_error(telemetry, caplog):
    err = ConnectionError("timed out after 30s")
    kwargs = {"model": "gpt-4o-mini", "exception": err}
    with caplog.at_level(logging.WARNING, logger="bridge.llm_telemetry"):
        telemetry._on_failure(kwargs, None, 0.0, 0.5)
    records = _records_for(caplog)
    assert len(records) == 1
    r = records[0]
    assert r.event == "llm_error"
    assert r.model == "gpt-4o-mini"
    assert "ConnectionError" in r.error
    assert "timed out" in r.error
    assert r.latency_ms == 500


def test_on_failure_truncates_long_errors(telemetry, caplog):
    huge = "x" * 10_000
    kwargs = {"model": "m", "exception": RuntimeError(huge)}
    with caplog.at_level(logging.WARNING, logger="bridge.llm_telemetry"):
        telemetry._on_failure(kwargs, None, 0.0, 0.0)
    assert len(_records_for(caplog)[0].error) <= 500


def test_on_failure_handles_string_exception(telemetry, caplog):
    """Older LiteLLM sometimes passes the error as a string."""
    kwargs = {"model": "m", "exception": "something went wrong"}
    with caplog.at_level(logging.WARNING, logger="bridge.llm_telemetry"):
        telemetry._on_failure(kwargs, None, 0.0, 0.0)
    assert "something went wrong" in _records_for(caplog)[0].error


def test_on_failure_swallows_callback_errors(telemetry):
    """Malformed kwargs/response must not raise."""
    telemetry._on_failure(None, None, None, None)  # all-None → must not raise


# ── helpers ──────────────────────────────────────────────────────────────────

def test_latency_ms_supports_floats(telemetry):
    assert telemetry._latency_ms(0.0, 1.5) == 1500


def test_latency_ms_supports_datetimes(telemetry):
    a = dt.datetime(2026, 1, 1, 0, 0, 0)
    b = dt.datetime(2026, 1, 1, 0, 0, 0, 750_000)
    assert telemetry._latency_ms(a, b) == 750


def test_latency_ms_returns_zero_for_none(telemetry):
    assert telemetry._latency_ms(None, None) == 0
    assert telemetry._latency_ms(0.0, None) == 0


def test_latency_ms_returns_zero_for_garbage(telemetry):
    assert telemetry._latency_ms("not", "numeric") == 0
