"""
Rate-limiting tests — Sprint 6 security hardening.

Tests the SlidingWindowRateLimiter and _parse_rate_spec helpers imported
directly from the bridge module.  All tests are offline and deterministic
(no FastAPI test client needed).

Run: python -m pytest tests/test_rate_limit.py -v
"""
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from bridge.moltis_bridge import (
    SlidingWindowRateLimiter,
    _parse_rate_spec,
    _build_limiter,
    _limiter_analyze,
    _limiter_educate,
    _limiter_webhook,
)


# ── _parse_rate_spec ─────────────────────────────────────────────────────────

def test_parse_valid_spec():
    req, win = _parse_rate_spec("10/60", 0, 0)
    assert req == 10
    assert win == 60


def test_parse_large_values():
    req, win = _parse_rate_spec("1000/3600", 0, 0)
    assert req == 1000
    assert win == 3600


def test_parse_invalid_spec_uses_defaults():
    req, win = _parse_rate_spec("not-a-spec", 5, 30)
    assert req == 5
    assert win == 30


def test_parse_empty_string_uses_defaults():
    req, win = _parse_rate_spec("", 7, 45)
    assert req == 7
    assert win == 45


def test_parse_zero_requests_disables():
    req, win = _parse_rate_spec("0/60", 10, 60)
    assert req == 0
    assert win == 60


# ── SlidingWindowRateLimiter: core allow/deny logic ──────────────────────────

def test_first_request_always_allowed():
    lim = SlidingWindowRateLimiter(5, 60)
    allowed, retry = lim.is_allowed("ip-1")
    assert allowed is True
    assert retry == 0


def test_requests_up_to_limit_all_allowed():
    lim = SlidingWindowRateLimiter(5, 60)
    for _ in range(5):
        allowed, _ = lim.is_allowed("ip-1")
        assert allowed is True


def test_request_beyond_limit_denied():
    lim = SlidingWindowRateLimiter(3, 60)
    for _ in range(3):
        lim.is_allowed("ip-1")
    allowed, retry_after = lim.is_allowed("ip-1")
    assert allowed is False
    assert retry_after >= 1


def test_denied_retry_after_positive():
    lim = SlidingWindowRateLimiter(1, 60)
    lim.is_allowed("ip-1")                     # consume the one slot
    allowed, retry_after = lim.is_allowed("ip-1")
    assert not allowed
    assert 1 <= retry_after <= 61               # must be within the window


def test_per_key_isolation():
    """Different IPs have independent windows."""
    lim = SlidingWindowRateLimiter(2, 60)
    lim.is_allowed("ip-A")
    lim.is_allowed("ip-A")
    denied, _ = lim.is_allowed("ip-A")
    assert denied is False

    # ip-B is untouched — should still be allowed
    allowed, _ = lim.is_allowed("ip-B")
    assert allowed is True


def test_zero_max_requests_always_allowed():
    """max_requests=0 disables the limiter for that endpoint."""
    lim = SlidingWindowRateLimiter(0, 60)
    for _ in range(100):
        allowed, _ = lim.is_allowed("ip-1")
        assert allowed is True


def test_window_expiry_resets_count():
    """Timestamps older than window_seconds are evicted, allowing new requests."""
    lim = SlidingWindowRateLimiter(2, 1)   # 2 req / 1 second window

    lim.is_allowed("ip-1")
    lim.is_allowed("ip-1")
    denied, _ = lim.is_allowed("ip-1")
    assert denied is False                  # third request denied

    time.sleep(1.05)                        # wait for window to expire

    allowed, _ = lim.is_allowed("ip-1")
    assert allowed is True                  # slot freed


def test_reset_clears_key():
    lim = SlidingWindowRateLimiter(2, 60)
    lim.is_allowed("ip-1")
    lim.is_allowed("ip-1")
    lim.reset("ip-1")
    # After reset, key should have a fresh slate
    allowed, _ = lim.is_allowed("ip-1")
    assert allowed is True


def test_reset_nonexistent_key_is_noop():
    lim = SlidingWindowRateLimiter(5, 60)
    lim.reset("never-seen")                 # must not raise


def test_repr_contains_config():
    lim = SlidingWindowRateLimiter(10, 60)
    r = repr(lim)
    assert "10" in r
    assert "60" in r


# ── Limiter with 1-request window ────────────────────────────────────────────

def test_single_request_window():
    """max_requests=1: second request in same window must be denied."""
    lim = SlidingWindowRateLimiter(1, 60)
    a1, _ = lim.is_allowed("solo")
    a2, retry = lim.is_allowed("solo")
    assert a1 is True
    assert a2 is False
    assert retry >= 1


# ── Module-level limiter defaults ────────────────────────────────────────────

def test_analyze_limiter_default_max():
    """Default: 10 requests per 60 seconds."""
    assert _limiter_analyze.max_requests == 10
    assert _limiter_analyze.window_seconds == 60


def test_educate_limiter_default_max():
    """Default: 20 requests per 60 seconds."""
    assert _limiter_educate.max_requests == 20
    assert _limiter_educate.window_seconds == 60


def test_webhook_limiter_default_max():
    """Default: 30 requests per 60 seconds."""
    assert _limiter_webhook.max_requests == 30
    assert _limiter_webhook.window_seconds == 60


def test_env_override_analyze(monkeypatch):
    """RATE_LIMIT_ANALYZE env var overrides the default."""
    monkeypatch.setenv("RATE_LIMIT_ANALYZE", "5/30")
    lim = _build_limiter("RATE_LIMIT_ANALYZE", 10, 60)
    assert lim.max_requests == 5
    assert lim.window_seconds == 30


def test_env_override_disabled(monkeypatch):
    """Setting limit to 0 disables the endpoint limiter."""
    monkeypatch.setenv("RATE_LIMIT_EDUCATE", "0/60")
    lim = _build_limiter("RATE_LIMIT_EDUCATE", 20, 60)
    assert lim.max_requests == 0
    for _ in range(50):
        allowed, _ = lim.is_allowed("heavy-user")
        assert allowed is True


# ── Thread-safety smoke test ─────────────────────────────────────────────────

def test_thread_safety():
    """Concurrent is_allowed calls from multiple threads should not panic."""
    import threading as _threading

    lim = SlidingWindowRateLimiter(100, 60)
    errors = []

    def worker():
        try:
            for _ in range(20):
                lim.is_allowed("shared-ip")
        except Exception as exc:
            errors.append(exc)

    threads = [_threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Thread-safety errors: {errors}"
