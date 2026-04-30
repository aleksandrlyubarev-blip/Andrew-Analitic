"""
tests/test_auth.py
==================
Unit tests for the per-tester X-Api-Key middleware that gates the GCP beta.

Strategy: build a minimal FastAPI app with the middleware mounted directly,
rather than importing bridge.api (which spins up the full LangGraph bridge
plus LLM clients). Lets each scenario set its own keys without env mutation.

Run: python -m pytest tests/test_auth.py -v
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from bridge.auth import (
    BetaApiKeyMiddleware,
    PUBLIC_PATHS,
    load_beta_keys,
    middleware_kwargs_from_env,
)


def _build_app(keys: dict[str, str], **mw_kwargs) -> FastAPI:
    """Tiny app exposing /health (public) and /private (gated)."""
    app = FastAPI()
    app.add_middleware(BetaApiKeyMiddleware, keys=keys, **mw_kwargs)

    @app.get("/health")
    async def _health():
        return {"status": "ok"}

    @app.get("/private")
    async def _private(request: Request):
        return {"user_slug": getattr(request.state, "user_slug", None)}

    return app


# ── load_beta_keys ────────────────────────────────────────────────────────────

def test_load_keys_unset_returns_empty():
    assert load_beta_keys(None) == {}
    assert load_beta_keys("") == {}


def test_load_keys_invalid_json_returns_empty():
    assert load_beta_keys("not json") == {}


def test_load_keys_non_object_returns_empty():
    assert load_beta_keys('["a", "b"]') == {}


def test_load_keys_drops_non_string_entries():
    raw = '{"alex": "key1", "bad": 42, "": "key2", "ok": ""}'
    assert load_beta_keys(raw) == {"alex": "key1"}


def test_load_keys_round_trip():
    raw = '{"alex": "k1", "sergei": "k2"}'
    assert load_beta_keys(raw) == {"alex": "k1", "sergei": "k2"}


# ── middleware behaviour ──────────────────────────────────────────────────────

def test_no_keys_no_fail_closed_is_noop():
    """Local-dev mode: no BETA_API_KEYS → all routes pass without a header."""
    app = _build_app(keys={})
    client = TestClient(app)
    r = client.get("/private")
    assert r.status_code == 200
    assert r.json() == {"user_slug": "anonymous"}


def test_health_bypasses_auth_even_when_keys_set():
    """Cloud Run startup probe must reach /healthz without a key. /health is
    the rich human-facing variant; both are public."""
    app = _build_app(keys={"alex": "secret"})
    # Add a /healthz route that mirrors the real cheap probe.
    @app.get("/healthz")
    async def _hz():
        return {"status": "ok"}

    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    rz = client.get("/healthz")
    assert rz.status_code == 200


def test_missing_header_returns_401():
    app = _build_app(keys={"alex": "secret"})
    client = TestClient(app)
    r = client.get("/private")
    assert r.status_code == 401
    body = r.json()
    assert body["error"] == "missing_api_key"
    assert body["header"] == "X-Api-Key"


def test_wrong_header_returns_403():
    app = _build_app(keys={"alex": "secret"})
    client = TestClient(app)
    r = client.get("/private", headers={"X-Api-Key": "wrong"})
    assert r.status_code == 403
    assert r.json() == {"error": "invalid_api_key"}


def test_valid_header_attaches_user_slug():
    app = _build_app(keys={"alex": "secret-alex", "sergei": "secret-sergei"})
    client = TestClient(app)
    r = client.get("/private", headers={"X-Api-Key": "secret-sergei"})
    assert r.status_code == 200
    assert r.json() == {"user_slug": "sergei"}


def test_custom_header_name():
    app = _build_app(keys={"alex": "secret"}, header_name="Authorization")
    client = TestClient(app)
    # Default X-Api-Key header is now ignored.
    r = client.get("/private", headers={"X-Api-Key": "secret"})
    assert r.status_code == 401
    assert r.json()["header"] == "Authorization"
    # Authorization header is honoured.
    r2 = client.get("/private", headers={"Authorization": "secret"})
    assert r2.status_code == 200


def test_fail_closed_with_no_keys_rejects_everything():
    """Production safety net: BETA_AUTH_REQUIRED=true but keys empty."""
    app = _build_app(keys={}, fail_closed=True)
    client = TestClient(app)
    # /health still bypasses (probes don't carry keys).
    assert client.get("/health").status_code == 200
    # Everything else is rejected with 401 (no header) or 403 (any header).
    assert client.get("/private").status_code == 401
    assert client.get("/private", headers={"X-Api-Key": "anything"}).status_code == 403


def test_public_paths_constant_includes_health():
    assert "/health" in PUBLIC_PATHS
    assert "/healthz" in PUBLIC_PATHS, (
        "/healthz must bypass auth — it's the Cloud Run probe target"
    )


# ── middleware_kwargs_from_env ────────────────────────────────────────────────

def test_kwargs_from_env_unset(monkeypatch):
    monkeypatch.delenv("BETA_API_KEYS", raising=False)
    monkeypatch.delenv("BETA_AUTH_REQUIRED", raising=False)
    assert middleware_kwargs_from_env() is None


def test_kwargs_from_env_required_but_no_keys(monkeypatch):
    monkeypatch.delenv("BETA_API_KEYS", raising=False)
    monkeypatch.setenv("BETA_AUTH_REQUIRED", "true")
    kwargs = middleware_kwargs_from_env()
    assert kwargs is not None
    assert kwargs["keys"] == {}
    assert kwargs["fail_closed"] is True


def test_kwargs_from_env_with_keys(monkeypatch):
    monkeypatch.setenv("BETA_API_KEYS", '{"alex": "k1"}')
    monkeypatch.delenv("BETA_AUTH_REQUIRED", raising=False)
    kwargs = middleware_kwargs_from_env()
    assert kwargs is not None
    assert kwargs["keys"] == {"alex": "k1"}
    assert kwargs["fail_closed"] is False
    assert kwargs["header_name"] == "X-Api-Key"


def test_kwargs_from_env_custom_header(monkeypatch):
    monkeypatch.setenv("BETA_API_KEYS", '{"alex": "k1"}')
    monkeypatch.setenv("BETA_AUTH_HEADER", "X-Beta-Token")
    kwargs = middleware_kwargs_from_env()
    assert kwargs["header_name"] == "X-Beta-Token"


# ── timing-safe compare smoke check ───────────────────────────────────────────

def test_timing_safe_compare_used_for_keys():
    """Sanity check that the middleware doesn't accept prefix matches.

    A trivial == on shorter strings would match; hmac.compare_digest does not.
    """
    app = _build_app(keys={"alex": "abcdefgh"})
    client = TestClient(app)
    # Truncated key must not pass.
    assert client.get("/private", headers={"X-Api-Key": "abcd"}).status_code == 403
    # Extended key must not pass.
    assert client.get("/private", headers={"X-Api-Key": "abcdefghX"}).status_code == 403
    # Exact key passes.
    assert client.get("/private", headers={"X-Api-Key": "abcdefgh"}).status_code == 200
