"""
tests/test_request_log.py
=========================
Tests for the RequestLogMiddleware that writes one row per request to
``andrew.request_log``.

Strategy:
  • For URL/scheme inspection and skip-paths: drive the middleware through
    a tiny FastAPI app, no DB stub needed.
  • For the actual INSERT path: stub the ``psycopg`` module via
    ``sys.modules`` so we exercise the SQL formatting without needing a
    live Postgres. The fake records every connect / execute call.

Run: python -m pytest tests/test_request_log.py -v
"""

from __future__ import annotations

import asyncio
import os
import sys
import types
from typing import Any
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from bridge.request_log import (
    RequestLogMiddleware,
    SKIP_PATHS,
    _looks_like_postgres,
    _to_libpq,
    middleware_kwargs_from_env,
)


# ── URL helpers ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("url, expected", [
    ("",                                          False),
    ("sqlite:///andrew.db",                       False),
    ("mysql://x",                                 False),
    ("postgresql://u:p@h/db",                     True),
    ("postgres://u:p@h/db",                       True),
    ("postgresql+psycopg://u@/db",                True),
    ("POSTGRESQL://u:p@h/db",                     True),
])
def test_looks_like_postgres(url, expected):
    assert _looks_like_postgres(url) is expected


def test_to_libpq_strips_driver_suffix():
    assert _to_libpq("postgresql+psycopg://u@/db") == "postgresql://u@/db"


def test_to_libpq_passthrough_when_no_driver():
    assert _to_libpq("postgresql://u@/db") == "postgresql://u@/db"


def test_to_libpq_passthrough_no_scheme():
    assert _to_libpq("not-a-url") == "not-a-url"


# ── middleware_kwargs_from_env ───────────────────────────────────────────────

def test_kwargs_from_env_unset(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("REQUEST_LOG_ENABLED", raising=False)
    assert middleware_kwargs_from_env() is None


def test_kwargs_from_env_sqlite_returns_none(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "sqlite:///andrew.db")
    monkeypatch.delenv("REQUEST_LOG_ENABLED", raising=False)
    assert middleware_kwargs_from_env() is None


def test_kwargs_from_env_postgres(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.delenv("REQUEST_LOG_ENABLED", raising=False)
    kwargs = middleware_kwargs_from_env()
    assert kwargs == {"db_url": "postgresql://u:p@h/db"}


def test_kwargs_from_env_disabled_via_kill_switch(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/db")
    monkeypatch.setenv("REQUEST_LOG_ENABLED", "false")
    assert middleware_kwargs_from_env() is None


# ── No-op behaviour (no DB writes) ───────────────────────────────────────────

def _build_app(db_url: str | None) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestLogMiddleware, db_url=db_url)

    @app.get("/health")
    async def _h():
        return {"status": "ok"}

    @app.post("/analyze")
    async def _a(request: Request):
        # Mimic the real handler: surface cost on request.state.
        request.state.cost_usd = 0.0042
        request.state.model = "andrew"
        return {"ok": True}

    @app.get("/boom")
    async def _b():
        raise RuntimeError("synthetic")

    return app


def test_disabled_when_db_url_empty():
    app = _build_app(db_url="")
    client = TestClient(app)
    # Just verify nothing crashes and the response goes through.
    assert client.get("/health").status_code == 200
    assert client.post("/analyze").status_code == 200


def test_disabled_when_db_url_sqlite():
    app = _build_app(db_url="sqlite:///andrew.db")
    client = TestClient(app)
    assert client.post("/analyze").status_code == 200


# ── Stubbed psycopg: capture INSERT payloads ─────────────────────────────────

class _FakeCursor:
    def __init__(self, log: list[tuple[str, dict]]):
        self._log = log

    async def execute(self, sql: str, params: dict):
        self._log.append((sql, params))

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeConnection:
    def __init__(self, log: list[tuple[str, dict]]):
        self._log = log
        self.committed = False

    def cursor(self):
        return _FakeCursor(self._log)

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeAsyncConnection:
    last_url: str | None = None
    log: list[tuple[str, dict]] = []
    raise_on_connect: Exception | None = None

    @classmethod
    async def connect(cls, url: str):
        cls.last_url = url
        if cls.raise_on_connect is not None:
            raise cls.raise_on_connect
        return _FakeConnection(cls.log)


@pytest.fixture
def fake_psycopg(monkeypatch):
    """Inject a fake psycopg module that records connects + executes."""
    _FakeAsyncConnection.last_url = None
    _FakeAsyncConnection.log = []
    _FakeAsyncConnection.raise_on_connect = None

    fake = types.ModuleType("psycopg")
    fake.AsyncConnection = _FakeAsyncConnection  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", fake)
    return _FakeAsyncConnection


# ── End-to-end through the middleware ───────────────────────────────────────

async def _run_and_drain(client: TestClient, *args, **kwargs):
    """TestClient.* + give the event loop a tick for asyncio.create_task."""
    response = client.request(*args, **kwargs)
    # The middleware fires its DB write via asyncio.create_task; under
    # TestClient's portal we need to yield once for it to run.
    await asyncio.sleep(0.05)
    return response


@pytest.mark.asyncio
async def test_postgres_url_inserts_row(fake_psycopg):
    app = _build_app(db_url="postgresql://u:p@h/db")
    client = TestClient(app)

    response = await _run_and_drain(client, "POST", "/analyze", json={})
    assert response.status_code == 200

    assert fake_psycopg.last_url == "postgresql://u:p@h/db"
    assert len(fake_psycopg.log) == 1
    sql, params = fake_psycopg.log[0]
    assert "INSERT INTO andrew.request_log" in sql
    assert params["route"] == "/analyze"
    assert params["status_code"] == 200
    assert params["cost_usd"] == 0.0042
    assert params["model"] == "andrew"
    assert params["latency_ms"] >= 0
    assert params["error"] is None


@pytest.mark.asyncio
async def test_skip_paths_not_logged(fake_psycopg):
    app = _build_app(db_url="postgresql://u:p@h/db")
    # Add a /healthz route so we can verify the probe path is also skipped.
    @app.get("/healthz")
    async def _hz():
        return {"status": "ok"}
    client = TestClient(app)

    response = await _run_and_drain(client, "GET", "/health")
    assert response.status_code == 200
    response = await _run_and_drain(client, "GET", "/healthz")
    assert response.status_code == 200
    assert fake_psycopg.log == []


@pytest.mark.asyncio
async def test_500_logs_error_and_propagates(fake_psycopg):
    app = _build_app(db_url="postgresql://u:p@h/db")
    client = TestClient(app, raise_server_exceptions=False)

    response = await _run_and_drain(client, "GET", "/boom")
    assert response.status_code == 500

    assert len(fake_psycopg.log) == 1
    sql, params = fake_psycopg.log[0]
    assert params["status_code"] == 500
    assert params["error"] is not None
    assert "RuntimeError" in params["error"]


@pytest.mark.asyncio
async def test_db_failure_does_not_break_response(fake_psycopg):
    fake_psycopg.raise_on_connect = ConnectionError("DB unreachable")

    app = _build_app(db_url="postgresql://u:p@h/db")
    client = TestClient(app)

    response = await _run_and_drain(client, "POST", "/analyze", json={})
    # Caller still gets their 200 even though the DB write failed.
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_sqlalchemy_url_normalised_to_libpq(fake_psycopg):
    app = _build_app(db_url="postgresql+psycopg://u@/db?host=/cloudsql/x")
    client = TestClient(app)

    await _run_and_drain(client, "POST", "/analyze", json={})

    # psycopg.AsyncConnection.connect must have been called with the
    # libpq form (no +psycopg suffix).
    assert fake_psycopg.last_url == "postgresql://u@/db?host=/cloudsql/x"


@pytest.mark.asyncio
async def test_anonymous_when_user_slug_unset(fake_psycopg):
    app = _build_app(db_url="postgresql://u:p@h/db")
    client = TestClient(app)

    await _run_and_drain(client, "POST", "/analyze", json={})
    _, params = fake_psycopg.log[0]
    # _build_app's handler doesn't set request.state.user_slug, so the
    # middleware uses its default.
    assert params["user_slug"] == "anonymous"


# ── Static surface checks ────────────────────────────────────────────────────

def test_skip_paths_includes_health():
    assert "/health" in SKIP_PATHS
    assert "/openapi.json" in SKIP_PATHS
    assert "/healthz" in SKIP_PATHS, (
        "/healthz fires every 5-30s from the Cloud Run probe; logging it "
        "would dominate the request_log table"
    )
