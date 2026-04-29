"""
tests/test_checkpointing.py
===========================
Unit tests for the LangGraph checkpointer factory in core.checkpointing.

The factory only inspects env / args and (when Postgres is asked for) tries
to instantiate a saver. We exercise:

  • Backend resolution from the (forced, db_url) pair
  • In-memory saver round-trip (always available; bundled with langgraph)
  • Postgres saver path with the import lazily mocked, so this test runs
    even on machines without langgraph-checkpoint-postgres installed
  • URL credential redaction in log output
  • backend="none" returns None

Run: python -m pytest tests/test_checkpointing.py -v
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.checkpointing import (
    _resolve_backend,
    _safe_url,
    get_checkpointer,
)


# ── _resolve_backend ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("forced, db_url, expected", [
    ("memory",   "",                                "memory"),
    ("postgres", "",                                "postgres"),
    ("none",     "postgresql://x",                  "none"),
    ("MEMORY",   "",                                "memory"),     # case-insensitive
    ("  postgres  ", "",                            "postgres"),   # trim
    (None, "postgresql://user:pw@host/db",          "postgres"),
    (None, "postgresql+psycopg://user@/db",         "postgres"),
    (None, "postgres://user@host/db",               "postgres"),   # legacy scheme
    (None, "sqlite:///andrew.db",                   "memory"),
    (None, "",                                      "memory"),
    (None, "mysql://x",                             "memory"),     # unknown → memory
    ("garbage", "postgresql://x",                   "postgres"),   # bad forced → URL fallback
    ("garbage", "",                                 "memory"),
])
def test_resolve_backend(forced, db_url, expected):
    assert _resolve_backend(forced, db_url) == expected


# ── _safe_url ─────────────────────────────────────────────────────────────────

def test_safe_url_no_credentials():
    assert _safe_url("postgresql://localhost/db") == "postgresql://localhost/db"


def test_safe_url_strips_credentials():
    redacted = _safe_url("postgresql://alice:secret@host:5432/andrew")
    assert "alice" not in redacted
    assert "secret" not in redacted
    assert "host:5432" in redacted
    assert "andrew" in redacted


def test_safe_url_strips_password_only():
    redacted = _safe_url("postgresql://user:p%40ssw0rd@h/db")
    assert "p@ssw0rd" not in redacted
    assert "p%40ssw0rd" not in redacted


# ── get_checkpointer: explicit "none" ─────────────────────────────────────────

def test_get_checkpointer_none(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_CHECKPOINTER", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_checkpointer(backend="none") is None


def test_get_checkpointer_none_via_env(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINTER", "none")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")  # would otherwise pick postgres
    assert get_checkpointer() is None


# ── get_checkpointer: in-memory ───────────────────────────────────────────────

def test_get_checkpointer_memory_default(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_CHECKPOINTER", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    saver = get_checkpointer()
    assert saver is not None
    # In-memory saver must implement the BaseCheckpointSaver get/put interface.
    assert hasattr(saver, "get_tuple") or hasattr(saver, "aget_tuple")


def test_get_checkpointer_memory_when_url_is_sqlite(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_CHECKPOINTER", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite:///andrew.db")
    saver = get_checkpointer()
    assert saver is not None
    assert "InMemorySaver" in type(saver).__name__ or "MemorySaver" in type(saver).__name__


def test_get_checkpointer_memory_explicit_arg(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")  # ignored
    saver = get_checkpointer(backend="memory")
    assert "InMemorySaver" in type(saver).__name__ or "MemorySaver" in type(saver).__name__


# ── get_checkpointer: Postgres ────────────────────────────────────────────────

class _StubPostgresSaver:
    """Stand-in for langgraph.checkpoint.postgres.PostgresSaver."""

    def __init__(self, url):
        self.url = url

    @classmethod
    def from_conn_string(cls, url):
        return cls(url)


def test_get_checkpointer_postgres_with_stub(monkeypatch):
    """Even when langgraph-checkpoint-postgres isn't installed, the factory
    routes correctly when we inject a stub via the import path."""

    fake_module = type(sys)("langgraph.checkpoint.postgres")
    fake_module.PostgresSaver = _StubPostgresSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake_module)

    saver = get_checkpointer(backend="postgres", db_url="postgresql://u:p@h/db")
    assert isinstance(saver, _StubPostgresSaver)
    assert saver.url == "postgresql://u:p@h/db"


def test_get_checkpointer_postgres_rewrites_sqlalchemy_scheme(monkeypatch):
    """SQLAlchemy-style postgresql+psycopg:// must be normalised to libpq."""
    fake_module = type(sys)("langgraph.checkpoint.postgres")
    fake_module.PostgresSaver = _StubPostgresSaver
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", fake_module)

    saver = get_checkpointer(
        backend="postgres",
        db_url="postgresql+psycopg://andrew@/andrew?host=/cloudsql/x:y:z",
    )
    assert saver.url.startswith("postgresql://")
    assert "+psycopg" not in saver.url


def test_get_checkpointer_postgres_missing_dep_raises(monkeypatch):
    """Importing PostgresSaver fails → factory raises with a clear hint."""
    # Block the import path.
    monkeypatch.setitem(sys.modules, "langgraph.checkpoint.postgres", None)

    with pytest.raises(RuntimeError) as exc:
        get_checkpointer(backend="postgres", db_url="postgresql://x")
    assert "langgraph-checkpoint-postgres" in str(exc.value)


# ── env precedence ────────────────────────────────────────────────────────────

def test_explicit_args_override_env(monkeypatch):
    monkeypatch.setenv("LANGGRAPH_CHECKPOINTER", "postgres")
    monkeypatch.setenv("DATABASE_URL", "postgresql://x")
    saver = get_checkpointer(backend="memory", db_url="")
    assert "InMemorySaver" in type(saver).__name__ or "MemorySaver" in type(saver).__name__


def test_unknown_forced_falls_back_to_url_inspection(monkeypatch):
    monkeypatch.delenv("LANGGRAPH_CHECKPOINTER", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    saver = get_checkpointer(backend="garbage", db_url="sqlite:///x.db")
    # garbage forced → ignored → URL says sqlite → memory
    assert "InMemorySaver" in type(saver).__name__ or "MemorySaver" in type(saver).__name__
