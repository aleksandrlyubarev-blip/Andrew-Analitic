"""
core/checkpointing.py
=====================
LangGraph checkpointer factory. Picks the right backend based on env:

    LANGGRAPH_CHECKPOINTER  Forced backend. One of: "memory", "postgres", "none".
                            If unset, falls back to URL inspection.
    DATABASE_URL            Inspected when LANGGRAPH_CHECKPOINTER is unset.
                            postgresql:// or postgres://  → Postgres
                            anything else                 → in-memory

Default behaviour is "memory" (LangGraph's :class:`InMemorySaver`), which keeps
graph state alive across nodes within a single ``.invoke()`` call but does not
persist across processes. Cloud Run instances are ephemeral, so to keep
conversation state across requests you must use the Postgres backend.

Usage:
    from core.checkpointing import get_checkpointer
    saver = get_checkpointer()
    if saver and not getattr(saver, "_setup_done", False):
        saver.setup()                       # creates tables in `public` schema
    graph = workflow.compile(checkpointer=saver)
    graph.invoke(state, config={"configurable": {"thread_id": session_id}})

PostgresSaver creates its tables in the default search_path (typically
``public``) on first ``setup()`` call. We don't pre-create them in
``deploy/migrations/001_init.sql`` because the layout can drift between
versions of ``langgraph-checkpoint-postgres`` and a mismatched pre-creation
would silently corrupt inserts.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger("core.checkpointing")


def _resolve_backend(forced: str | None, db_url: str) -> str:
    """Map env values to a canonical backend name.

    >>> _resolve_backend("memory", "")
    'memory'
    >>> _resolve_backend("postgres", "")
    'postgres'
    >>> _resolve_backend("none", "")
    'none'
    >>> _resolve_backend(None, "postgresql://x")
    'postgres'
    >>> _resolve_backend(None, "postgres://x")
    'postgres'
    >>> _resolve_backend(None, "sqlite:///foo.db")
    'memory'
    >>> _resolve_backend(None, "")
    'memory'
    """
    if forced:
        normalised = forced.strip().lower()
        if normalised in {"memory", "postgres", "none"}:
            return normalised
        logger.warning(
            "Unknown LANGGRAPH_CHECKPOINTER=%r; falling back to URL inspection",
            forced,
        )
    if db_url.startswith(("postgresql://", "postgresql+", "postgres://")):
        return "postgres"
    return "memory"


def _build_postgres_checkpointer(db_url: str) -> Any:
    """Construct a :class:`PostgresSaver` for the given URL.

    Imported lazily so Andrew can boot without the optional
    ``langgraph-checkpoint-postgres`` dependency installed (e.g. in CI runs
    that only exercise SQLite paths). Returns the saver in setup-already-done
    mode, since the migration file owns the schema.
    """
    try:
        from langgraph.checkpoint.postgres import PostgresSaver  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Postgres checkpointer requested but langgraph-checkpoint-postgres "
            "is not installed. Add it to requirements.txt or set "
            "LANGGRAPH_CHECKPOINTER=memory to fall back to in-memory."
        ) from exc

    # langgraph's PostgresSaver expects a libpq URL; rewrite SQLAlchemy-style
    # postgresql+psycopg:// → postgresql:// so the saver's connection works.
    libpq_url = db_url
    if "+" in libpq_url.split("://", 1)[0]:
        scheme, rest = libpq_url.split("://", 1)
        libpq_url = scheme.split("+", 1)[0] + "://" + rest

    saver = PostgresSaver.from_conn_string(libpq_url)
    logger.info("LangGraph checkpointer: postgres (%s)", _safe_url(libpq_url))
    return saver


def _build_memory_checkpointer() -> Any:
    """In-memory saver. Survives within one process; lost at restart."""
    try:
        # langgraph >=0.2: InMemorySaver. Fall back to MemorySaver for older.
        from langgraph.checkpoint.memory import InMemorySaver  # type: ignore
    except ImportError:
        try:
            from langgraph.checkpoint.memory import MemorySaver as InMemorySaver  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "langgraph.checkpoint.memory not available; upgrade langgraph"
            ) from exc
    logger.info("LangGraph checkpointer: in-memory (ephemeral)")
    return InMemorySaver()


def _safe_url(url: str) -> str:
    """Strip credentials from a URL for log output."""
    if "@" not in url:
        return url
    scheme, rest = url.split("://", 1) if "://" in url else ("", url)
    creds_host, _, tail = rest.partition("/")
    _, _, host = creds_host.rpartition("@")
    return f"{scheme}://***@{host}/{tail}" if scheme else f"***@{host}/{tail}"


def get_checkpointer(
    *,
    backend: str | None = None,
    db_url: str | None = None,
) -> Optional[Any]:
    """Return a LangGraph checkpointer chosen from env/args.

    :param backend: Override ``LANGGRAPH_CHECKPOINTER``. ``"none"`` returns
                    ``None`` (graph compiled without persistence — current
                    default for the existing graphs).
    :param db_url:  Override ``DATABASE_URL``.

    Returns ``None`` only when backend resolves to ``"none"``. Otherwise
    raises if Postgres is requested but the optional package isn't installed.
    """
    forced = backend if backend is not None else os.getenv("LANGGRAPH_CHECKPOINTER")
    url = db_url if db_url is not None else os.getenv("DATABASE_URL", "")
    chosen = _resolve_backend(forced, url)

    if chosen == "none":
        logger.info("LangGraph checkpointer: disabled")
        return None
    if chosen == "postgres":
        return _build_postgres_checkpointer(url)
    return _build_memory_checkpointer()
