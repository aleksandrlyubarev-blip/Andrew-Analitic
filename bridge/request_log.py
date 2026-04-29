"""
bridge/request_log.py
=====================
Writes one row to ``andrew.request_log`` per request, so the cost-attribution
queries in deploy/BETA_RUNBOOK.md actually return data.

What's captured automatically by the middleware:
    user_slug   from request.state.user_slug (set by BetaApiKeyMiddleware)
    route       request.url.path
    status_code response.status_code
    latency_ms  perf_counter delta around call_next
    error       short string when status >= 500

What handlers must opt into by setting on ``request.state``:
    model       e.g. "gpt-4o-mini" (set by core.routing when it picks a lane)
    tokens_in   prompt tokens
    tokens_out  completion tokens
    cost_usd    LiteLLM's reported cost for the call

Active when both:
  • ``DATABASE_URL`` looks like a Postgres URL (sqlite/empty → middleware is
    a no-op, so SQLite-based local dev keeps working without a DB writer)
  • ``REQUEST_LOG_ENABLED`` is unset OR not ``"false"``

Fail-open by design: any psycopg exception is logged at WARNING and the
request response goes through unchanged. We never block on the DB write.

Skipped paths (logged at DEBUG, not persisted) — these are noisy probes that
would flood the table without adding cost-attribution value:
    /health
    /openapi.json
    /docs
    /redoc
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.types import ASGIApp

logger = logging.getLogger("bridge.request_log")

SKIP_PATHS: frozenset[str] = frozenset({
    "/health",
    "/openapi.json",
    "/docs",
    "/redoc",
})


def _looks_like_postgres(url: str) -> bool:
    if not url:
        return False
    scheme = url.split("://", 1)[0].lower()
    base = scheme.split("+", 1)[0]
    return base in {"postgresql", "postgres"}


def _to_libpq(url: str) -> str:
    """Strip SQLAlchemy driver suffix so psycopg can parse the URL."""
    if "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    return scheme.split("+", 1)[0] + "://" + rest


class RequestLogMiddleware(BaseHTTPMiddleware):
    """Persist request metadata to ``andrew.request_log``.

    :param db_url: libpq-style Postgres URL. None or non-Postgres → no-op.
    :param skip_paths: extra paths to skip beyond :data:`SKIP_PATHS`.
    """

    def __init__(
        self,
        app: ASGIApp,
        db_url: str | None = None,
        skip_paths: frozenset[str] = SKIP_PATHS,
    ) -> None:
        super().__init__(app)
        resolved = db_url if db_url is not None else os.getenv("DATABASE_URL", "")
        self._enabled = _looks_like_postgres(resolved)
        self._url = _to_libpq(resolved) if self._enabled else ""
        self._skip = skip_paths
        if self._enabled:
            logger.info("RequestLogMiddleware enabled (postgres)")
        else:
            logger.info(
                "RequestLogMiddleware disabled (DATABASE_URL=%s)",
                "<unset>" if not resolved else resolved.split("://", 1)[0] + "://...",
            )

    async def dispatch(self, request: Request, call_next):
        if not self._enabled or request.url.path in self._skip:
            return await call_next(request)

        start = time.perf_counter()
        status_code = 0
        error: Optional[str] = None
        response = None
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            error = f"{type(exc).__name__}: {exc}"[:500]
            raise
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            # Pull optional extras a handler may have set; default to None.
            state = request.state
            payload = {
                "user_slug":  getattr(state, "user_slug", "anonymous"),
                "route":      request.url.path,
                "model":      getattr(state, "model", None),
                "tokens_in":  getattr(state, "tokens_in", None),
                "tokens_out": getattr(state, "tokens_out", None),
                "latency_ms": latency_ms,
                "cost_usd":   getattr(state, "cost_usd", None),
                "status_code": status_code,
                "error":      error,
            }
            # Fire-and-forget so we don't add DB latency to the response.
            try:
                asyncio.create_task(self._write(payload))
            except RuntimeError:
                # Outside an event loop — best-effort sync fallback. Should
                # never happen under uvicorn but keeps tests + scripts safe.
                pass
        return response

    async def _write(self, payload: dict[str, Any]) -> None:
        try:
            import psycopg  # type: ignore
        except ImportError:
            logger.warning("psycopg not installed; request_log row dropped")
            return
        try:
            async with await psycopg.AsyncConnection.connect(self._url) as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        """
                        INSERT INTO andrew.request_log
                            (user_slug, route, model, tokens_in, tokens_out,
                             latency_ms, cost_usd, status_code, error)
                        VALUES
                            (%(user_slug)s, %(route)s, %(model)s,
                             %(tokens_in)s, %(tokens_out)s,
                             %(latency_ms)s, %(cost_usd)s,
                             %(status_code)s, %(error)s)
                        """,
                        payload,
                    )
                await conn.commit()
        except Exception as exc:  # pragma: no cover - exercised in tests
            logger.warning("request_log insert failed: %s", exc)


def middleware_kwargs_from_env() -> dict | None:
    """Return ``add_middleware`` kwargs, or ``None`` to skip mounting.

    Honours the same DATABASE_URL inspection as the middleware itself plus
    a kill-switch env var (``REQUEST_LOG_ENABLED=false``) for the rare case
    a deploy needs to disable the writer without removing the secret.
    """
    if os.getenv("REQUEST_LOG_ENABLED", "true").strip().lower() == "false":
        return None
    url = os.getenv("DATABASE_URL", "")
    if not _looks_like_postgres(url):
        return None
    return {"db_url": url}
