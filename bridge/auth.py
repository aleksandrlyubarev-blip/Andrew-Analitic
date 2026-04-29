"""
bridge/auth.py
==============
Per-tester API-key authentication for the GCP beta.

Cloud Run injects a `BETA_API_KEYS` env var (sourced from Secret Manager) that
holds a JSON object: ``{"<user_slug>": "<32-byte hex key>", ...}``. Callers
pass the key in the ``X-Api-Key`` header. The middleware:

  • Loads keys once at startup (constant-time dict lookup per request).
  • Validates the header with :func:`hmac.compare_digest` (timing-safe).
  • Stores the resolved ``user_slug`` on ``request.state.user_slug`` for
    downstream handlers and structured logs.
  • Bypasses public paths — health probes, demo routes — listed in
    :data:`PUBLIC_PATHS` so Cloud Run / load balancer probes don't need a key.
  • If ``BETA_API_KEYS`` is unset or empty, the middleware is a no-op
    (unauthenticated mode for local dev). The deploy scripts always set it
    in production, so this default keeps tests and ``docker compose up``
    working without ceremony.

Configure (Cloud Run):
    --set-secrets=BETA_API_KEYS=BETA_API_KEYS:latest
    --set-env-vars=BETA_AUTH_HEADER=X-Api-Key   # optional, defaults shown
"""

from __future__ import annotations

import hmac
import json
import logging
import os
from typing import Mapping

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger("bridge.auth")

# Paths exempt from auth. Health is required by Cloud Run's startup/liveness
# probes (which can't carry a header). Demo routes are public on purpose so
# RomeoFlexVision frontend integration doesn't need a key.
PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/scene/ops",
    "/scene/ops/demo",
    "/docs",
    "/openapi.json",
    "/redoc",
})


def load_beta_keys(env_value: str | None) -> dict[str, str]:
    """Parse the ``BETA_API_KEYS`` env value into a ``{slug: key}`` dict.

    Returns an empty dict for unset / blank / malformed input. Callers should
    treat an empty dict as "auth disabled".
    """
    if not env_value:
        return {}
    try:
        parsed = json.loads(env_value)
    except json.JSONDecodeError:
        logger.error("BETA_API_KEYS is not valid JSON; auth disabled")
        return {}
    if not isinstance(parsed, dict):
        logger.error("BETA_API_KEYS must be a JSON object; auth disabled")
        return {}
    out: dict[str, str] = {}
    for slug, key in parsed.items():
        if isinstance(slug, str) and isinstance(key, str) and slug and key:
            out[slug] = key
    return out


class BetaApiKeyMiddleware(BaseHTTPMiddleware):
    """ASGI middleware that gates non-public routes on ``X-Api-Key``."""

    def __init__(
        self,
        app: ASGIApp,
        keys: Mapping[str, str] | None = None,
        header_name: str = "X-Api-Key",
        public_paths: frozenset[str] = PUBLIC_PATHS,
        fail_closed: bool = False,
    ) -> None:
        super().__init__(app)
        self._keys: dict[str, str] = dict(keys or {})
        self._header_name = header_name
        self._public_paths = public_paths
        self._fail_closed = fail_closed
        # No keys + not fail-closed = no-op (local dev). No keys + fail-closed
        # = reject everything non-public (production safety net).
        self._enabled = bool(self._keys) or fail_closed
        if not self._keys and not fail_closed:
            logger.warning(
                "BetaApiKeyMiddleware loaded with no keys — auth disabled. "
                "Set BETA_API_KEYS for production."
            )

    async def dispatch(self, request: Request, call_next):
        if not self._enabled:
            request.state.user_slug = "anonymous"
            return await call_next(request)

        if request.url.path in self._public_paths:
            request.state.user_slug = "public"
            return await call_next(request)

        provided = request.headers.get(self._header_name)
        if not provided:
            return JSONResponse(
                {"error": "missing_api_key", "header": self._header_name},
                status_code=401,
            )

        # Timing-safe compare against every active key. With ≤10 testers the
        # linear scan is fine; if this ever grows, replace with a constant-time
        # hash of the key as the lookup index.
        for slug, expected in self._keys.items():
            if hmac.compare_digest(provided, expected):
                request.state.user_slug = slug
                return await call_next(request)

        # No match (or fail_closed with no keys configured at all).
        return JSONResponse({"error": "invalid_api_key"}, status_code=403)


def middleware_kwargs_from_env() -> dict | None:
    """Return ``app.add_middleware`` kwargs for the env, or ``None`` if disabled.

    Returns ``None`` when ``BETA_API_KEYS`` is unset and ``BETA_AUTH_REQUIRED``
    is not ``"true"``. Otherwise returns a dict the caller can splat into
    :py:meth:`fastapi.FastAPI.add_middleware`. When ``BETA_AUTH_REQUIRED=true``
    but no keys are configured, the returned middleware will reject every
    non-public request (fail-closed).
    """
    keys = load_beta_keys(os.getenv("BETA_API_KEYS"))
    required = os.getenv("BETA_AUTH_REQUIRED", "").lower() == "true"
    if not keys and not required:
        return None
    return {
        "keys": keys,
        "header_name": os.getenv("BETA_AUTH_HEADER", "X-Api-Key"),
        "fail_closed": required,
    }
