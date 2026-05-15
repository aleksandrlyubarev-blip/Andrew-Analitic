"""
core/anthropic_auth.py
======================
Anthropic provider auth-mode toggle.

Selects how Anthropic-bound LLM calls authenticate without touching the call
sites in `lcb/`, `core/romeo_swarm.py`, `core/romeo_phd.py`, `core/memory.py`.
Other providers (Grok, OpenAI) are unaffected.

Modes (env var ANTHROPIC_AUTH_MODE):
  - "api_key" (default): standard ANTHROPIC_API_KEY path used by LiteLLM today.
  - "sdk":               plan auth via `claude login` (Claude Code / Agent SDK
                          cached credentials). Hot-path wiring lands in
                          Phase 1.3 of the SDK-credit migration.

This module is intentionally minimal in Phase 0 — it exposes the toggle and
logs the active mode so operators can verify env propagation. The actual
LiteLLM redirection lands when `claude login` is available on prod (15.06).
"""

from __future__ import annotations

import logging
import os
from typing import Literal

logger = logging.getLogger("anthropic_auth")

AuthMode = Literal["api_key", "sdk"]
VALID_MODES: tuple[AuthMode, ...] = ("api_key", "sdk")
DEFAULT_MODE: AuthMode = "api_key"


def get_auth_mode() -> AuthMode:
    raw = (os.getenv("ANTHROPIC_AUTH_MODE") or DEFAULT_MODE).strip().lower()
    if raw not in VALID_MODES:
        logger.warning(
            "Unknown ANTHROPIC_AUTH_MODE=%r, falling back to %r", raw, DEFAULT_MODE
        )
        return DEFAULT_MODE
    return raw  # type: ignore[return-value]


def is_sdk_mode() -> bool:
    return get_auth_mode() == "sdk"


def require_api_key_mode(component: str) -> None:
    """
    Hot-path guard for code paths that still assume ANTHROPIC_API_KEY.
    Call this at the boundary of any module that calls `litellm.completion`
    against an `anthropic/...` model. Raises a clear error if the operator has
    flipped to sdk mode before Phase 1.3 wiring is in place.
    """
    if is_sdk_mode():
        raise NotImplementedError(
            f"{component}: ANTHROPIC_AUTH_MODE=sdk requested, but plan-auth "
            "wiring is not yet implemented (Phase 1.3 of the SDK-credit "
            "migration). Unset ANTHROPIC_AUTH_MODE or set it to 'api_key' to "
            "use the current path."
        )
