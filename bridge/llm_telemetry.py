"""
bridge/llm_telemetry.py
=======================
Per-LLM-call telemetry. Installs a :data:`litellm.success_callback` and
:data:`litellm.failure_callback` that emit one structured log line per
completion call. With ``LOG_FORMAT=json`` (set by Cloud Run), each call
becomes one ``jsonPayload`` row in Cloud Logging that you can filter on.

Why log instead of pushing into ``andrew.request_log``?
  • One ``/analyze`` request makes 3-5 LiteLLM calls (route, SQL gen,
    Python gen, sandbox, hypothesis gate). Aggregating them into a single
    row would lose per-call detail; logging each keeps full granularity
    and pairs naturally with Cloud Logging's free 50 GiB/mo quota.
  • The middleware-driven row in ``andrew.request_log`` still gets the
    summed ``cost_usd`` from the bridge's result dict — no double counting.

Fields emitted on success:
    event       "llm_call"
    model       e.g. "gpt-4o-mini" (LiteLLM's resolved model id)
    tokens_in   prompt_tokens (or 0 if usage missing)
    tokens_out  completion_tokens (or 0 if usage missing)
    cost_usd    response_cost (LiteLLM-computed; 0.0 if unavailable)
    latency_ms  end_time - start_time, rounded
    request_id  LiteLLM's per-call id (correlates retries)

Fields emitted on failure:
    event       "llm_error"
    model       attempted model
    error       exception class + message (truncated)
    latency_ms  end_time - start_time

Querying in Cloud Logging:
    jsonPayload.event="llm_call" AND jsonPayload.cost_usd > 0.10

The module is import-safe even when ``litellm`` is missing — install() is a
no-op in that case so the bridge keeps working in CI envs without the dep.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Mapping

logger = logging.getLogger("bridge.llm_telemetry")

# Module-level marker so install() is idempotent across reload cycles.
_INSTALLED = False


def _coerce_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _coerce_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict OR an attribute from an object. LiteLLM
    sometimes hands the callback a ModelResponse object, sometimes a dict."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_usage(response: Any) -> tuple[int, int]:
    usage = _get(response, "usage")
    if usage is None:
        return 0, 0
    return _coerce_int(_get(usage, "prompt_tokens")), _coerce_int(_get(usage, "completion_tokens"))


def _latency_ms(start_time: Any, end_time: Any) -> int:
    """Compute latency in ms from LiteLLM's start/end timestamps.

    LiteLLM passes datetime objects in modern versions, floats in older.
    Handles both without raising.
    """
    if start_time is None or end_time is None:
        return 0
    try:
        delta = end_time - start_time
        # datetime.timedelta has total_seconds(); float - float is a float.
        secs = delta.total_seconds() if hasattr(delta, "total_seconds") else float(delta)
        return int(secs * 1000)
    except Exception:
        return 0


def _on_success(kwargs: Any, response: Any, start_time: Any, end_time: Any) -> None:
    """Emit one structured INFO log per successful LiteLLM completion."""
    try:
        tokens_in, tokens_out = _extract_usage(response)
        cost = _coerce_float(_get(response, "response_cost"))
        # Older LiteLLM returns cost via kwargs["response_cost"] instead.
        if cost == 0.0 and isinstance(kwargs, Mapping):
            cost = _coerce_float(kwargs.get("response_cost"))
        model = _get(response, "model") or (kwargs.get("model") if isinstance(kwargs, Mapping) else None)
        request_id = _get(response, "id") or _get(response, "request_id")
        logger.info(
            "llm_call",
            extra={
                "event": "llm_call",
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cost_usd": round(cost, 6),
                "latency_ms": _latency_ms(start_time, end_time),
                "request_id": request_id,
            },
        )
    except Exception as exc:  # pragma: no cover - never raise from a callback
        logger.warning("llm_telemetry success callback failed: %s", exc)


def _on_failure(kwargs: Any, response: Any, start_time: Any, end_time: Any) -> None:
    """Emit one structured WARNING log per failed LiteLLM completion."""
    try:
        model = (kwargs.get("model") if isinstance(kwargs, Mapping) else None) or _get(response, "model")
        # LiteLLM stuffs the original exception into kwargs["exception"] on failure.
        err = kwargs.get("exception") if isinstance(kwargs, Mapping) else None
        if err is None:
            err = _get(response, "exception") or response
        err_str = f"{type(err).__name__}: {err}" if not isinstance(err, str) else err
        logger.warning(
            "llm_error",
            extra={
                "event": "llm_error",
                "model": model,
                "error": err_str[:500],
                "latency_ms": _latency_ms(start_time, end_time),
            },
        )
    except Exception as exc:  # pragma: no cover - never raise from a callback
        logger.warning("llm_telemetry failure callback failed: %s", exc)


def install() -> bool:
    """Register the callbacks with LiteLLM. Idempotent.

    Returns True if installation succeeded (or was already done), False if
    LiteLLM isn't importable. Honours ``LLM_TELEMETRY_ENABLED=false`` as a
    kill switch for the rare case a deploy needs to silence per-call logs.
    """
    global _INSTALLED
    if _INSTALLED:
        return True
    if os.getenv("LLM_TELEMETRY_ENABLED", "true").strip().lower() == "false":
        logger.info("llm_telemetry disabled via LLM_TELEMETRY_ENABLED=false")
        return False
    try:
        import litellm  # type: ignore
    except ImportError:
        logger.warning("litellm not installed; llm_telemetry not active")
        return False

    # litellm.success_callback / failure_callback are module-level lists.
    # Append (don't replace) so we don't clobber other observability hooks.
    if _on_success not in getattr(litellm, "success_callback", []):
        litellm.success_callback.append(_on_success)  # type: ignore[attr-defined]
    if _on_failure not in getattr(litellm, "failure_callback", []):
        litellm.failure_callback.append(_on_failure)  # type: ignore[attr-defined]

    _INSTALLED = True
    logger.info("llm_telemetry installed (success + failure callbacks)")
    return True


def uninstall() -> None:
    """Remove our callbacks. Mostly for tests."""
    global _INSTALLED
    try:
        import litellm  # type: ignore
    except ImportError:
        _INSTALLED = False
        return
    for name in ("success_callback", "failure_callback"):
        seq = getattr(litellm, name, None)
        if seq is None:
            continue
        for cb in (_on_success, _on_failure):
            if cb in seq:
                seq.remove(cb)
    _INSTALLED = False
