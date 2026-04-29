"""
bridge/logging_setup.py
=======================
Structured logging configuration. Toggled by ``LOG_FORMAT``:

    LOG_FORMAT=json   → one JSON object per log line, parsed by Cloud Logging
                        into ``jsonPayload`` with severity mapped from level.
    LOG_FORMAT=plain  → human-readable single-line text (default for local dev)

Cloud Logging recognises the following top-level keys when emitted as JSON:
``severity``, ``message``, ``timestamp``, ``logging.googleapis.com/trace``.
Anything else lands under ``jsonPayload`` for filtering in the Logs Explorer.

Calling :func:`configure_logging` more than once is safe — the root logger's
handlers are reset on each call so reload-friendly servers (uvicorn --reload)
don't accumulate duplicates.
"""

from __future__ import annotations

import logging
import os
import sys

try:
    from pythonjsonlogger import jsonlogger  # type: ignore
    _HAVE_JSONLOGGER = True
except ImportError:  # pragma: no cover - optional dep
    _HAVE_JSONLOGGER = False


_LEVEL_TO_SEVERITY = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


class _CloudLoggingFormatter(logging.Formatter):
    """JSON formatter that maps Python log levels to Cloud Logging severities.

    Used as a fallback when ``python-json-logger`` is not installed. Output
    schema is intentionally minimal — Cloud Logging treats unknown keys as
    ``jsonPayload`` fields automatically.
    """

    def format(self, record: logging.LogRecord) -> str:
        import json
        payload = {
            "severity": _LEVEL_TO_SEVERITY.get(record.levelname, "DEFAULT"),
            "message": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything attached via logger.info("msg", extra={"k": v}) is on the
        # record as an attribute — surface a small whitelist of useful ones.
        for attr in ("user_slug", "route", "model", "latency_ms",
                     "tokens_in", "tokens_out", "cost_usd", "status_code"):
            value = getattr(record, attr, None)
            if value is not None:
                payload[attr] = value
        return json.dumps(payload, default=str)


def configure_logging(
    level: str | None = None,
    fmt: str | None = None,
    stream=None,
) -> None:
    """Initialise the root logger.

    :param level: Override ``LOG_LEVEL``. Defaults to ``INFO``.
    :param fmt:   Override ``LOG_FORMAT``. ``"json"`` enables structured logs.
    :param stream: Stream for the handler; defaults to ``sys.stdout``.
    """
    level_name = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    fmt_name = (fmt or os.getenv("LOG_FORMAT", "plain")).lower()
    stream = stream or sys.stdout

    handler = logging.StreamHandler(stream)
    if fmt_name == "json":
        if _HAVE_JSONLOGGER:
            formatter: logging.Formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s",
                rename_fields={"levelname": "severity", "asctime": "timestamp"},
            )
        else:
            formatter = _CloudLoggingFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "%Y-%m-%dT%H:%M:%S",
        )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Reset handlers so repeat calls (uvicorn --reload, pytest) don't pile up.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level_name, logging.INFO))

    # Quiet down noisy third-party loggers; Cloud Run logs each request line
    # via gunicorn's access log, which is enough.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
