"""
tests/conftest.py
=================
Pytest auto-loaded fixture file. Two responsibilities:

  1. Put the repo root on ``sys.path`` so tests can ``import bridge.*`` /
     ``import core.*`` without each file re-running the same boilerplate.
  2. Quiet down a couple of chatty third-party loggers during test runs
     (LiteLLM and LangSmith both default to INFO and pollute output with
     credential / SDK init banners).

Anything that needs to be *opt-in* per test (DB stubs, network mocks,
stubbed psycopg) stays in the individual test file — this conftest is for
project-wide setup only.
"""

from __future__ import annotations

import logging
import os
import sys

# ── 1. Repo root on sys.path ────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── 2. Quiet noisy third-party loggers during tests ─────────────────────────
# These two pollute the test output with init banners on every collection
# even when the actual test code never imports them. Cap at WARNING so
# legitimate failures (missing API key, SDK error) still surface.
for noisy in ("litellm", "LiteLLM", "langsmith", "httpx", "httpcore"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
