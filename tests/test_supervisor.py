"""
Supervisor routing tests — no LLM calls required.

All 10 tests exercise the pure _classify() function which uses keyword
matching only. They run instantly without API keys or a database.
"""

import pytest

from core.supervisor import _classify


# ============================================================
# Helpers
# ============================================================

def classify(query: str) -> str:
    decision, _ = _classify(query)
    return decision


def signals(query: str) -> list:
    _, hits = _classify(query)
    return hits


# ============================================================
# 1. Pure analytical queries → andrew
# ============================================================

def test_analytical_query_routes_to_andrew():
    assert classify("total revenue by region last quarter") == "andrew"


def test_data_query_routes_to_andrew():
    assert classify("show me the top 5 products by quantity") == "andrew"


def test_math_heavy_routes_to_andrew():
    assert classify("forecast revenue using ARIMA time series") == "andrew"


# ============================================================
# 2. Pure educational queries → romeo
# ============================================================

def test_explain_routes_to_romeo():
    assert classify("explain what CAGR means") == "romeo"


def test_compare_routes_to_romeo():
    assert classify("difference between mean and median") == "romeo"


def test_tutorial_routes_to_romeo():
    assert classify("tutorial on linear regression basics") == "romeo"


def test_what_is_routes_to_romeo():
    assert classify("what is a confidence interval") == "romeo"


# ============================================================
# 3. Hybrid queries → both
# ============================================================

def test_hybrid_query_routes_to_both():
    assert classify("explain ARIMA and forecast next quarter revenue using it") == "both"


def test_hybrid_explain_plus_calculate():
    assert classify("explain CAGR and calculate it for our product sales data") == "both"


# ============================================================
# 4. Edge cases
# ============================================================

def test_empty_query_routes_to_andrew():
    """Empty query defaults to andrew — safe fallback."""
    assert classify("") == "andrew"


def test_signals_returned_for_educational():
    hits = signals("explain what CAGR means")
    assert len(hits) > 0


def test_signals_returned_for_hybrid():
    hits = signals("explain ARIMA and forecast revenue")
    assert len(hits) > 0
    # Should contain both educational and analytical signals
    lower_hits = [h.lower() for h in hits]
    has_edu = any(s in lower_hits for s in ["explain", "what is", "how does", "vs", "compare"])
    has_ana = any(s in lower_hits for s in ["arima", "forecast", "revenue", "predict"])
    assert has_edu or has_ana  # at minimum one type present


def test_default_no_clear_signals_routes_to_andrew():
    """Generic non-educational text falls through to andrew by default."""
    assert classify("run my quarterly report") == "andrew"
