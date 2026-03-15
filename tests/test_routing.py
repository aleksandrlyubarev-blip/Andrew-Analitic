"""
Routing smoke tests — validates the 48-keyword weighted router.
Run: python -m pytest tests/test_routing.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.andrew_swarm import route_query_intent


def _route(query: str) -> dict:
    return route_query_intent({"user_request": query, "goal": query})


def test_arima_routes_to_reasoning():
    r = _route("Forecast next quarter revenue using ARIMA and confidence intervals")
    assert r["routing_decision"] == "reasoning_math"
    assert r["routing_score"] >= 4


def test_bar_chart_routes_to_standard():
    r = _route("Show total revenue by region as a bar chart")
    assert r["routing_decision"] == "standard"
    assert r["routing_score"] == 0


def test_monte_carlo_routes_to_reasoning():
    r = _route("Run Monte Carlo simulation for sales growth")
    assert r["routing_decision"] == "reasoning_math"


def test_average_by_month_routes_to_analytics():
    r = _route("What is the average revenue by month?")
    assert r["routing_decision"] == "analytics_fastlane"
    assert r["routing_score"] <= 3


def test_neural_network_routes_to_reasoning():
    r = _route("Build a neural network to predict customer churn")
    assert r["routing_decision"] == "reasoning_math"


def test_pie_chart_routes_to_standard():
    r = _route("Create a simple pie chart of market share")
    assert r["routing_decision"] == "standard"


def test_regression_routes_to_reasoning():
    r = _route("Perform linear regression on price vs quantity")
    assert r["routing_decision"] == "reasoning_math"


def test_cagr_with_analytics_context():
    r = _route("Calculate CAGR and break-even point for product lines")
    assert r["routing_decision"] in ("analytics_fastlane", "reasoning_math", "standard")
    assert r["routing_score"] >= 2


def test_group_by_routes_to_standard():
    r = _route("Group sales by region and show percentages")
    assert r["routing_decision"] == "standard"


def test_linear_programming_routes_to_reasoning():
    r = _route("Optimize inventory levels with linear programming")
    assert r["routing_decision"] == "reasoning_math"
    assert r["routing_score"] >= 7


def test_routing_returns_all_fields():
    r = _route("test query")
    assert "python_model" in r
    assert "orchestrator_model" in r
    assert "routing_decision" in r
    assert "routing_score" in r
    assert "routing_hits" in r


def test_empty_query_routes_to_standard():
    r = _route("")
    assert r["routing_decision"] == "standard"
    assert r["routing_score"] == 0
