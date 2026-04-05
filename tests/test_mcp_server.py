"""
MCP Server tests — Sprint 10.
Tests tool registration, input schema validation, call routing,
output formatting, metrics tracking, and error handling.

Run: python -m pytest tests/test_mcp_server.py -v
"""

import asyncio
import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Helpers ──────────────────────────────────────────────────

def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _fake_analyze_result(success=True, narrative="Revenue Q3: $1.2M", cost=0.003, confidence=0.9):
    return {
        "query": "test",
        "narrative": narrative,
        "sql_query": "SELECT * FROM sales",
        "confidence": confidence,
        "cost_usd": cost,
        "success": success,
        "error": None if success else "DB error",
        "elapsed_seconds": 0.5,
        "routing": "analytics_fastlane",
        "model_used": "gpt-4o-mini",
        "agent_used": "andrew",
        "channel": "mcp",
        "warnings": [],
        "formatted_message": "Analysis done.",
        "hitl_decision": "skipped",
        "hitl_review_id": None,
    }


def _fake_swarm_result(risk="medium", defect_p=0.35):
    return {
        "query": "test",
        "final_report": "Defect risk is moderate at station 3.",
        "defect_probability": defect_p,
        "recommended_action": "inspect_batch",
        "visual_evidence": ["./evidence/frame_001.jpg"],
        "risk_level": risk,
        "forecast_horizon": "8h",
        "confidence": 0.78,
        "cost_usd": 0.002,
        "success": True,
        "error": None,
        "elapsed_seconds": 0.3,
        "simulation_stats": {
            "scenario_count": 1000,
            "mean_defect_rate": defect_p,
            "std_defect_rate": 0.05,
            "p5_defect_rate": 0.28,
            "p95_defect_rate": 0.42,
            "risk_level": risk,
        },
        "formatted_message": "AndrewSim done.",
        "warnings": [],
    }


# ── Tool registration ────────────────────────────────────────

def test_list_tools_returns_five_tools():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    assert len(tools) == 5


def test_tool_names_are_correct():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    names = {t.name for t in tools}
    assert "andrew_analyze" in names
    assert "andrew_educate" in names
    assert "andrew_swarm_simulate" in names
    assert "andrew_get_metrics" in names
    assert "andrew_health" in names


def test_analyze_tool_has_required_query_param():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    analyze = next(t for t in tools if t.name == "andrew_analyze")
    assert "query" in analyze.inputSchema["properties"]
    assert "query" in analyze.inputSchema["required"]


def test_swarm_tool_has_production_data_schema():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    swarm = next(t for t in tools if t.name == "andrew_swarm_simulate")
    props = swarm.inputSchema["properties"]
    assert "production_data" in props
    assert "scenario_count" in props
    assert "personality_profiles" in props
    assert swarm.inputSchema["required"] == ["query"]


def test_swarm_tool_scenario_count_has_bounds():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    swarm = next(t for t in tools if t.name == "andrew_swarm_simulate")
    sc = swarm.inputSchema["properties"]["scenario_count"]
    assert sc["minimum"] == 10
    assert sc["maximum"] == 10000


def test_metrics_tool_requires_no_params():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    metrics = next(t for t in tools if t.name == "andrew_get_metrics")
    assert metrics.inputSchema["required"] == []


def test_all_tools_have_descriptions():
    from bridge.mcp_server import list_tools
    tools = run(list_tools())
    for tool in tools:
        assert tool.description, f"{tool.name} has no description"
        assert len(tool.description) > 20


# ── andrew_analyze calls ─────────────────────────────────────

def test_analyze_empty_query_returns_error():
    from bridge.mcp_server import call_tool
    result = run(call_tool("andrew_analyze", {"query": ""}))
    assert len(result) == 1
    assert "required" in result[0].text.lower() or "error" in result[0].text.lower()


def test_analyze_success_contains_narrative():
    from bridge.mcp_server import call_tool, _get_bridge
    mock_bridge = AsyncMock()
    mock_bridge.handle_query = AsyncMock(return_value=_fake_analyze_result())
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_analyze", {"query": "total revenue"}))
    assert len(result) == 1
    assert "Revenue Q3" in result[0].text


def test_analyze_success_contains_sql():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_query = AsyncMock(return_value=_fake_analyze_result())
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_analyze", {"query": "sales by region"}))
    assert "SELECT" in result[0].text


def test_analyze_failure_shows_error():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_query = AsyncMock(
        return_value=_fake_analyze_result(success=False, narrative="", cost=0.001)
    )
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_analyze", {"query": "broken query"}))
    assert "failed" in result[0].text.lower() or "error" in result[0].text.lower()


def test_analyze_exception_handled_gracefully():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_query = AsyncMock(side_effect=RuntimeError("DB down"))
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_analyze", {"query": "crash test"}))
    assert "failed" in result[0].text.lower()


# ── andrew_swarm_simulate calls ──────────────────────────────

def test_swarm_empty_query_returns_error():
    from bridge.mcp_server import call_tool
    result = run(call_tool("andrew_swarm_simulate", {}))
    assert "required" in result[0].text.lower() or "error" in result[0].text.lower()


def test_swarm_success_contains_defect_probability():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_swarm_simulation = AsyncMock(return_value=_fake_swarm_result())
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_swarm_simulate", {
            "query": "predict defects",
            "production_data": {"anomaly_score": 0.7},
            "scenario_count": 500,
        }))
    text = result[0].text
    assert "35.0%" in text or "Defect Probability" in text


def test_swarm_success_contains_risk_level():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_swarm_simulation = AsyncMock(
        return_value=_fake_swarm_result(risk="high", defect_p=0.6)
    )
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_swarm_simulate", {"query": "high risk factory"}))
    assert "HIGH" in result[0].text


def test_swarm_scenario_count_clamped():
    from bridge.mcp_server import call_tool
    captured = {}

    async def fake_sim(query, production_data, scenario_count, personality_profiles, context):
        captured["scenario_count"] = scenario_count
        return _fake_swarm_result()

    mock_bridge = AsyncMock()
    mock_bridge.handle_swarm_simulation = fake_sim
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        run(call_tool("andrew_swarm_simulate", {
            "query": "test",
            "scenario_count": 99999,  # exceeds max 10000
        }))
    assert captured.get("scenario_count") == 10000


def test_swarm_contains_simulation_stats():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_swarm_simulation = AsyncMock(return_value=_fake_swarm_result())
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_swarm_simulate", {"query": "stats test"}))
    assert "Scenarios run" in result[0].text or "scenario_count" in result[0].text.lower()


def test_swarm_contains_visual_evidence():
    from bridge.mcp_server import call_tool
    mock_bridge = AsyncMock()
    mock_bridge.handle_swarm_simulation = AsyncMock(return_value=_fake_swarm_result())
    with patch("bridge.mcp_server._get_bridge", return_value=mock_bridge):
        result = run(call_tool("andrew_swarm_simulate", {"query": "evidence test"}))
    assert "frame_001.jpg" in result[0].text


# ── andrew_get_metrics ───────────────────────────────────────

def test_metrics_returns_json():
    from bridge.mcp_server import call_tool, _metrics
    result = run(call_tool("andrew_get_metrics", {}))
    data = json.loads(result[0].text)
    assert "total_calls" in data
    assert "total_cost_usd" in data
    assert "tool_calls" in data
    assert "uptime_seconds" in data


def test_metrics_tracks_tool_calls():
    from bridge.mcp_server import call_tool, _metrics
    before = _metrics["tool_calls"].get("andrew_get_metrics", 0)
    run(call_tool("andrew_get_metrics", {}))
    run(call_tool("andrew_get_metrics", {}))
    after = _metrics["tool_calls"].get("andrew_get_metrics", 0)
    assert after >= before + 2


def test_metrics_avg_cost_per_call_non_negative():
    from bridge.mcp_server import call_tool
    result = run(call_tool("andrew_get_metrics", {}))
    data = json.loads(result[0].text)
    assert data["avg_cost_per_call"] >= 0.0


# ── Unknown tool ─────────────────────────────────────────────

def test_unknown_tool_returns_error():
    from bridge.mcp_server import call_tool
    result = run(call_tool("nonexistent_tool", {}))
    assert "unknown" in result[0].text.lower()


# ── Output format helpers ────────────────────────────────────

def test_format_analyze_result_success():
    from bridge.mcp_server import _format_analyze_result
    result = _fake_analyze_result()
    text = _format_analyze_result(result, elapsed=0.5)
    assert "Revenue Q3" in text
    assert "confidence" in text.lower()
    assert "SELECT" in text


def test_format_analyze_result_failure():
    from bridge.mcp_server import _format_analyze_result
    result = _fake_analyze_result(success=False, narrative="")
    text = _format_analyze_result(result, elapsed=0.1)
    assert "failed" in text.lower()


def test_format_swarm_result_high_risk():
    from bridge.mcp_server import _format_swarm_result
    result = _fake_swarm_result(risk="critical", defect_p=0.85)
    text = _format_swarm_result(result, elapsed=1.2)
    assert "CRITICAL" in text
    assert "85.0%" in text


def test_format_swarm_result_contains_action():
    from bridge.mcp_server import _format_swarm_result
    result = _fake_swarm_result()
    text = _format_swarm_result(result, elapsed=0.8)
    assert "inspect_batch" in text
