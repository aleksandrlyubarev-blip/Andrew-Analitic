"""
MiroFish swarm tests — no LLM calls required.

All tests use max_cost=0.0 to skip LLM paths, exercising only the
deterministic numpy simulation and classification logic.

Run: python -m pytest tests/test_mirofish_swarm.py -v
"""

import pytest

from core.mirofish_swarm import (
    MiroFishExecutor,
    MiroFishResult,
    MiroFishState,
    PERSONALITY_PARAMS,
    _extract_base_defect_rate,
    aggregate_results,
    parse_input,
    spawn_agents,
)
from core.supervisor import MIROFISH_SIGNALS, _classify


# ============================================================
# Helpers
# ============================================================

def classify(query: str) -> str:
    decision, _ = _classify(query)
    return decision


def run_no_llm(query: str = "swarm forecast", production_data: dict = None) -> MiroFishResult:
    ex = MiroFishExecutor(max_cost=0.0)
    return ex.execute(query, production_data or {})


# ============================================================
# 1. Classification routing
# ============================================================

def test_mirofish_signals_route_to_mirofish():
    decision, signals = _classify("simulate defect forecast next shift")
    assert decision == "mirofish"


def test_swarm_keyword_routes_to_mirofish():
    decision, _ = _classify("run swarm simulation")
    assert decision == "mirofish"


def test_roboqc_keyword_routes_to_mirofish():
    decision, _ = _classify("roboqc quality control forecast")
    assert decision == "mirofish"


def test_patchcore_routes_to_mirofish():
    decision, _ = _classify("patchcore score anomaly")
    assert decision == "mirofish"


def test_no_mirofish_signals_stay_andrew():
    decision, _ = _classify("total revenue by region")
    assert decision == "andrew"


def test_educational_query_stays_romeo():
    decision, _ = _classify("explain what CAGR means")
    assert decision == "romeo"


def test_mirofish_signals_list_not_empty():
    assert len(MIROFISH_SIGNALS) >= 5


def test_mirofish_signals_include_expected_terms():
    assert "swarm" in MIROFISH_SIGNALS
    assert "roboqc" in MIROFISH_SIGNALS
    assert "patchcore" in MIROFISH_SIGNALS


# ============================================================
# 2. MiroFishExecutor basic execution
# ============================================================

def test_executor_returns_result():
    result = run_no_llm()
    assert isinstance(result, MiroFishResult)


def test_result_defect_probability_in_range():
    result = run_no_llm()
    assert 0.0 <= result.defect_probability <= 1.0


def test_result_has_recommended_action():
    result = run_no_llm()
    assert result.recommended_action in (
        "continue_production", "inspect_batch", "halt_line", "emergency_stop"
    )


def test_result_has_risk_level():
    result = run_no_llm()
    assert result.risk_level in ("low", "medium", "high", "critical")


def test_result_has_forecast_horizon():
    result = run_no_llm()
    assert isinstance(result.forecast_horizon, str)
    assert result.forecast_horizon.endswith("h")


def test_result_routing_is_mirofish():
    result = run_no_llm()
    assert result.routing == "mirofish"


# ============================================================
# 3. Simulation statistics
# ============================================================

def test_simulation_stats_present():
    result = run_no_llm("swarm forecast", {"anomaly_score": 0.5})
    assert "scenario_count" in result.simulation_stats
    assert "mean_defect_rate" in result.simulation_stats


def test_simulation_stats_scenario_count_matches():
    ex = MiroFishExecutor(max_cost=0.0)
    result = ex.execute("simulate", {}, scenario_count=500)
    assert result.simulation_stats.get("scenario_count") == 500


def test_higher_anomaly_score_raises_defect_rate():
    low_result = run_no_llm("forecast", {"anomaly_score": 0.05})
    high_result = run_no_llm("forecast", {"anomaly_score": 0.9})
    assert high_result.defect_probability >= low_result.defect_probability


def test_simulation_stats_has_p95():
    result = run_no_llm("swarm simulate", {"defect_rate": 0.2})
    assert "p95_defect_rate" in result.simulation_stats


def test_simulation_stats_has_mean_and_std():
    result = run_no_llm()
    stats = result.simulation_stats
    assert "mean_defect_rate" in stats
    assert "std_defect_rate" in stats


# ============================================================
# 4. MiroFishResult properties
# ============================================================

def test_result_cost_usd_non_negative():
    result = run_no_llm()
    assert result.cost_usd >= 0.0


def test_result_confidence_in_range():
    result = run_no_llm()
    assert 0.0 <= result.confidence <= 1.0


def test_result_success_true_when_no_error():
    result = run_no_llm()
    assert result.success is True


def test_result_sql_query_is_none():
    result = run_no_llm()
    assert result.sql_query is None


def test_result_python_code_is_none():
    result = run_no_llm()
    assert result.python_code is None


def test_to_roma_output_contains_goal():
    result = run_no_llm("predict assembly line defects")
    roma = result.to_roma_output()
    assert "predict assembly line defects" in roma


def test_to_roma_output_contains_defect_probability():
    result = run_no_llm()
    roma = result.to_roma_output()
    assert "Defect Probability" in roma or "defect" in roma.lower()


def test_str_representation():
    result = run_no_llm()
    s = str(result)
    assert "MiroFish" in s


def test_audit_log_present():
    result = run_no_llm()
    assert isinstance(result.audit_log, list)
    assert len(result.audit_log) > 0


# ============================================================
# 5. Error handling
# ============================================================

def test_bad_production_data_still_returns_result():
    ex = MiroFishExecutor(max_cost=0.0)
    result = ex.execute("forecast", {"anomaly_score": "not_a_number"})
    assert isinstance(result, MiroFishResult)


def test_empty_production_data_uses_defaults():
    result = run_no_llm("simulate", {})
    assert result.defect_probability >= 0.0


def test_zero_budget_skips_llm():
    ex = MiroFishExecutor(max_cost=0.0)
    result = ex.execute("simulate defect", {"patchcore_score": 0.6})
    assert result.cost_usd == 0.0


# ============================================================
# 6. Internal node functions
# ============================================================

def test_parse_input_defaults_scenario_count():
    state: MiroFishState = {"query": "test", "production_data": {}}
    out = parse_input(state)
    assert out.get("scenario_count", 1000) == 1000


def test_parse_input_clamps_scenario_count():
    state: MiroFishState = {"query": "test", "production_data": {}, "scenario_count": 99999}
    out = parse_input(state)
    assert out["scenario_count"] <= 10000


def test_spawn_agents_returns_configs():
    state: MiroFishState = {
        "production_data": {},
        "scenario_count": 100,
        "personality_profiles": ["qc_expert", "fast_worker"],
        "audit_log": [],
        "warnings": [],
    }
    out = spawn_agents(state)
    configs = out.get("agent_configs", [])
    assert len(configs) >= 2


def test_agent_personalities_valid():
    state: MiroFishState = {
        "production_data": {},
        "scenario_count": 50,
        "personality_profiles": ["cautious_operator"],
        "audit_log": [],
        "warnings": [],
    }
    out = spawn_agents(state)
    for agent in out.get("agent_configs", []):
        assert agent["personality"] in PERSONALITY_PARAMS


def test_extract_base_defect_rate_uses_defect_rate_key():
    rate = _extract_base_defect_rate({"defect_rate": 0.42})
    assert abs(rate - 0.42) < 1e-6


def test_extract_base_defect_rate_falls_back_to_01():
    rate = _extract_base_defect_rate({})
    assert abs(rate - 0.1) < 1e-6


# ============================================================
# 7. Supervisor integration
# ============================================================

def test_supervisor_state_has_production_data_field():
    from core.supervisor import SupervisorState
    # SupervisorState is a TypedDict — verify the key exists in annotations
    annotations = SupervisorState.__annotations__
    assert "production_data" in annotations


def test_supervisor_state_has_mirofish_result_field():
    from core.supervisor import SupervisorState
    annotations = SupervisorState.__annotations__
    assert "mirofish_result" in annotations
