"""
MiroFish Swarm Core v1.0.0
===================================================
Multi-agent swarm simulation engine for Physical AI / RoboQC (factory quality
control) forecasting. LangGraph-based, follows the same code patterns as
andrew_swarm.py.

Simulation pipeline:
  parse_input → build_knowledge_graph → spawn_agents → run_simulations
  → aggregate_results → generate_report
"""

import hashlib
import json
import logging
import os
import random
import time
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger("mirofish")

MODEL_ANALYTICS = os.getenv("MODEL_ANALYTICS", "gpt-4o-mini")
MAX_COST_DEFAULT = float(os.getenv("MIROFISH_MAX_COST", "1.00"))

PERSONALITY_DEFAULTS = ["cautious_operator", "qc_expert", "fast_worker"]

PERSONALITY_PARAMS: Dict[str, Dict[str, float]] = {
    "cautious_operator": {"risk_tolerance": 0.2, "speed_bias": 0.7, "quality_threshold": 0.95},
    "qc_expert":         {"risk_tolerance": 0.15, "speed_bias": 0.8, "quality_threshold": 0.98},
    "fast_worker":       {"risk_tolerance": 0.6,  "speed_bias": 1.3, "quality_threshold": 0.75},
    "maintenance_tech":  {"risk_tolerance": 0.3,  "speed_bias": 0.9, "quality_threshold": 0.90},
    "shift_supervisor":  {"risk_tolerance": 0.4,  "speed_bias": 1.0, "quality_threshold": 0.88},
}


# ============================================================
# 1. STATE
# ============================================================

class MiroFishState(TypedDict, total=False):
    # Input
    production_data: Dict[str, Any]
    query: str
    scenario_count: int
    personality_profiles: List[str]

    # KG
    knowledge_graph: Dict[str, Any]
    kg_summary: str

    # Swarm
    agent_configs: List[Dict[str, Any]]

    # Simulation
    simulation_results: List[Dict[str, Any]]
    simulation_stats: Dict[str, Any]

    # Output
    defect_probability: float
    recommended_action: str
    visual_evidence: List[str]
    risk_level: str
    forecast_horizon: str

    # Meta
    confidence: float
    warnings: List[str]
    audit_log: List[Dict[str, Any]]
    error_message: str
    cost_usd: float
    state_hash: str
    hitl_required: bool
    hitl_reason: str
    final_report: str

    # Internal budget guard
    _max_cost: float


# ============================================================
# 2. HELPER UTILITIES
# ============================================================

def _audit(state: MiroFishState, step: str, status: str, details: Any = None) -> None:
    state.setdefault("audit_log", [])
    state["audit_log"].append({"step": step, "status": status, "details": details or {}})


def _warn(state: MiroFishState, msg: str) -> None:
    state.setdefault("warnings", [])
    state["warnings"].append(msg)


def _budget_ok(state: MiroFishState, fraction: float = 1.0) -> bool:
    max_cost = state.get("_max_cost", MAX_COST_DEFAULT)
    return state.get("cost_usd", 0.0) < max_cost * fraction


def _track_cost(response, state: MiroFishState) -> float:
    try:
        from litellm import completion_cost
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0
    return state.get("cost_usd", 0.0) + cost


def _data_hash(production_data: Dict[str, Any]) -> str:
    try:
        payload = json.dumps(production_data, sort_keys=True, default=str)
    except Exception:
        payload = str(production_data)
    return hashlib.sha256(payload.encode()).hexdigest()


def _extract_text_fields(production_data: Dict[str, Any]) -> str:
    text_keys = {"description", "notes", "operator_log", "anomaly_description",
                 "comment", "remarks", "log", "message", "summary"}
    parts = []
    for k, v in production_data.items():
        if k.lower() in text_keys and isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v}")
    return "\n".join(parts)


def _extract_base_defect_rate(production_data: Dict[str, Any]) -> float:
    for key in ("defect_rate", "anomaly_score", "patchcore_score", "defect_probability"):
        if key in production_data:
            try:
                return float(production_data[key])
            except (TypeError, ValueError):
                pass
    return 0.1


# ============================================================
# 3. GRAPH NODES
# ============================================================

def parse_input(state: MiroFishState) -> Dict[str, Any]:
    try:
        production_data = state.get("production_data") or {}
        query = (state.get("query") or "").strip()
        scenario_count = state.get("scenario_count") or 1000
        personality_profiles = state.get("personality_profiles") or PERSONALITY_DEFAULTS[:]

        scenario_count = max(10, min(10000, int(scenario_count)))

        valid_profiles = []
        for p in personality_profiles:
            if p in PERSONALITY_PARAMS:
                valid_profiles.append(p)
        if not valid_profiles:
            valid_profiles = PERSONALITY_DEFAULTS[:]

        _audit(state, "parse_input", "ok", {
            "scenario_count": scenario_count,
            "profiles": valid_profiles,
            "data_keys": list(production_data.keys()),
        })
        return {
            "production_data": production_data,
            "query": query,
            "scenario_count": scenario_count,
            "personality_profiles": valid_profiles,
            "warnings": state.get("warnings", []),
            "audit_log": state.get("audit_log", []),
            "cost_usd": state.get("cost_usd", 0.0),
            "error_message": "",
        }
    except Exception as exc:
        _audit(state, "parse_input", "error", {"error": str(exc)})
        return {"error_message": str(exc)}


def build_knowledge_graph(state: MiroFishState) -> Dict[str, Any]:
    try:
        production_data = state.get("production_data") or {}
        text_fields = _extract_text_fields(production_data)

        # Default minimal KG from numeric data
        machine_ids = production_data.get("machine_ids", production_data.get("machines", []))
        if isinstance(machine_ids, str):
            machine_ids = [machine_ids]
        machine_ids = list(machine_ids) if machine_ids else ["machine_01"]

        defect_types = production_data.get("defect_types", ["surface_defect", "dimensional_error"])
        if isinstance(defect_types, str):
            defect_types = [defect_types]

        operators = production_data.get("operators", ["operator_A"])
        if isinstance(operators, str):
            operators = [operators]

        nodes = (
            [{"id": m, "type": "machine"} for m in machine_ids] +
            [{"id": d, "type": "defect"} for d in defect_types] +
            [{"id": o, "type": "operator"} for o in operators]
        )
        edges = []
        for m in machine_ids:
            for d in defect_types:
                edges.append({"from": m, "to": d, "relation": "produces"})
        for o in operators:
            for m in machine_ids:
                edges.append({"from": o, "to": m, "relation": "operates"})

        temporal_chains = production_data.get("temporal_chains", [])
        if not temporal_chains and machine_ids:
            temporal_chains = [{"t0": machine_ids[0], "t1": defect_types[0] if defect_types else "defect"}]

        kg = {"nodes": nodes, "edges": edges, "temporal_chains": temporal_chains}
        kg_summary = f"KG: {len(nodes)} entities, {len(edges)} relations, {len(temporal_chains)} temporal chains."

        # Optionally enrich with LLM if there's unstructured text and budget allows
        if text_fields and _budget_ok(state, fraction=0.5):
            try:
                from litellm import completion
                response = completion(
                    model=MODEL_ANALYTICS,
                    messages=[{"role": "user", "content": (
                        f"Extract entities (machines, defects, operators) and their relations "
                        f"from this factory log. Return compact JSON with keys: entities (list of "
                        f"{{name, type}}), relations (list of {{from, to, type}}).\n\nLog:\n{text_fields[:800]}"
                    )}],
                    max_tokens=400,
                )
                raw = response.choices[0].message.content.strip()
                if raw.startswith("```"):
                    raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
                llm_data = json.loads(raw)
                for ent in llm_data.get("entities", []):
                    nodes.append({"id": ent.get("name", "unknown"), "type": ent.get("type", "entity")})
                for rel in llm_data.get("relations", []):
                    edges.append({"from": rel.get("from", ""), "to": rel.get("to", ""), "relation": rel.get("type", "related")})
                kg = {"nodes": nodes, "edges": edges, "temporal_chains": temporal_chains}
                kg_summary = (
                    f"KG (LLM-enriched): {len(nodes)} entities, {len(edges)} relations. "
                    f"Text sources: {list(production_data.keys())}"
                )
                new_cost = _track_cost(response, state)
                _audit(state, "build_knowledge_graph", "ok_llm", {"nodes": len(nodes), "cost": new_cost})
                return {"knowledge_graph": kg, "kg_summary": kg_summary, "cost_usd": new_cost}
            except Exception as llm_exc:
                _warn(state, f"KG LLM enrichment skipped: {llm_exc}")

        _audit(state, "build_knowledge_graph", "ok", {"nodes": len(nodes), "edges": len(edges)})
        return {"knowledge_graph": kg, "kg_summary": kg_summary}
    except Exception as exc:
        _audit(state, "build_knowledge_graph", "error", {"error": str(exc)})
        _warn(state, f"KG build failed: {exc}")
        return {"knowledge_graph": {"nodes": [], "edges": [], "temporal_chains": []}, "kg_summary": "KG build failed."}


def spawn_agents(state: MiroFishState) -> Dict[str, Any]:
    try:
        scenario_count = state.get("scenario_count", 1000)
        profiles = state.get("personality_profiles") or PERSONALITY_DEFAULTS[:]
        n_agents = min(max(scenario_count // 10, len(profiles)), 50)

        rng = random.Random(_data_hash(state.get("production_data") or {}))

        agent_configs = []
        for i in range(n_agents):
            personality = profiles[i % len(profiles)]
            base = PERSONALITY_PARAMS.get(personality, PERSONALITY_PARAMS["qc_expert"])
            agent_configs.append({
                "id": f"agent_{i:03d}",
                "personality": personality,
                "risk_tolerance": float(base["risk_tolerance"]) + rng.gauss(0, 0.02),
                "speed_bias": float(base["speed_bias"]) + rng.gauss(0, 0.05),
                "quality_threshold": float(base["quality_threshold"]) + rng.gauss(0, 0.01),
            })

        _audit(state, "spawn_agents", "ok", {"n_agents": len(agent_configs)})
        return {"agent_configs": agent_configs}
    except Exception as exc:
        _audit(state, "spawn_agents", "error", {"error": str(exc)})
        _warn(state, f"Agent spawn failed: {exc}")
        return {"agent_configs": []}


def _action_for_rate(rate: float) -> str:
    if rate > 0.7:
        return "emergency_stop"
    if rate > 0.5:
        return "halt_line"
    if rate > 0.3:
        return "inspect_batch"
    return "continue_production"


def _gauss(rng_state: random.Random, mu: float, sigma: float) -> float:
    return rng_state.gauss(mu, sigma)


def run_simulations(state: MiroFishState) -> Dict[str, Any]:
    production_data = state.get("production_data") or {}
    scenario_count = state.get("scenario_count", 1000)
    agent_configs = state.get("agent_configs") or []
    temporal_chains = (state.get("knowledge_graph") or {}).get("temporal_chains", [])
    base_defect_rate = _extract_base_defect_rate(production_data)
    n_steps = max(len(temporal_chains), 1)

    if agent_configs:
        mean_risk = sum(a.get("risk_tolerance", 0.3) for a in agent_configs) / len(agent_configs)
        mean_speed = sum(a.get("speed_bias", 1.0) for a in agent_configs) / len(agent_configs)
    else:
        mean_risk = 0.3
        mean_speed = 1.0

    speed_effect = (mean_speed - 1.0) * 0.05
    noise_scale = mean_risk * 0.1
    seed_val = int(_data_hash(production_data)[:8], 16) % (2**31)

    # Try numpy for vectorised path; fall back to pure Python
    try:
        import numpy as np
        rng = np.random.default_rng(seed_val)
        drift = np.linspace(0, speed_effect, n_steps)
        noise = rng.normal(0.0, noise_scale, size=(scenario_count, n_steps))
        rates = np.clip(base_defect_rate + drift + noise, 0.0, 1.0)
        final_rates: List[float] = rates[:, -1].tolist()
    except ImportError:
        rng_py = random.Random(seed_val)
        drift_step = speed_effect / max(n_steps - 1, 1)
        final_rates = []
        for _ in range(scenario_count):
            r = base_defect_rate
            for step in range(n_steps):
                r = max(0.0, min(1.0, r + step * drift_step + _gauss(rng_py, 0.0, noise_scale)))
            final_rates.append(r)

    sample_size = min(scenario_count, 20)
    step = max(1, scenario_count // sample_size)
    simulation_results = [
        {
            "scenario_id": i,
            "defect_rate": round(final_rates[i], 4),
            "action_taken": _action_for_rate(final_rates[i]),
            "outcome": "defect_detected" if final_rates[i] > 0.3 else "nominal",
        }
        for i in range(0, scenario_count, step)
    ][:sample_size]

    _audit(state, "run_simulations", "ok", {"scenarios": scenario_count, "n_steps": n_steps})
    return {"simulation_results": simulation_results, "_sim_final_rates": final_rates}


def _percentile_py(data: List[float], pct: float) -> float:
    """Pure-Python percentile (linear interpolation, matches numpy default)."""
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = (pct / 100.0) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def aggregate_results(state: MiroFishState) -> Dict[str, Any]:
    production_data = state.get("production_data") or {}
    scenario_count = state.get("scenario_count", 1000)

    raw_rates = state.get("_sim_final_rates")
    if raw_rates:
        rates = [float(r) for r in raw_rates]
    elif state.get("simulation_results"):
        rates = [float(r["defect_rate"]) for r in state["simulation_results"]]
    else:
        rates = [_extract_base_defect_rate(production_data)]

    n = len(rates)
    mean_rate = sum(rates) / n
    variance = sum((r - mean_rate) ** 2 for r in rates) / max(n, 1)
    std_rate = variance ** 0.5

    try:
        import numpy as np
        arr = np.array(rates, dtype=float)
        p5 = float(np.percentile(arr, 5))
        p95 = float(np.percentile(arr, 95))
    except ImportError:
        p5 = _percentile_py(rates, 5)
        p95 = _percentile_py(rates, 95)

    confidence = float(max(0.0, min(1.0, 1.0 - std_rate / mean_rate))) if mean_rate > 0 else 0.5

    if p95 > 0.7:
        risk_level = "critical"
    elif p95 > 0.5:
        risk_level = "high"
    elif p95 > 0.3:
        risk_level = "medium"
    else:
        risk_level = "low"

    action_counts: Dict[str, int] = {}
    for r in (state.get("simulation_results") or []):
        a = r.get("action_taken", "continue_production")
        action_counts[a] = action_counts.get(a, 0) + 1
    if action_counts:
        recommended_action = max(action_counts, key=lambda k: action_counts[k])
    else:
        recommended_action = "inspect_batch" if mean_rate > 0.3 else "continue_production"

    visual_evidence = []
    for key in ("image_paths", "images", "evidence_files", "frames", "snapshots"):
        val = production_data.get(key)
        if isinstance(val, list):
            visual_evidence.extend(str(v) for v in val)
        elif isinstance(val, str) and val:
            visual_evidence.append(val)

    horizon_hours = production_data.get("forecast_hours", production_data.get("horizon_hours", 8))
    try:
        horizon_hours = int(horizon_hours)
    except (TypeError, ValueError):
        horizon_hours = 8
    forecast_horizon = f"{horizon_hours}h"

    simulation_stats = {
        "scenario_count": scenario_count,
        "mean_defect_rate": round(mean_rate, 4),
        "std_defect_rate": round(std_rate, 4),
        "p5_defect_rate": round(p5, 4),
        "p95_defect_rate": round(p95, 4),
        "risk_level": risk_level,
        "recommended_action": recommended_action,
        "visual_evidence_count": len(visual_evidence),
    }

    _audit(state, "aggregate_results", "ok", {"mean": mean_rate, "std": std_rate, "risk": risk_level})
    return {
        "defect_probability": round(mean_rate, 4),
        "recommended_action": recommended_action,
        "visual_evidence": visual_evidence,
        "risk_level": risk_level,
        "forecast_horizon": forecast_horizon,
        "confidence": round(confidence, 4),
        "simulation_stats": simulation_stats,
    }


def generate_report(state: MiroFishState) -> Dict[str, Any]:
    try:
        kg_summary = state.get("kg_summary", "")
        sim_stats = state.get("simulation_stats") or {}
        query = state.get("query", "")
        confidence = state.get("confidence", 0.5)
        defect_prob = state.get("defect_probability", 0.0)
        recommended_action = state.get("recommended_action", "inspect_batch")
        risk_level = state.get("risk_level", "medium")

        hitl_required = confidence < 0.4
        hitl_reason = "Low confidence simulation — human review recommended." if hitl_required else ""

        # Try LLM synthesis if budget allows
        if _budget_ok(state):
            try:
                from litellm import completion
                prompt = (
                    f"You are a factory QC forecasting AI. Synthesize the following simulation results "
                    f"into a concise actionable report for the production floor.\n\n"
                    f"Query: {query}\n"
                    f"Knowledge Graph: {kg_summary}\n"
                    f"Simulation Stats: {json.dumps(sim_stats, default=str)}\n"
                    f"Defect Probability: {defect_prob:.1%}\n"
                    f"Risk Level: {risk_level.upper()}\n"
                    f"Recommended Action: {recommended_action}\n\n"
                    f"Write a 3-5 sentence report. Be specific and actionable."
                )
                response = completion(
                    model=MODEL_ANALYTICS,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=350,
                )
                final_report = response.choices[0].message.content.strip()
                new_cost = _track_cost(response, state)
                _audit(state, "generate_report", "ok_llm", {"cost": new_cost})
                return {
                    "final_report": final_report,
                    "recommended_action": recommended_action,
                    "hitl_required": hitl_required,
                    "hitl_reason": hitl_reason,
                    "cost_usd": new_cost,
                }
            except Exception as llm_exc:
                _warn(state, f"Report LLM failed: {llm_exc}")

        # Fallback: deterministic report from stats
        final_report = (
            f"AndrewSim Forecast for: '{query}'\n"
            f"Based on {sim_stats.get('scenario_count', '?')} simulated scenarios, "
            f"the estimated defect probability is {defect_prob:.1%} "
            f"(95th percentile: {sim_stats.get('p95_defect_rate', '?'):.1%} "
            f"if available).\n"
            f"Risk Level: {risk_level.upper()}. "
            f"Recommended Action: {recommended_action}.\n"
            f"{kg_summary}"
        )
        _audit(state, "generate_report", "ok_deterministic", {})
        return {
            "final_report": final_report,
            "recommended_action": recommended_action,
            "hitl_required": hitl_required,
            "hitl_reason": hitl_reason,
        }
    except Exception as exc:
        _audit(state, "generate_report", "error", {"error": str(exc)})
        _warn(state, f"Report generation failed: {exc}")
        return {
            "final_report": f"Simulation complete. Defect probability: {state.get('defect_probability', 0):.1%}. Action: {state.get('recommended_action', 'inspect_batch')}.",
            "hitl_required": True,
            "hitl_reason": str(exc),
        }


def finalize_state(state: MiroFishState) -> Dict[str, Any]:
    payload = json.dumps({
        "query": state.get("query", ""),
        "defect_probability": state.get("defect_probability", 0.0),
        "risk_level": state.get("risk_level", ""),
        "final_report": (state.get("final_report") or "")[:200],
    }, sort_keys=True)
    state_hash = hashlib.sha256(payload.encode()).hexdigest()
    _audit(state, "finalize_state", "ok", {"hash": state_hash[:12]})
    return {"state_hash": state_hash}


# ============================================================
# 4. GRAPH WIRING
# ============================================================

_mirofish_graph = None


def _get_graph():
    global _mirofish_graph
    if _mirofish_graph is not None:
        return _mirofish_graph

    from langgraph.graph import START, END, StateGraph

    workflow = StateGraph(MiroFishState)

    workflow.add_node("parse_input", parse_input)
    workflow.add_node("build_knowledge_graph", build_knowledge_graph)
    workflow.add_node("spawn_agents", spawn_agents)
    workflow.add_node("run_simulations", run_simulations)
    workflow.add_node("aggregate_results", aggregate_results)
    workflow.add_node("generate_report", generate_report)
    workflow.add_node("finalize_state", finalize_state)

    workflow.add_edge(START, "parse_input")
    workflow.add_edge("parse_input", "build_knowledge_graph")
    workflow.add_edge("build_knowledge_graph", "spawn_agents")
    workflow.add_edge("spawn_agents", "run_simulations")
    workflow.add_edge("run_simulations", "aggregate_results")
    workflow.add_edge("aggregate_results", "generate_report")
    workflow.add_edge("generate_report", "finalize_state")
    workflow.add_edge("finalize_state", END)

    _mirofish_graph = workflow.compile()
    return _mirofish_graph


# ============================================================
# 5. RESULT CLASS
# ============================================================

class MiroFishResult:
    """Structured output — same public API as AndrewResult for bridge compatibility."""

    def __init__(self, state: MiroFishState):
        self.goal = state.get("query", "")
        self.output = state.get("final_report", "")
        self.error = state.get("error_message") or None
        self.cost = state.get("cost_usd", 0.0)
        self.confidence = state.get("confidence", 0.0)
        self.warnings = state.get("warnings", [])
        self.audit_log = state.get("audit_log", [])
        self.state_hash = state.get("state_hash", "")
        self.routing = "mirofish"
        self.hitl_required = state.get("hitl_required", False)
        self.hitl_reason = state.get("hitl_reason", "")

        # MiroFish-specific
        self.defect_probability = state.get("defect_probability", 0.0)
        self.recommended_action = state.get("recommended_action", "inspect_batch")
        self.visual_evidence = state.get("visual_evidence", [])
        self.risk_level = state.get("risk_level", "medium")
        self.simulation_stats = state.get("simulation_stats", {})
        self.forecast_horizon = state.get("forecast_horizon", "8h")

        # Bridge compatibility
        self.sql_query = None
        self.python_code = None

    @property
    def success(self) -> bool:
        return not self.error and bool(self.output)

    @property
    def cost_usd(self) -> float:
        return self.cost

    @property
    def error_message(self) -> Optional[str]:
        return self.error

    def to_roma_output(self) -> str:
        parts = [f"## MiroFish Forecast: {self.goal}"]
        if self.output:
            parts.append(self.output[:3000])
        parts.append(
            f"\n**Defect Probability:** {self.defect_probability:.1%} | "
            f"**Risk:** {self.risk_level.upper()} | "
            f"**Action:** {self.recommended_action}"
        )
        if self.warnings:
            parts.append("\n**Warnings:**\n" + "\n".join(f"- {w}" for w in self.warnings))
        if self.error:
            parts.append(f"\n**Error:** {self.error}")
        parts.append(
            f"\n<metadata confidence=\"{self.confidence:.2f}\" cost=\"${self.cost:.4f}\" "
            f"route=\"{self.routing}\" hash=\"{self.state_hash[:12]}\" "
            f"hitl=\"{self.hitl_required}\" />"
        )
        return "\n".join(parts)

    def __str__(self):
        s = "OK" if self.success else "FAIL"
        return (
            f"[{s}] MiroFish ({self.risk_level}) | {self.goal}\n"
            f"  Defect P: {self.defect_probability:.1%} | Action: {self.recommended_action}\n"
            f"  Output: {(self.output or '-')[:120]}\n"
            f"  Confidence: {self.confidence:.2f} | Cost: ${self.cost:.4f} | Warnings: {len(self.warnings)}"
        )


# ============================================================
# 6. EXECUTOR CLASS
# ============================================================

class MiroFishExecutor:
    def __init__(self, max_cost: float = 1.0):
        self.max_cost = max_cost
        self.model = MODEL_ANALYTICS

    def execute(
        self,
        query: str,
        production_data: Optional[Dict] = None,
        scenario_count: int = 1000,
        personality_profiles: Optional[List[str]] = None,
    ) -> MiroFishResult:
        logger.info(f"MiroFishExecutor v1.0.0 | {query[:80]}")
        initial_state: MiroFishState = {
            "query": query,
            "production_data": production_data or {},
            "scenario_count": scenario_count,
            "personality_profiles": personality_profiles or [],
            "cost_usd": 0.0,
            "warnings": [],
            "audit_log": [],
            "error_message": "",
            "_max_cost": self.max_cost,
        }
        try:
            final_state = _get_graph().invoke(initial_state)
        except Exception as exc:
            logger.warning(f"LangGraph unavailable ({exc}), running nodes directly")
            final_state = dict(initial_state)
            for node_fn in (
                parse_input, build_knowledge_graph, spawn_agents,
                run_simulations, aggregate_results, generate_report, finalize_state,
            ):
                try:
                    patch = node_fn(final_state)
                    if patch:
                        final_state.update(patch)
                except Exception as node_exc:
                    logger.error(f"Node {node_fn.__name__} failed: {node_exc}")
                    final_state.setdefault("warnings", []).append(
                        f"{node_fn.__name__} error: {node_exc}"
                    )
        return MiroFishResult(final_state)


# ============================================================
# 7. CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="MiroFish Swarm Forecaster CLI")
    parser.add_argument("--data", "-d", default="{}", help="Production data JSON string or file path")
    parser.add_argument("--query", "-q", default="forecast defect probability", help="Forecast query")
    parser.add_argument("--scenarios", "-n", type=int, default=1000, help="Number of scenarios")
    args = parser.parse_args()

    import pathlib
    production_data = {}
    if args.data and args.data.strip() != "{}":
        p = pathlib.Path(args.data)
        if p.exists():
            with open(p) as f:
                production_data = json.load(f)
        else:
            try:
                production_data = json.loads(args.data)
            except json.JSONDecodeError:
                print(f"[ERROR] --data must be valid JSON or a file path", file=sys.stderr)
                sys.exit(1)

    executor = MiroFishExecutor()
    result = executor.execute(
        query=args.query,
        production_data=production_data,
        scenario_count=args.scenarios,
    )
    print(result.to_roma_output())
    print(f"\n[Cost: ${result.cost:.4f}]")
    sys.exit(0 if result.success else 1)
