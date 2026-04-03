"""
SwarmSupervisor v1.0.0
===================================================
Top-level LangGraph router that dispatches queries to:
  - Andrew Swarm (analytical: SQL → Python → sandbox)
  - Romeo PhD   (educational: concept explanation, examples, takeaways)
  - Both         (hybrid: analytical + educational in one response)

Sprint 8: Multi-Agent Supervisor.

Classification is keyword-based (no LLM call) — zero latency overhead
for pure analytical queries that already ran through Andrew.

Graph:
  START → classify_query
            ├─ andrew → run_andrew ────────────────► fuse_results → END
            ├─ romeo  → run_romeo  ────────────────► fuse_results → END
            └─ both   → run_andrew → run_romeo ────► fuse_results → END
"""

import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, TypedDict

logger = logging.getLogger("supervisor")


# ============================================================
# 1. CLASSIFICATION SIGNALS
# ============================================================

EDUCATIONAL_SIGNALS: List[str] = [
    "explain", "what is", "what are", "how does", "how do",
    "teach me", "why does", "why is", "define", "definition",
    "tutorial", "introduction to", "concept", "difference between",
    "vs", "versus", "compare", "comparison", "example of",
    "understand", "overview", "beginner", "primer",
    "in simple terms", "break down", "tell me about",
]

# DATA_REQUEST_SIGNALS — signals that mean "I want you to query / analyse actual
# data from a database". Math vocabulary alone (mean, cagr, etc.) is NOT a
# data-request signal; it may just be educational vocabulary.
DATA_REQUEST_SIGNALS: List[str] = [
    "show me", "total", "revenue", "how much", "how many",
    "sum", "count", "group by", "bar chart", "pie chart",
    "table", "rows", "data", "database", "chart", "graph",
    "calculate", "compute", "forecast", "predict", "run analysis",
    "query", "sql", "analyse", "analyze", "report", "breakdown",
    "by region", "by month", "by quarter", "per day", "per week",
]

# ANALYTICAL_SIGNALS — broader set for the no-educational-hit path.
ANALYTICAL_SIGNALS: List[str] = DATA_REQUEST_SIGNALS + [
    "average", "mean", "median", "mode", "trend", "variance", "correlation",
    "regression", "simulation", "arima", "lstm", "prophet", "monte carlo",
    "machine learning", "neural network", "time series",
]


def _classify(query: str) -> tuple[str, List[str]]:
    """
    Classify a query into 'andrew', 'romeo', or 'both'.
    Returns (decision, matched_signals).

    Decision rule:
    - Educational intent + explicit data request → "both"
    - Educational intent only               → "romeo"
    - Everything else (default)             → "andrew"

    "Both" requires BOTH an educational signal AND a data-request signal.
    Math vocabulary alone (cagr, mean, etc.) appearing in an explanatory
    question does NOT trigger "both" — it is just vocabulary context.
    """
    q = (query or "").lower().strip()
    edu_hits = [s for s in EDUCATIONAL_SIGNALS if s in q]
    data_hits = [s for s in DATA_REQUEST_SIGNALS if s in q]
    ana_hits = [s for s in ANALYTICAL_SIGNALS if s in q]

    if edu_hits and data_hits:
        return "both", edu_hits + data_hits
    if edu_hits:
        return "romeo", edu_hits
    return "andrew", ana_hits  # default to andrew (safe fallback)


# ============================================================
# 2. STATE
# ============================================================

class SupervisorState(TypedDict, total=False):
    # Input
    query: str
    db_url: str
    schema_context: Dict[str, Dict[str, str]]

    # Routing
    agent_decision: str       # "andrew" | "romeo" | "both"
    classification_signals: List[str]

    # Results (stored as dicts to avoid TypedDict nesting issues)
    andrew_result: Optional[Any]   # AndrewResult instance
    romeo_result: Optional[Any]    # RomeoResult instance

    # Final output
    final_output: str
    confidence: float
    cost_usd: float
    warnings: List[str]
    error_message: str
    state_hash: str
    agent_used: str


# ============================================================
# 3. RESULT CLASS
# ============================================================

class SupervisorResult:
    """
    Unified result — same public surface as AndrewResult so the bridge
    needs zero changes beyond swapping the import.
    """

    def __init__(self, state: SupervisorState):
        self.goal = state.get("query", "")
        self.output = state.get("final_output", "")
        self.error = state.get("error_message")
        self.cost = state.get("cost_usd", 0.0)
        self.confidence = state.get("confidence", 0.0)
        self.warnings = state.get("warnings", [])
        self.state_hash = state.get("state_hash", "")
        self.routing = state.get("agent_decision", "andrew")
        self.agent_used = state.get("agent_used", "andrew")
        self.model_used = self._infer_model(state)

        # Bridge compatibility
        self.sql_query = self._extract_sql(state)
        self.python_code = None
        self.audit_log = self._merge_audit_logs(state)

    def _infer_model(self, state: SupervisorState) -> str:
        ar = state.get("andrew_result")
        rr = state.get("romeo_result")
        models = []
        if ar and hasattr(ar, "model_used") and ar.model_used:
            models.append(ar.model_used)
        if rr and hasattr(rr, "model_used") and rr.model_used:
            models.append(rr.model_used)
        return " + ".join(models) if models else "unknown"

    def _extract_sql(self, state: SupervisorState) -> Optional[str]:
        ar = state.get("andrew_result")
        if ar and hasattr(ar, "sql_query"):
            return ar.sql_query
        return None

    def _merge_audit_logs(self, state: SupervisorState) -> List[Dict]:
        logs = []
        ar = state.get("andrew_result")
        rr = state.get("romeo_result")
        if ar and hasattr(ar, "audit_log"):
            logs.extend(ar.audit_log or [])
        if rr and hasattr(rr, "audit_log"):
            logs.extend(rr.audit_log or [])
        return logs

    @property
    def success(self) -> bool:
        return not self.error and bool(self.output)

    # Bridge uses result.cost_usd in one place and result.cost in another;
    # expose both to keep it compatible.
    @property
    def cost_usd(self) -> float:
        return self.cost

    @property
    def error_message(self) -> Optional[str]:
        return self.error

    def to_roma_output(self) -> str:
        parts = [f"## Result: {self.goal}"]
        if self.output:
            parts.append(self.output[:3000])
        if self.warnings:
            parts.append("\n**Warnings:**\n" + "\n".join(f"- {w}" for w in self.warnings))
        if self.error:
            parts.append(f"\n**Error:** {self.error}")
        parts.append(
            f"\n<metadata confidence=\"{self.confidence:.2f}\" cost=\"${self.cost:.4f}\" "
            f"route=\"{self.routing}\" agent=\"{self.agent_used}\" hash=\"{self.state_hash[:12]}\" />"
        )
        return "\n".join(parts)

    def __str__(self):
        s = "OK" if self.success else "FAIL"
        return (
            f"[{s}] Supervisor ({self.agent_used}) | {self.goal}\n"
            f"  Route: {self.routing} → {self.model_used}\n"
            f"  Output: {(self.output or '-')[:120]}\n"
            f"  Confidence: {self.confidence:.2f} | Cost: ${self.cost:.4f} | Warnings: {len(self.warnings)}"
        )


# ============================================================
# 4. GRAPH NODES
# ============================================================

def classify_query(state: SupervisorState) -> Dict[str, Any]:
    """Keyword-based routing — no LLM call."""
    query = state.get("query", "")
    decision, signals = _classify(query)
    logger.info(f"Supervisor: '{query[:60]}' → {decision} (signals: {signals[:5]})")
    return {
        "agent_decision": decision,
        "classification_signals": signals,
        "warnings": state.get("warnings", []),
        "cost_usd": state.get("cost_usd", 0.0),
    }


def run_andrew(state: SupervisorState) -> Dict[str, Any]:
    """Execute the Andrew analytical pipeline."""
    from core.andrew_swarm import AndrewExecutor
    executor = AndrewExecutor(db_url=state.get("db_url", ""))
    if state.get("schema_context"):
        executor._schema = state["schema_context"]

    result = executor.execute(state.get("query", ""))
    logger.info(f"Andrew done: success={result.success}, cost=${result.cost:.4f}")

    new_cost = state.get("cost_usd", 0.0) + result.cost
    warnings = list(state.get("warnings", []))
    warnings.extend(result.warnings or [])

    return {
        "andrew_result": result,
        "cost_usd": new_cost,
        "warnings": warnings,
    }


def run_romeo(state: SupervisorState) -> Dict[str, Any]:
    """Execute the Romeo educational pipeline."""
    from core.romeo_swarm import RomeoExecutor
    executor = RomeoExecutor()
    result = executor.execute(state.get("query", ""))
    logger.info(f"Romeo done: success={result.success}, cost=${result.cost:.4f}")

    new_cost = state.get("cost_usd", 0.0) + result.cost
    warnings = list(state.get("warnings", []))
    warnings.extend(result.warnings or [])

    return {
        "romeo_result": result,
        "cost_usd": new_cost,
        "warnings": warnings,
    }


def fuse_results(state: SupervisorState) -> Dict[str, Any]:
    """Merge one or both agent results into a single final_output."""
    ar = state.get("andrew_result")
    rr = state.get("romeo_result")

    parts = []
    error_parts = []
    confidences = []
    agent_labels = []

    if ar is not None:
        agent_labels.append("andrew")
        if ar.success:
            parts.append(f"### Analytical Answer\n{ar.output}")
            confidences.append(ar.confidence)
        elif ar.error:
            error_parts.append(f"Andrew error: {ar.error}")

    if rr is not None:
        agent_labels.append("romeo")
        if rr.success:
            parts.append(f"### Educational Explanation\n{rr.output}")
            confidences.append(rr.confidence)
        elif rr.error:
            error_parts.append(f"Romeo error: {rr.error}")

    final_output = "\n\n".join(parts)
    error_message = "; ".join(error_parts) if not parts else ""

    # Conservative confidence: take the minimum of successful agents
    confidence = min(confidences) if confidences else 0.0

    # Compute state hash
    payload = json.dumps({
        "query": state.get("query", ""),
        "agent_decision": state.get("agent_decision", ""),
        "final_output": final_output[:500],
    }, sort_keys=True)
    state_hash = hashlib.sha256(payload.encode()).hexdigest()

    return {
        "final_output": final_output,
        "confidence": confidence,
        "error_message": error_message,
        "state_hash": state_hash,
        "agent_used": " + ".join(agent_labels) if agent_labels else "none",
    }


# ============================================================
# 5. CONDITIONAL EDGE FUNCTIONS
# ============================================================

def _route_from_classify(state: SupervisorState) -> str:
    return state.get("agent_decision", "andrew")


def _after_andrew(state: SupervisorState) -> str:
    """After running Andrew: go to romeo if 'both', otherwise fuse."""
    if state.get("agent_decision") == "both":
        return "run_romeo"
    return "fuse_results"


# ============================================================
# 6. GRAPH WIRING (lazy compile — avoids langgraph import at module load)
# ============================================================

_supervisor_graph = None


def _get_checkpointer():
    """
    Return a MongoDB checkpointer when MONGODB_URI is set, else None.

    A checkpointer gives LangGraph persistent conversation state across
    restarts.  Each thread_id maps to a saved graph snapshot in MongoDB
    (collection: langgraph_checkpoints).

    Requires: langgraph-checkpoint-mongodb (pip install langgraph-checkpoint-mongodb)
    """
    uri = os.getenv("MONGODB_URI", "")
    if not uri:
        return None
    try:
        from langgraph.checkpoint.mongodb import MongoDBSaver
        db_name = os.getenv("MONGODB_DB", "romeoflexvision")
        checkpointer = MongoDBSaver.from_conn_string(uri, db_name=db_name)
        logger.info("LangGraph supervisor: MongoDB checkpointer active")
        return checkpointer
    except Exception as exc:
        logger.warning(
            f"MongoDB checkpointer init failed ({exc}); "
            "running stateless (no session persistence)"
        )
        return None


def _get_graph():
    """Compile the LangGraph supervisor on first call."""
    global _supervisor_graph
    if _supervisor_graph is not None:
        return _supervisor_graph

    from langgraph.graph import START, END, StateGraph

    workflow = StateGraph(SupervisorState)

    workflow.add_node("classify_query", classify_query)
    workflow.add_node("run_andrew", run_andrew)
    workflow.add_node("run_romeo", run_romeo)
    workflow.add_node("fuse_results", fuse_results)

    workflow.add_edge(START, "classify_query")

    workflow.add_conditional_edges("classify_query", _route_from_classify, {
        "andrew": "run_andrew",
        "romeo": "run_romeo",
        "both": "run_andrew",    # both: andrew first, then romeo
    })

    workflow.add_conditional_edges("run_andrew", _after_andrew, {
        "run_romeo": "run_romeo",
        "fuse_results": "fuse_results",
    })

    workflow.add_edge("run_romeo", "fuse_results")
    workflow.add_edge("fuse_results", END)

    checkpointer = _get_checkpointer()
    _supervisor_graph = workflow.compile(checkpointer=checkpointer)
    return _supervisor_graph


# ============================================================
# 7. SUPERVISOR CLASS
# ============================================================

class SwarmSupervisor:
    """
    Drop-in replacement for AndrewExecutor in the Moltis bridge.
    Same interface: execute(goal: str) → result with .success, .output, etc.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "")
        self._schema: Optional[Dict] = None
        self._andrew_executor = None

    @property
    def schema(self) -> Dict[str, Dict[str, str]]:
        if self._schema is None:
            from core.andrew_swarm import discover_schema
            self._schema = discover_schema(self.db_url)
            logger.info(f"Schema: {list(self._schema.keys())}")
        return self._schema

    def _get_andrew_executor(self):
        if self._andrew_executor is None:
            from core.andrew_swarm import AndrewExecutor

            self._andrew_executor = AndrewExecutor(db_url=self.db_url)
        if self._schema is not None:
            self._andrew_executor._schema = self._schema
        return self._andrew_executor

    def available_tools(self) -> List[str]:
        return self._get_andrew_executor().available_tools()

    async def get_tool_prompts(self) -> Dict[str, str]:
        return await self._get_andrew_executor().get_tool_prompts()

    async def run_tool_calls(
        self,
        calls,
        *,
        working_directory: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        return await self._get_andrew_executor().run_tool_calls(
            calls,
            working_directory=working_directory,
            metadata=metadata,
        )

    def run_tool_calls_sync(
        self,
        calls,
        *,
        working_directory: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        return self._get_andrew_executor().run_tool_calls_sync(
            calls,
            working_directory=working_directory,
            metadata=metadata,
        )

    def execute(self, goal: str) -> SupervisorResult:
        logger.info(f"SwarmSupervisor v1.0.0 | {goal[:80]}")
        state = _get_graph().invoke({
            "query": goal,
            "db_url": self.db_url,
            "schema_context": self.schema,
            "cost_usd": 0.0,
            "warnings": [],
            "error_message": "",
        })
        return SupervisorResult(state)

    def invalidate_schema(self):
        self._schema = None
        if self._andrew_executor is not None:
            self._andrew_executor.invalidate_schema()


# ============================================================
# 8. CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        db_url = os.getenv("DATABASE_URL", "")
        print("=" * 70)
        print(f"SWARM SUPERVISOR v1.0.0 — LIVE QUERY")
        print(f"DB: {db_url or '(none)'}")
        print("=" * 70)
        sup = SwarmSupervisor(db_url=db_url or None)
        result = sup.execute(query)
        print("\n" + result.to_roma_output())
    else:
        # Smoke test: classification decisions
        test_queries = [
            "total revenue by region last quarter",
            "explain what CAGR means",
            "explain ARIMA and forecast next quarter revenue",
            "difference between mean and median",
            "show me top 5 products by quantity",
            "",
        ]
        print("=" * 70)
        print("SUPERVISOR CLASSIFICATION SMOKE TEST")
        print("=" * 70)
        for q in test_queries:
            decision, signals = _classify(q)
            label = q[:55] + "..." if len(q) > 55 else (q or "(empty)")
            print(f"  [{decision:6}] {label}")
            if signals:
                print(f"           signals: {signals[:4]}")
