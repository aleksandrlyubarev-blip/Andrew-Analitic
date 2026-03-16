"""
Andrew Swarm Core v1.0.0-rc1
===================================================
Single-file drop-in: state → routing → intent → SQL → validate → execute
→ Python → AST safety → sandbox → Pandera → semantic guardrails → finalize

Sprint 4 additions over v0.3:
- Weighted keyword routing (48 terms, score-based, 3 lanes)
- Model registry via env vars (swap providers without code changes)
- Provider parameter adapter (Grok vs Claude vs OpenAI quirks)
- Routing audit trail (matched terms, score, lane decision)
"""

import ast
import hashlib
import json
import logging
import os
import re
import tempfile
from typing import Any, Dict, List, Optional, Tuple, TypedDict

import pandas as pd
import pandera.pandas as pa
from pandera import Check
import sqlglot
import sqlglot.expressions as exp
from sqlglot.optimizer.qualify import qualify

from langgraph.graph import START, END, StateGraph
from langgraph.types import RetryPolicy
from litellm import completion, completion_cost
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("andrew")


# ============================================================
# 1. STATE
# ============================================================

class AndrewState(TypedDict, total=False):
    # Input
    user_request: str
    goal: str  # Alias for ROMA compatibility
    schema_context: Dict[str, Dict[str, str]]
    db_url: str

    # Routing
    python_model: str
    orchestrator_model: str
    routing_decision: str
    routing_score: int
    routing_hits: List[str]

    # Intent
    intent_contract: Dict[str, Any]

    # SQL pipeline
    sql_query: str
    sql_validated: bool
    sql_result_path: str
    query_results: List[Dict[str, Any]]

    # Python pipeline
    python_code: str
    python_validated: bool
    sandbox_output_path: str
    sandbox_output: str

    # Quality
    confidence: float
    warnings: List[str]
    audit_log: List[Dict[str, Any]]

    # Control
    error_message: str
    cost_usd: float
    state_hash: str

    # HITL escalation
    hitl_required: bool
    hitl_reason: str

    # Semantic routing (Sprint 5)
    routing_log: Dict[str, Any]
    session_id: str
    session_summary: str
    session_length: int
    memory_records_retrieved: int


class AndrewResult:
    """Structured output for ROMA bridge and human consumption."""

    def __init__(self, state: AndrewState):
        self.goal = state.get("user_request", state.get("goal", ""))
        self.sql_query = state.get("sql_query")
        self.python_code = state.get("python_code")
        self.query_results = state.get("query_results", [])
        self.output = state.get("sandbox_output", "") or str(self.query_results or "")
        self.error = state.get("error_message")
        self.cost = state.get("cost_usd", 0.0)
        self.confidence = state.get("confidence", 0.0)
        self.warnings = state.get("warnings", [])
        self.audit_log = state.get("audit_log", [])
        self.state_hash = state.get("state_hash", "")
        self.routing = state.get("routing_decision", "")
        self.routing_log = state.get("routing_log", {})
        self.model_used = state.get("python_model", "")
        self.hitl_required = state.get("hitl_required", False)
        self.hitl_reason = state.get("hitl_reason", "")

    @property
    def success(self) -> bool:
        return not self.error and bool(self.output)

    def to_roma_output(self) -> str:
        parts = [f"## Analysis: {self.goal}"]
        if self.output:
            parts.append(self.output[:3000])
        if self.warnings:
            parts.append("\n**Warnings:**\n" + "\n".join(f"- {w}" for w in self.warnings))
        if self.error:
            parts.append(f"\n**Error:** {self.error}")
        parts.append(
            f"\n<metadata confidence=\"{self.confidence:.2f}\" cost=\"${self.cost:.4f}\" "
            f"route=\"{self.routing}\" hash=\"{self.state_hash[:12]}\" />"
        )
        return "\n".join(parts)

    def __str__(self):
        s = "OK" if self.success else "FAIL"
        return (
            f"[{s}] {self.goal}\n"
            f"  Route: {self.routing} → {self.model_used}\n"
            f"  SQL: {(self.sql_query or '-')[:80]}\n"
            f"  Output: {(self.output or '-')[:120]}\n"
            f"  Confidence: {self.confidence:.2f} | Cost: ${self.cost:.4f} | Warnings: {len(self.warnings)}"
        )


# ============================================================
# 2. MODEL REGISTRY & PROVIDER ADAPTER
# ============================================================

MODEL_REGISTRY = {
    "reasoning_math": os.getenv("MODEL_REASONING_MATH", "grok-4"),
    "default_orchestrator": os.getenv("MODEL_ORCHESTRATOR", "anthropic/claude-sonnet-4-20250514"),
    "default_python": os.getenv("MODEL_PYTHON", "anthropic/claude-sonnet-4-20250514"),
    "analytics_fastlane": os.getenv("MODEL_ANALYTICS", "gpt-4o-mini"),
    "sql_generation": os.getenv("MODEL_SQL", "gpt-4o-mini"),
}


def build_model_params(model_name: str) -> dict:
    """
    Provider-specific parameter adapter.
    Different models support different params — blindly passing
    the same bundle causes errors (e.g., Grok rejects 'effort').
    """
    name = (model_name or "").lower()

    if "grok" in name:
        # xAI reasoning models: limited parameter support
        return {"temperature": 0.2}

    if "claude" in name:
        return {"temperature": 0.0}

    if "gpt" in name or "openai" in name:
        return {"temperature": 0.0}

    # Fallback for unknown providers
    return {"temperature": 0.2}


# ============================================================
# 3. WEIGHTED QUERY ROUTER (Sprint 4)
# ============================================================

# Weights: 1=basic analytics, 2=intermediate stats, 3=advanced, 4=heavy ML/math
MATH_KEYWORDS: Dict[str, int] = {
    # Basic analytics (weight 1)
    "average": 1, "mean": 1, "median": 1, "mode": 1, "trend": 1,
    # Intermediate stats (weight 2)
    "std": 2, "standard deviation": 2, "variance": 2, "covariance": 2,
    "correlation": 2, "seasonality": 2, "probability": 2, "hypothesis": 2,
    "significance": 2, "minimize": 2, "maximize": 2, "matrix": 2,
    "roi": 2, "cagr": 2, "distribution": 2, "statistical": 2, "quantitative": 2,
    # Advanced (weight 3)
    "regression": 3, "forecast": 3, "predict": 3, "prediction": 3,
    "extrapolate": 3, "interpolate": 3, "time series": 3, "chi-square": 3,
    "t-test": 3, "anova": 3, "p-value": 3, "confidence interval": 3,
    "bayesian": 3, "optimize": 3, "npv": 3, "irr": 3, "simulate": 3,
    "simulation": 3,
    # Heavy ML/math (weight 4)
    "arima": 4, "lstm": 4, "prophet": 4, "linear programming": 4,
    "calculus": 4, "derivative": 4, "integral": 4, "eigenvalue": 4,
    "monte carlo": 4, "machine learning": 4, "neural network": 4,
    "fit model": 4, "curve fitting": 4,
}

LIGHT_ANALYTICS_TERMS = {
    "sum", "count", "group by", "bar chart", "pie chart", "region",
    "month", "quarter", "daily", "weekly", "table", "show", "display",
}

HEAVY_ML_TERMS = {
    "arima", "lstm", "prophet", "monte carlo", "regression",
    "machine learning", "neural network", "linear programming",
    "fit model", "curve fitting", "forecast", "predict",
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_keywords(request: str) -> List[Tuple[str, int]]:
    hits = []
    padded = f" {request} "
    for kw, weight in MATH_KEYWORDS.items():
        if f" {kw} " in padded or kw in request:
            hits.append((kw, weight))
    return hits


def route_query_intent(state: AndrewState) -> Dict[str, Any]:
    """
    Semantic routing (Sprint 5 §4) — embedding-based cosine similarity against
    the Capability Registry.  Falls back to keyword scoring when embeddings are
    unavailable (no API key, test environment, etc.).

    Routing lanes:
      reasoning_math      — heavy ML / forecasting / simulation
      standard_analytics  — SQL, cohorts, BI reports
      analytics_fastlane  — simple aggregations, quick charts
    """
    from core.semantic_router import _semantic_router  # lazy import avoids circular

    request = _normalize(state.get("user_request", state.get("goal", "")))

    routing_log = _semantic_router.route(
        query=request,
        session_summary=state.get("session_summary"),
        session_length=state.get("session_length", 0),
        memory_records_retrieved=state.get("memory_records_retrieved", 0),
    )
    route = routing_log.selected_agent

    # Map registry agent_id → model registry key
    _model_map = {
        "reasoning_math":     MODEL_REGISTRY["reasoning_math"],
        "standard_analytics": MODEL_REGISTRY["default_python"],
        "analytics_fastlane": MODEL_REGISTRY["analytics_fastlane"],
    }
    model = _model_map.get(route, MODEL_REGISTRY["default_python"])

    # Keep legacy keyword fields for backward compatibility
    hits = _match_keywords(request)
    matched_terms = [kw for kw, _ in hits]
    keyword_score = sum(w for _, w in hits)

    logger.info(
        f"Route: {route} (semantic_score={routing_log.top_score:.3f}, "
        f"fallback={routing_log.fallback_used}) → {model}"
    )

    return {
        "python_model": model,
        "orchestrator_model": MODEL_REGISTRY["default_orchestrator"],
        "routing_decision": route,
        "routing_score": keyword_score,
        "routing_hits": matched_terms,
        "routing_log": routing_log.as_dict(),
    }


# ============================================================
# 4. RETRY POLICIES
# ============================================================

llm_retry = RetryPolicy(max_attempts=3, initial_interval=1.0, backoff_factor=2.0)
sandbox_retry = RetryPolicy(max_attempts=2, initial_interval=3.0, backoff_factor=2.5)


# ============================================================
# 5. CONSTANTS & HELPERS
# ============================================================

MAX_COST_USD = float(os.getenv("ANDREW_MAX_COST", "1.00"))
HITL_CONFIDENCE_THRESHOLD = float(os.getenv("HITL_CONFIDENCE_THRESHOLD", "0.35"))

BLOCKED_SQL_KEYWORDS = {
    "DROP", "DELETE", "TRUNCATE", "ALTER", "GRANT", "REVOKE",
    "INSERT", "UPDATE", "CREATE", "ATTACH", "DETACH", "COPY",
    "PRAGMA", "CALL", "EXEC", "EXECUTE",
}
ALLOWED_SQL_STATEMENTS = {"select", "with"}

SUSPICIOUS_PY_MODULES = {"os", "subprocess", "socket", "shutil", "pathlib", "requests"}
SUSPICIOUS_PY_CALLS = {"exec", "eval", "__import__", "compile", "open"}
OVERLAP_SPLIT_PATTERNS = [
    re.compile(r"\btest_size\s*=\s*0\b"),
    re.compile(r"\bshuffle\s*=\s*False\b"),
]


def clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _warn(state: AndrewState, msg: str) -> None:
    state.setdefault("warnings", [])
    state["warnings"].append(msg)


def _audit(state: AndrewState, stage: str, payload: Dict[str, Any]) -> None:
    state.setdefault("audit_log", [])
    state["audit_log"].append({"stage": stage, **payload})


def _fail(state: AndrewState, msg: str, stage: str) -> Dict[str, Any]:
    _audit(state, stage, {"status": "failed", "error": msg})
    return {"error_message": msg}


def _normalize_schema(raw: Optional[Dict]) -> Dict[str, Dict[str, str]]:
    if not raw:
        return {}
    return {
        str(t).lower(): {str(c).lower(): str(d) for c, d in cols.items()}
        for t, cols in raw.items()
    }


def _track_cost(response, state: AndrewState) -> float:
    try:
        cost = completion_cost(completion_response=response)
    except Exception:
        cost = 0.0
    return state.get("cost_usd", 0.0) + cost


def _budget_ok(state: AndrewState) -> bool:
    return state.get("cost_usd", 0.0) < MAX_COST_USD


# ============================================================
# 6. SCHEMA DISCOVERY
# ============================================================

def discover_schema(db_url: str) -> Dict[str, Dict[str, str]]:
    """Auto-discover schema as {table: {column: type}}."""
    fallback = {
        "sales": {"id": "int", "product": "text", "revenue": "float",
                  "quantity": "int", "date": "date", "region": "text"}
    }
    if not db_url:
        return fallback
    try:
        engine = create_engine(db_url)
        schema = {}
        with engine.connect() as conn:
            if "sqlite" in db_url:
                tables = [r[0] for r in conn.execute(text(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"))]
                for t in tables:
                    cols = conn.execute(text(f"PRAGMA table_info('{t}')"))
                    schema[t] = {row[1]: row[2] for row in cols}
            else:
                tables = [r[0] for r in conn.execute(text(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"))]
                for t in tables:
                    cols = conn.execute(text(
                        f"SELECT column_name, data_type FROM information_schema.columns "
                        f"WHERE table_name='{t}' ORDER BY ordinal_position"))
                    schema[t] = {row[0]: row[1] for row in cols}
        return schema or fallback
    except Exception as e:
        logger.warning(f"Schema discovery failed: {e}")
        return fallback


# ============================================================
# 7. NODE: Intent Contract
# ============================================================

def build_intent_contract(state: AndrewState) -> Dict[str, Any]:
    request = _normalize(state.get("user_request", state.get("goal", "")))
    schema = _normalize_schema(state.get("schema_context"))

    contract = {
        "intent_type": "analytics",
        "allowed_tables": sorted(schema.keys()),
        "allowed_metrics": sorted({"total_revenue", "revenue", "count", "avg", "sum", "min", "max"}),
        "allowed_ops": sorted({"filter", "groupby", "aggregate", "sort", "limit"}),
        "must_be_read_only": True,
        "deny_row_level_pii_export": True,
        "must_not_train_model_unless_requested": ("model" in request or "predict" in request or "forecast" in request),
    }
    _audit(state, "build_intent_contract", {"status": "ok", "contract": contract})
    return {"intent_contract": contract}


# ============================================================
# 8. NODE: Generate SQL
# ============================================================

def generate_sql(state: AndrewState) -> Dict[str, Any]:
    if not _budget_ok(state):
        return _fail(state, "Budget exhausted", "generate_sql")

    schema = state.get("schema_context", {})
    schema_str = "\n".join(
        f"  {t}: {', '.join(f'{c} ({d})' for c, d in cols.items())}"
        for t, cols in schema.items()
    )

    model = state.get("orchestrator_model", MODEL_REGISTRY["sql_generation"])
    params = build_model_params(model)

    response = completion(
        model=model,
        messages=[{"role": "user", "content": (
            f"Goal: {state.get('user_request', state.get('goal', ''))}\n"
            f"Database schema:\n{schema_str}\n\n"
            f"Write ONLY a single SELECT or WITH query. No markdown. No explanation."
        )}],
        **params,
    )

    sql = response.choices[0].message.content.strip()
    if sql.startswith("```"):
        sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    new_cost = _track_cost(response, state)
    _audit(state, "generate_sql", {"status": "ok", "model": model, "cost": new_cost})
    return {"sql_query": sql, "cost_usd": new_cost, "error_message": ""}


# ============================================================
# 9. NODE: Validate SQL (hardened — sqlglot qualify + schema audit)
# ============================================================

def validate_sql(state: AndrewState) -> Dict[str, Any]:
    query = (state.get("sql_query") or "").strip()
    if not query:
        return _fail(state, "Empty SQL query", "validate_sql")

    upper = query.upper()
    for kw in BLOCKED_SQL_KEYWORDS:
        if kw in upper:
            return _fail(state, f"Security block: '{kw}'", "validate_sql")

    schema = _normalize_schema(state.get("schema_context"))
    if not schema:
        return _fail(state, "Schema context missing", "validate_sql")

    try:
        parsed = sqlglot.parse_one(query, read="duckdb")
    except Exception as e:
        return _fail(state, f"SQL parse error: {e}", "validate_sql")

    stmt = parsed.key.lower() if getattr(parsed, "key", None) else ""
    if stmt not in ALLOWED_SQL_STATEMENTS:
        return _fail(state, f"Only SELECT/WITH allowed, got '{stmt}'", "validate_sql")

    try:
        qualified = qualify(
            parsed, schema=schema, dialect="duckdb",
            validate_qualify_columns=True, quote_identifiers=False, identify=False,
        )
    except Exception as e:
        return _fail(state, f"Schema qualification error: {e}", "validate_sql")

    used_tables = set()
    used_columns = []

    for table in qualified.find_all(exp.Table):
        t = (table.name or "").lower()
        used_tables.add(t)
        if t not in schema:
            return _fail(state, f"Hallucinated table '{t}'", "validate_sql")

    for col in qualified.find_all(exp.Column):
        c = (col.name or "").lower()
        t = (col.table or "").lower()
        if not t:
            return _fail(state, f"Ambiguous column '{c}'", "validate_sql")
        if t not in schema:
            return _fail(state, f"Unknown table '{t}' for column '{c}'", "validate_sql")
        if c not in schema[t]:
            return _fail(state, f"Column '{c}' not in table '{t}'", "validate_sql")
        used_columns.append(f"{t}.{c}")

    contract = state.get("intent_contract", {})
    allowed = set(contract.get("allowed_tables", []))
    if allowed and not used_tables.issubset(allowed):
        return _fail(state, f"Intent violation: unauthorized tables {sorted(used_tables - allowed)}", "validate_sql")

    _audit(state, "validate_sql", {
        "status": "ok", "tables": sorted(used_tables),
        "columns": sorted(set(used_columns)),
        "qualified_sql": qualified.sql(dialect="duckdb"),
    })
    return {"sql_validated": True, "sql_query": qualified.sql(dialect="duckdb"), "error_message": ""}


# ============================================================
# 10. NODE: Execute SQL
# ============================================================

def execute_sql_load_df(state: AndrewState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}

    sql = state.get("sql_query", "")
    db_url = state.get("db_url", os.getenv("DATABASE_URL", ""))

    if not db_url:
        _warn(state, "No database — SQL execution skipped")
        return {"query_results": []}

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [dict(zip(columns, r)) for r in result.fetchall()]

        csv_path = os.path.join(tempfile.gettempdir(), "andrew_sql_result.csv")
        pd.DataFrame(rows[:500]).to_csv(csv_path, index=False)

        _audit(state, "execute_sql", {"status": "ok", "rows": len(rows)})
        return {"query_results": rows[:500], "sql_result_path": csv_path, "error_message": ""}
    except Exception as e:
        return _fail(state, f"SQL execution error: {e}", "execute_sql")


# ============================================================
# 11. NODE: Generate Python (uses routed model)
# ============================================================

def generate_python(state: AndrewState) -> Dict[str, Any]:
    if state.get("error_message") or not _budget_ok(state):
        return {}

    query_results = state.get("query_results", [])
    sample = json.dumps(query_results[:5], default=str, indent=2) if query_results else "No data"

    # Use the model selected by the router
    model = state.get("python_model", MODEL_REGISTRY["default_python"])
    params = build_model_params(model)

    response = completion(
        model=model,
        messages=[{"role": "user", "content": (
            f"Goal: {state.get('user_request', state.get('goal', ''))}\n"
            f"SQL: {state.get('sql_query', 'N/A')}\n"
            f"Data ({len(query_results)} rows):\n{sample}\n\n"
            f"Write pandas analysis code. Data available as df = pd.DataFrame(data).\n"
            f"Print result as JSON: print(json.dumps(result, default=str))\n"
            f"Charts: save to '/tmp/chart.png'.\n"
            f"ONLY code. No markdown."
        )}],
        **params,
    )

    code = response.choices[0].message.content.strip()
    if code.startswith("```"):
        code = code.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    new_cost = _track_cost(response, state)
    _audit(state, "generate_python", {"status": "ok", "model": model, "cost": new_cost})
    return {"python_code": code, "cost_usd": new_cost, "error_message": ""}


# ============================================================
# 12. NODE: Validate Python (AST safety + leakage detection)
# ============================================================

class _PySafetyVisitor(ast.NodeVisitor):
    def __init__(self):
        self.imports: set = set()
        self.calls: list = []
        self.attr_calls: list = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.add(node.module.split(".")[0])
        self.generic_visit(node)

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            self.calls.append((node.func.id, node.lineno))
        elif isinstance(node.func, ast.Attribute):
            self.attr_calls.append((node.func.attr, node.lineno))
        self.generic_visit(node)


def validate_python_static(state: AndrewState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}

    code = state.get("python_code", "").strip()
    if not code:
        return _fail(state, "Empty Python code", "validate_python_static")

    confidence = state.get("confidence", 0.5)

    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return _fail(state, f"Python syntax error: {e}", "validate_python_static")

    visitor = _PySafetyVisitor()
    visitor.visit(tree)

    bad_imports = sorted(visitor.imports & SUSPICIOUS_PY_MODULES)
    if bad_imports:
        return _fail(state, f"Blocked imports: {bad_imports}", "validate_python_static")

    bad_calls = sorted({n for n, _ in visitor.calls if n in SUSPICIOUS_PY_CALLS})
    if bad_calls:
        return _fail(state, f"Blocked calls: {bad_calls}", "validate_python_static")

    # Data leakage: fit before split
    call_seq = sorted(visitor.attr_calls, key=lambda x: x[1])
    split_lines = [ln for name, ln in call_seq if name == "train_test_split"]
    fit_lines = [ln for name, ln in call_seq if name in {"fit", "fit_transform"}]

    first_split = min(split_lines) if split_lines else None
    first_fit = min(fit_lines) if fit_lines else None

    if first_split and first_fit and first_fit < first_split:
        _warn(state, "Data leakage: fit/fit_transform before train_test_split")
        confidence -= 0.35

    for pattern in OVERLAP_SPLIT_PATTERNS:
        if pattern.search(code):
            _warn(state, f"Overlap leakage: {pattern.pattern}")
            confidence -= 0.20

    # Intent: block model training unless requested
    contract = state.get("intent_contract", {})
    if not contract.get("must_not_train_model_unless_requested", False):
        training_markers = {"fit", "fit_transform", "predict", "score"}
        if any(n in training_markers for n, _ in visitor.attr_calls):
            return _fail(state, "Intent violation: model training not requested", "validate_python_static")

    _audit(state, "validate_python_static", {"status": "ok", "imports": sorted(visitor.imports)})
    return {"python_validated": True, "confidence": clamp(confidence), "error_message": ""}


# ============================================================
# 13. NODE: Sandbox Execute (E2B + local fallback)
# ============================================================

def sandbox_execute(state: AndrewState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}

    code = state.get("python_code", "").strip()
    if not code:
        return _fail(state, "No code to execute", "sandbox_execute")

    query_results = state.get("query_results", [])
    inject = f"import json, pandas as pd\ndata = {json.dumps(query_results, default=str)}\ndf = pd.DataFrame(data)\n\n"
    full_code = inject + code

    if os.getenv("E2B_API_KEY"):
        try:
            from e2b_code_interpreter import Sandbox
            with Sandbox() as sbx:
                execution = sbx.run_code(full_code)
                if execution.error:
                    return _fail(state, f"Sandbox: {execution.error.name}: {execution.error.value}", "sandbox_execute")
                output = ""
                if execution.results:
                    output += str([r.text for r in execution.results if r.text])
                if execution.logs.stdout:
                    output += "\n" + "\n".join(execution.logs.stdout)
                _audit(state, "sandbox_execute", {"status": "ok", "mode": "e2b"})
                return {"sandbox_output": output.strip(), "error_message": ""}
        except Exception as e:
            return _fail(state, f"E2B failure: {e}", "sandbox_execute")
    else:
        import subprocess, sys
        logger.warning("E2B not configured — local subprocess fallback")
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(full_code)
                tmp = f.name
            result = subprocess.run(
                [sys.executable, tmp], capture_output=True, text=True,
                timeout=30, cwd=tempfile.gettempdir(),
            )
            if result.returncode != 0:
                return _fail(state, f"Code error: {result.stderr[:500]}", "sandbox_execute")
            _audit(state, "sandbox_execute", {"status": "ok", "mode": "subprocess"})
            return {"sandbox_output": result.stdout[:10000].strip(), "error_message": ""}
        except subprocess.TimeoutExpired:
            return _fail(state, "Timeout (30s)", "sandbox_execute")
        except Exception as e:
            return _fail(state, f"Subprocess: {e}", "sandbox_execute")


# ============================================================
# 14. NODE: Validate Results (Pandera + numerical)
# ============================================================

def validate_results(state: AndrewState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}

    confidence = state.get("confidence", 0.5)

    csv_path = state.get("sql_result_path")
    if csv_path:
        try:
            df = pd.read_csv(csv_path)
            schema = pa.DataFrameSchema(
                {"total_revenue": pa.Column(float, checks=[Check.ge(0)], required=False, nullable=False)},
                strict=False, coerce=True,
            )
            schema.validate(df)
            confidence += 0.15
            _audit(state, "validate_results", {"status": "ok", "pandera": "passed"})
        except FileNotFoundError:
            _warn(state, "SQL result CSV not found")
        except Exception as e:
            _warn(state, f"Pandera: {e}")
            confidence -= 0.15
    else:
        _warn(state, "No CSV — Pandera skipped")

    if not state.get("sandbox_output") and not state.get("query_results"):
        confidence -= 0.3
        _warn(state, "No output data")

    return {"confidence": clamp(confidence), "error_message": ""}


# ============================================================
# 15. NODE: Semantic Guardrails
# ============================================================

def semantic_guardrails(state: AndrewState) -> Dict[str, Any]:
    if state.get("error_message"):
        return {}

    request = _normalize(state.get("user_request", state.get("goal", "")))
    query = (state.get("sql_query") or "").lower()
    confidence = state.get("confidence", 0.5)
    issues = []

    if "revenue" in request and "revenue" not in query:
        issues.append("Asked about revenue but SQL doesn't reference it")
    if "top" in request and "order by" not in query:
        issues.append("Asked for top-N but no ORDER BY")
    if any(w in request for w in ["monthly", "by month", "per month"]) and "group by" not in query:
        issues.append("Asked for monthly aggregation but no GROUP BY")

    if issues:
        confidence -= 0.20
        for i in issues:
            _warn(state, f"Semantic: {i}")

    _audit(state, "semantic_guardrails", {"status": "ok", "issues": issues})
    return {"confidence": clamp(confidence)}


# ============================================================
# 16. NODE: Finalize
# ============================================================

def hitl_escalate(state: AndrewState) -> Dict[str, Any]:
    """
    Human-in-the-Loop escalation node.

    Fires when confidence drops below HITL_CONFIDENCE_THRESHOLD after all
    validation stages.  Marks the result as requiring human review and
    summarises the reasons so downstream systems (bridge, Moltis channel,
    UI) can surface a clear message to an operator.
    """
    confidence = state.get("confidence", 0.0)
    warnings = state.get("warnings", [])

    reasons: List[str] = []
    if confidence < HITL_CONFIDENCE_THRESHOLD:
        reasons.append(f"confidence {confidence:.2f} < threshold {HITL_CONFIDENCE_THRESHOLD:.2f}")
    if warnings:
        reasons.extend(warnings[:5])  # cap to keep the message readable

    reason_str = "; ".join(reasons) if reasons else "low confidence"
    _audit(state, "hitl_escalate", {"confidence": confidence, "reasons": reasons})
    logger.warning(f"HITL escalation triggered: {reason_str}")
    return {
        "hitl_required": True,
        "hitl_reason": reason_str,
    }


def finalize_state(state: AndrewState) -> Dict[str, Any]:
    blob = json.dumps({
        "user_request": state.get("user_request", state.get("goal")),
        "sql_query": state.get("sql_query"),
        "python_code": state.get("python_code"),
        "routing_decision": state.get("routing_decision"),
    }, sort_keys=True, default=str).encode()
    h = hashlib.sha256(blob).hexdigest()
    _audit(state, "finalize", {"status": "ok", "hash": h})
    return {"state_hash": h, "confidence": clamp(state.get("confidence", 0.5))}


# ============================================================
# 17. ROUTING EDGES
# ============================================================

def route_on_error(state: AndrewState) -> str:
    return END if state.get("error_message") else "continue"


# ============================================================
# 18. GRAPH WIRING
# ============================================================

workflow = StateGraph(AndrewState)

workflow.add_node("route_query_intent", route_query_intent)
workflow.add_node("build_intent_contract", build_intent_contract)
workflow.add_node("generate_sql", generate_sql, retry_policy=llm_retry)
workflow.add_node("validate_sql", validate_sql)
workflow.add_node("execute_sql_load_df", execute_sql_load_df)
workflow.add_node("generate_python", generate_python, retry_policy=llm_retry)
workflow.add_node("validate_python_static", validate_python_static)
workflow.add_node("sandbox_execute", sandbox_execute, retry_policy=sandbox_retry)
workflow.add_node("validate_results", validate_results)
workflow.add_node("semantic_guardrails", semantic_guardrails)
workflow.add_node("hitl_escalate", hitl_escalate)
workflow.add_node("finalize_state", finalize_state)

workflow.add_edge(START, "route_query_intent")
workflow.add_edge("route_query_intent", "build_intent_contract")
workflow.add_edge("build_intent_contract", "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")

workflow.add_conditional_edges("validate_sql", route_on_error, {
    END: END, "continue": "execute_sql_load_df",
})

workflow.add_edge("execute_sql_load_df", "generate_python")
workflow.add_edge("generate_python", "validate_python_static")

workflow.add_conditional_edges("validate_python_static", route_on_error, {
    END: END, "continue": "sandbox_execute",
})

workflow.add_conditional_edges("sandbox_execute", route_on_error, {
    END: END, "continue": "validate_results",
})

workflow.add_edge("validate_results", "semantic_guardrails")
workflow.add_conditional_edges(
    "semantic_guardrails",
    lambda s: "hitl" if s.get("confidence", 1.0) < HITL_CONFIDENCE_THRESHOLD else "ok",
    {"hitl": "hitl_escalate", "ok": "finalize_state"},
)
workflow.add_edge("hitl_escalate", "finalize_state")
workflow.add_edge("finalize_state", END)

langgraph_executor = workflow.compile()


# ============================================================
# 19. ROMA BRIDGE
# ============================================================

class AndrewExecutor:
    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or os.getenv("DATABASE_URL", "")
        self._schema: Optional[Dict] = None
        # Initialise Capability Registry embeddings (Sprint 5 §3).
        # No-ops silently when the embedding API is unavailable; keyword
        # fallback in SemanticRouter stays active in that case.
        try:
            from core.semantic_router import init_registry
            init_registry()
        except Exception as _exc:
            logger.debug(f"Semantic registry init skipped: {_exc}")

    @property
    def schema(self) -> Dict[str, Dict[str, str]]:
        if self._schema is None:
            self._schema = discover_schema(self.db_url)
            logger.info(f"Schema: {list(self._schema.keys())}")
        return self._schema

    def execute(self, goal: str) -> AndrewResult:
        logger.info(f"Andrew Swarm v1.0.0-rc1 | {goal[:80]}")
        state = langgraph_executor.invoke({
            "user_request": goal, "goal": goal,
            "schema_context": self.schema, "db_url": self.db_url,
            "sql_query": "", "sql_validated": False,
            "python_code": "", "python_validated": False,
            "sandbox_output": "", "error_message": "",
            "cost_usd": 0.0, "confidence": 0.5,
            "warnings": [], "audit_log": [],
        })
        return AndrewResult(state)

    def invalidate_schema(self):
        self._schema = None


# ============================================================
# 20. CLI ENTRY POINT & SMOKE TESTS
# ============================================================

if __name__ == "__main__":
    import sys

    # Live query mode: python3 core/andrew_swarm.py "<question>"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        db_url = os.getenv("DATABASE_URL", "")
        print("=" * 70)
        print(f"ANDREW SWARM v1.0.0-rc1 — LIVE QUERY")
        print(f"DB: {db_url or '(none)'}")
        print("=" * 70)
        executor = AndrewExecutor(db_url=db_url or None)
        result = executor.execute(query)
        print("\n" + result.to_roma_output())
        print("\n" + str(result))
        sys.exit(0 if result.success else 1)

    # Routing smoke tests (no args)
    print("=" * 70)
    print("ANDREW SWARM v0.4 — ROUTING SMOKE TEST")
    print("=" * 70)

    tests = [
        "Forecast next quarter revenue using ARIMA and confidence intervals",
        "Show total revenue by region as a bar chart",
        "Run Monte Carlo simulation for sales growth",
        "Calculate CAGR and break-even point for product lines",
        "What is the average revenue by month?",
        "Build a neural network to predict customer churn",
        "Group sales by region and show percentages",
        "Perform linear regression on price vs quantity",
        "Create a simple pie chart of market share",
        "Optimize inventory levels with linear programming",
    ]

    for i, query in enumerate(tests, 1):
        state = {"goal": query, "user_request": query}
        r = route_query_intent(state)
        print(f"\n[{i:2d}] {query}")
        print(f"     route={r['routing_decision']:20s} score={r['routing_score']:2d}  model={r['python_model']}")
        print(f"     hits={r['routing_hits']}")

    print("\n" + "=" * 70)
    print("All 10 routing tests completed.")
    print("=" * 70)
