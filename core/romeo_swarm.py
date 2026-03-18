"""
Romeo PhD Agent v1.0.0
===================================================
Educational companion to Andrew Swarm. Answers conceptual questions,
explains terminology, provides examples, and teaches analytical topics.

Sprint 8: Multi-Agent Supervisor integration.

Pipeline (4 nodes, no SQL, no data execution):
  classify_topic → generate_explanation → validate_educational → finalize_romeo
"""

import hashlib
import json
import logging
import os
from typing import Any, Dict, List, Optional, TypedDict

from core.andrew_swarm import (
    MODEL_REGISTRY,
    llm_retry,
    build_model_params,
    _track_cost,
    _warn,
    _audit,
    _fail,
    clamp,
    MAX_COST_USD,
    _budget_ok,
)

logger = logging.getLogger("romeo")

# ============================================================
# 1. STATE
# ============================================================

class RomeoState(TypedDict, total=False):
    # Input
    user_request: str

    # Routing
    education_model: str
    topic_type: str        # "concept" | "comparison" | "derivation" | "tutorial"

    # Output
    explanation: str
    examples: List[str]
    key_takeaways: List[str]

    # Quality
    confidence: float
    warnings: List[str]
    audit_log: List[Dict[str, Any]]

    # Control
    error_message: str
    cost_usd: float
    state_hash: str


# ============================================================
# 2. MODEL REGISTRY
# ============================================================

ROMEO_MODEL = os.getenv("MODEL_EDUCATOR", "anthropic/claude-sonnet-4-20250514")


# ============================================================
# 3. TOPIC CLASSIFIER KEYWORDS (no LLM)
# ============================================================

COMPARISON_SIGNALS = {
    "difference between", "vs", "versus", "compare", "comparison",
    "contrast", "distinguish", "which is better", "pros and cons",
    "similarities", "differences",
}

DERIVATION_SIGNALS = {
    "derive", "derivation", "proof", "prove", "show that", "step by step",
    "step-by-step", "work out", "calculate from scratch", "formula for",
    "where does", "why is the formula",
}

TUTORIAL_SIGNALS = {
    "tutorial", "how to", "walk me through", "guide me", "teach me how",
    "beginner guide", "introduction", "getting started", "learn how",
    "hands-on", "practical example",
}


# ============================================================
# 4. RESULT CLASS
# ============================================================

class RomeoResult:
    """Structured output — same public surface as AndrewResult for bridge compatibility."""

    def __init__(self, state: RomeoState):
        self.goal = state.get("user_request", "")
        self.sql_query = None  # Romeo produces no SQL
        self.python_code = None
        self.output = self._build_output(state)
        self.error = state.get("error_message")
        self.cost = state.get("cost_usd", 0.0)
        self.confidence = state.get("confidence", 0.0)
        self.warnings = state.get("warnings", [])
        self.audit_log = state.get("audit_log", [])
        self.state_hash = state.get("state_hash", "")
        self.routing = "romeo"
        self.model_used = state.get("education_model", ROMEO_MODEL)
        self.topic_type = state.get("topic_type", "concept")
        self.examples = state.get("examples", [])
        self.key_takeaways = state.get("key_takeaways", [])

    def _build_output(self, state: RomeoState) -> str:
        parts = []
        explanation = state.get("explanation", "")
        if explanation:
            parts.append(explanation)
        examples = state.get("examples", [])
        if examples:
            parts.append("\n**Examples:**")
            parts.extend(f"- {e}" for e in examples)
        takeaways = state.get("key_takeaways", [])
        if takeaways:
            parts.append("\n**Key Takeaways:**")
            parts.extend(f"- {t}" for t in takeaways)
        return "\n".join(parts)

    @property
    def success(self) -> bool:
        return not self.error and bool(self.output)

    def to_roma_output(self) -> str:
        parts = [f"## Explanation: {self.goal}"]
        if self.output:
            parts.append(self.output[:3000])
        if self.warnings:
            parts.append("\n**Warnings:**\n" + "\n".join(f"- {w}" for w in self.warnings))
        if self.error:
            parts.append(f"\n**Error:** {self.error}")
        parts.append(
            f"\n<metadata confidence=\"{self.confidence:.2f}\" cost=\"${self.cost:.4f}\" "
            f"route=\"romeo/{self.topic_type}\" hash=\"{self.state_hash[:12]}\" />"
        )
        return "\n".join(parts)

    def __str__(self):
        s = "OK" if self.success else "FAIL"
        return (
            f"[{s}] Romeo | {self.goal}\n"
            f"  Topic type: {self.topic_type}\n"
            f"  Output: {(self.output or '-')[:120]}\n"
            f"  Confidence: {self.confidence:.2f} | Cost: ${self.cost:.4f} | Warnings: {len(self.warnings)}"
        )


# ============================================================
# 5. PIPELINE NODES
# ============================================================

def classify_topic(state: RomeoState) -> Dict[str, Any]:
    """Keyword-based topic classification — no LLM call."""
    request = (state.get("user_request", "") or "").lower()

    if any(sig in request for sig in COMPARISON_SIGNALS):
        topic = "comparison"
    elif any(sig in request for sig in DERIVATION_SIGNALS):
        topic = "derivation"
    elif any(sig in request for sig in TUTORIAL_SIGNALS):
        topic = "tutorial"
    else:
        topic = "concept"

    model = ROMEO_MODEL
    logger.info(f"Romeo topic: {topic} | model: {model}")
    _audit(state, "classify_topic", {"status": "ok", "topic_type": topic, "model": model})

    return {
        "topic_type": topic,
        "education_model": model,
        "confidence": 0.5,
        "warnings": state.get("warnings", []),
        "audit_log": state.get("audit_log", []),
    }


def generate_explanation(state: RomeoState) -> Dict[str, Any]:
    """LLM call to produce structured educational content."""
    if not _budget_ok(state):
        return _fail(state, f"Budget exceeded (${state.get('cost_usd', 0):.4f} >= ${MAX_COST_USD})", "generate_explanation")

    request = state.get("user_request", "")
    topic_type = state.get("topic_type", "concept")
    model = state.get("education_model", ROMEO_MODEL)

    type_instructions = {
        "concept": "Explain the concept clearly for a non-expert. Include intuition, definition, and real-world relevance.",
        "comparison": "Compare and contrast the items. Use a balanced structure: similarities first, then key differences, then when to use each.",
        "derivation": "Walk through the derivation step by step. Show each mathematical/logical step with justification.",
        "tutorial": "Provide a practical, hands-on walkthrough. Use concrete steps, code or formulas where helpful, and common pitfalls.",
    }

    system_prompt = (
        "You are Romeo, a PhD-level educator and data science expert. "
        "Your explanations are clear, precise, and accessible to non-experts. "
        "You always give concrete examples and actionable takeaways.\n\n"
        f"TASK: {type_instructions.get(topic_type, type_instructions['concept'])}\n\n"
        "Return ONLY valid JSON with this exact structure:\n"
        '{"explanation": "...", "examples": ["...", "..."], "key_takeaways": ["...", "..."]}\n'
        "No markdown fences, no preamble — just the JSON object."
    )

    params = build_model_params(model)
    try:
        from litellm import completion
        response = completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request},
            ],
            **params,
        )
    except Exception as e:
        return _fail(state, f"LLM call failed: {e}", "generate_explanation")

    new_cost = _track_cost(response, state)
    raw = (response.choices[0].message.content or "").strip()

    # Parse JSON from LLM response
    try:
        # Strip optional markdown fences if model ignores instruction
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```[a-z]*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```$", "", cleaned)
        data = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"JSON parse failed, using raw text: {e}")
        data = {
            "explanation": raw[:2000],
            "examples": [],
            "key_takeaways": [],
        }

    explanation = str(data.get("explanation", "")).strip()
    examples = [str(x) for x in data.get("examples", []) if x]
    key_takeaways = [str(t) for t in data.get("key_takeaways", []) if t]

    _audit(state, "generate_explanation", {
        "status": "ok",
        "model": model,
        "cost": new_cost - state.get("cost_usd", 0.0),
        "has_examples": bool(examples),
        "has_takeaways": bool(key_takeaways),
    })

    return {
        "explanation": explanation,
        "examples": examples,
        "key_takeaways": key_takeaways,
        "cost_usd": new_cost,
        "warnings": state.get("warnings", []),
        "audit_log": state.get("audit_log", []),
    }


def validate_educational(state: RomeoState) -> Dict[str, Any]:
    """Non-LLM quality checks on educational output."""
    explanation = state.get("explanation", "")
    key_takeaways = state.get("key_takeaways", [])
    confidence = state.get("confidence", 0.5)

    issues = []
    if not explanation or len(explanation) < 50:
        issues.append("Explanation too short or missing")
    if not key_takeaways:
        _warn(state, "No key takeaways generated")
        confidence = clamp(confidence - 0.1)

    if issues:
        return _fail(state, "; ".join(issues), "validate_educational")

    # Boost confidence if output is rich
    if len(explanation) > 300:
        confidence = clamp(confidence + 0.2)
    if key_takeaways:
        confidence = clamp(confidence + 0.1)
    if state.get("examples", []):
        confidence = clamp(confidence + 0.1)

    _audit(state, "validate_educational", {
        "status": "ok",
        "explanation_len": len(explanation),
        "examples_count": len(state.get("examples", [])),
        "takeaways_count": len(key_takeaways),
        "confidence": confidence,
    })

    return {
        "confidence": confidence,
        "warnings": state.get("warnings", []),
        "audit_log": state.get("audit_log", []),
    }


def finalize_romeo(state: RomeoState) -> Dict[str, Any]:
    """Compute state hash and seal audit log."""
    payload = json.dumps({
        "user_request": state.get("user_request", ""),
        "explanation": state.get("explanation", ""),
        "topic_type": state.get("topic_type", ""),
    }, sort_keys=True)
    state_hash = hashlib.sha256(payload.encode()).hexdigest()

    _audit(state, "finalize_romeo", {
        "status": "ok",
        "state_hash": state_hash[:12],
        "total_cost": state.get("cost_usd", 0.0),
        "confidence": state.get("confidence", 0.5),
    })

    return {
        "state_hash": state_hash,
        "audit_log": state.get("audit_log", []),
    }


# ============================================================
# 6. ERROR ROUTING
# ============================================================

import re  # needed for markdown fence stripping in generate_explanation


def _route_on_error(state: RomeoState) -> str:
    from langgraph.graph import END
    return END if state.get("error_message") else "continue"


# ============================================================
# 7. GRAPH WIRING (lazy compile)
# ============================================================

_romeo_graph = None


def _get_romeo_graph():
    """Compile the LangGraph pipeline on first call."""
    global _romeo_graph
    if _romeo_graph is not None:
        return _romeo_graph

    from langgraph.graph import START, END, StateGraph

    def _route(state):
        return END if state.get("error_message") else "continue"

    workflow = StateGraph(RomeoState)

    workflow.add_node("classify_topic", classify_topic)
    workflow.add_node("generate_explanation", generate_explanation, retry=llm_retry)
    workflow.add_node("validate_educational", validate_educational)
    workflow.add_node("finalize_romeo", finalize_romeo)

    workflow.add_edge(START, "classify_topic")
    workflow.add_edge("classify_topic", "generate_explanation")
    workflow.add_conditional_edges("generate_explanation", _route, {
        END: END,
        "continue": "validate_educational",
    })
    workflow.add_conditional_edges("validate_educational", _route, {
        END: END,
        "continue": "finalize_romeo",
    })
    workflow.add_edge("finalize_romeo", END)

    _romeo_graph = workflow.compile()
    return _romeo_graph


# ============================================================
# 8. EXECUTOR
# ============================================================

class RomeoExecutor:
    """Thin wrapper around romeo_graph — same interface as AndrewExecutor."""

    def execute(self, goal: str) -> RomeoResult:
        logger.info(f"Romeo PhD v1.0.0 | {goal[:80]}")
        state = _get_romeo_graph().invoke({
            "user_request": goal,
            "cost_usd": 0.0,
            "confidence": 0.5,
            "warnings": [],
            "audit_log": [],
            "error_message": "",
        })
        return RomeoResult(state)


# ============================================================
# 9. CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print("=" * 70)
        print("ROMEO PHD v1.0.0 — LIVE QUERY")
        print("=" * 70)
        executor = RomeoExecutor()
        result = executor.execute(query)
        print("\n" + result.to_roma_output())
    else:
        print("Usage: python3 core/romeo_swarm.py '<educational question>'")
