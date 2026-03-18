"""
core/routing.py
===============
Weighted keyword query router — classifies natural language queries into
three model lanes based on mathematical/analytical complexity.

Lanes:
  reasoning_math      score >= 4 or heavy ML term  →  Grok-4 / reasoning model
  analytics_fastlane  light math + BI context       →  GPT-4o-mini
  standard            pure BI / no math             →  Claude Sonnet

All model assignments are configurable via environment variables; no
vendor lock-in at the code level.
"""

import logging
import os
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("routing")


# ── Model registry ────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, str] = {
    "reasoning_math":      os.getenv("MODEL_REASONING_MATH", "grok-4"),
    "default_orchestrator": os.getenv("MODEL_ORCHESTRATOR", "anthropic/claude-sonnet-4-20250514"),
    "default_python":      os.getenv("MODEL_PYTHON", "anthropic/claude-sonnet-4-20250514"),
    "analytics_fastlane":  os.getenv("MODEL_ANALYTICS", "gpt-4o-mini"),
    "sql_generation":      os.getenv("MODEL_SQL", "gpt-4o-mini"),
}


def build_model_params(model_name: str) -> dict:
    """
    Provider-specific parameter adapter.
    Different APIs accept different parameter sets; this returns a safe
    subset for the given model.
    """
    name = (model_name or "").lower()
    if "grok" in name:
        return {"temperature": 0.2}
    if "claude" in name:
        return {"temperature": 0.0}
    if "gpt" in name or "openai" in name:
        return {"temperature": 0.0}
    return {"temperature": 0.2}


# ── Keyword tables ────────────────────────────────────────────

# Weights: 1=basic analytics  2=intermediate stats  3=advanced  4=heavy ML/math
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


# ── Router ────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _match_keywords(request: str) -> List[Tuple[str, int]]:
    hits = []
    padded = f" {request} "
    for kw, weight in MATH_KEYWORDS.items():
        if f" {kw} " in padded or kw in request:
            hits.append((kw, weight))
    return hits


def route_query_intent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Score-based routing into three model lanes.

    Reads:   state["user_request"] or state["goal"]
    Returns: routing_decision, routing_score, routing_hits,
             python_model, orchestrator_model
    """
    request = _normalize(state.get("user_request", state.get("goal", "")))
    hits = _match_keywords(request)
    score = sum(w for _, w in hits)

    has_light = any(t in request for t in LIGHT_ANALYTICS_TERMS)
    has_heavy = any(t in request for t in HEAVY_ML_TERMS)

    if has_heavy or score >= 4:
        model = MODEL_REGISTRY["reasoning_math"]
        route = "reasoning_math"
    elif hits and has_light:
        model = MODEL_REGISTRY["analytics_fastlane"]
        route = "analytics_fastlane"
    else:
        model = MODEL_REGISTRY["default_python"]
        route = "standard"

    matched_terms = [kw for kw, _ in hits]
    logger.info(f"Route: {route} (score={score}, hits={matched_terms}) → {model}")

    return {
        "python_model": model,
        "orchestrator_model": MODEL_REGISTRY["default_orchestrator"],
        "routing_decision": route,
        "routing_score": score,
        "routing_hits": matched_terms,
    }
