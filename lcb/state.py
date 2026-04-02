"""
lcb/state.py
============
LangGraph TypedDict state for the Andrew LCB pipeline.

Flow:
  classify → extract_constraints → retrieve_templates → generate_self_tests
    → generate_candidates → evaluate_candidates → repair_loop → select_best
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TypedDict

from .runner import LCBProblem, ProblemTestCase


class LCBState(TypedDict, total=False):
    # ── Input ─────────────────────────────────────────────────
    problem: LCBProblem

    # ── Difficulty routing ────────────────────────────────────
    difficulty: str          # 'easy' | 'hard'
    model: str               # LiteLLM model string
    classifier_score: int
    classifier_hits: List[str]

    # ── Constraint extraction ─────────────────────────────────
    constraints: Dict[str, Any]
    # Keys: n_max (int), m_max (int), value_range (int),
    #       structure (str: array/graph/string/tree/grid),
    #       is_weighted (bool), is_directed (bool)

    algorithm_hints: List[str]
    # e.g. ['dp', 'bfs', 'binary_search'] — inferred from constraints

    time_complexity_target: str
    # e.g. 'O(n log n)' — from n_max

    # ── RAG templates ─────────────────────────────────────────
    template_names: List[str]
    template_context: str    # formatted prompt block

    # ── Self-generated test cases ─────────────────────────────
    self_tests: List[ProblemTestCase]
    self_tests_raw: str      # raw LLM JSON for debug

    # ── Candidate generation ──────────────────────────────────
    candidates: List[str]                       # code strings
    candidate_strategies: List[str]             # which strategy used

    # ── Candidate evaluation (after sandbox runs) ─────────────
    candidate_results: List[Tuple[int, int]]    # (passed, total)
    candidate_errors: List[Optional[str]]       # first failure msg

    # ── Repair ───────────────────────────────────────────────
    repair_history: List[Dict[str, Any]]
    # Each entry: {candidate_idx, iteration, code_before, code_after, error}

    # ── Best solution ─────────────────────────────────────────
    best_code: str
    best_passed: int
    best_total: int
    best_strategy: str

    # ── Budget & observability ────────────────────────────────
    cost_usd: float
    audit_log: List[Dict[str, Any]]
    error: Optional[str]
