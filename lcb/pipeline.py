"""
lcb/pipeline.py
===============
LangGraph pipeline for the Andrew LCB competitive programming solver.

Graph nodes:
  classify_difficulty  → infer easy/hard, select model
  extract_constraints  → regex parse N, M, structure
  retrieve_templates   → RAG lookup of algo templates
  generate_self_tests  → LLM edge-case generation
  generate_candidates  → multi-strategy code generation
  evaluate_candidates  → subprocess sandbox
  repair_candidates    → per-candidate repair loop
  select_best          → majority vote + heuristic

Routing mirrors core/andrew_swarm.py patterns:
  - litellm.completion() for all LLM calls
  - completion_cost() for budget tracking
  - RetryPolicy style retry via simple loop
  - _budget_ok() threshold check
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from langgraph.graph import END, START, StateGraph
    _LANGGRAPH_AVAILABLE = True
except ImportError:
    _LANGGRAPH_AVAILABLE = False

try:
    from litellm import completion, completion_cost
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

from .classifier import classify_from_metadata
from .algo_templates import retrieve_templates, format_templates_for_prompt
from .constraints import extract_constraints, infer_algorithm_hints
from .prompts import (
    SYSTEM_PROMPT, SELF_TEST_SYSTEM,
    build_direct_prompt, build_plan_then_code_prompt,
    build_analogy_prompt, build_repair_prompt, build_self_test_prompt,
    CANDIDATE_PLAN,
)
from .runner import (
    LCBProblem, ProblemTestCase, evaluate_solution,
    LCB_MODEL_EASY, LCB_MODEL_HARD,
    EASY_TEMPERATURE, HARD_TEMPERATURE, MAX_REPAIR_ITERATIONS,
)

# Parallel candidate generation (SLIME-style async — each candidate independent)
_PARALLEL_CANDIDATES = os.getenv("LCB_PARALLEL_CANDIDATES", "true").lower() == "true"
from .state import LCBState

logger = logging.getLogger("lcb.pipeline")

# ── Budget ────────────────────────────────────────────────────

LCB_MAX_COST = float(os.getenv("LCB_MAX_COST", "5.00"))  # per problem

def _budget_ok(state: LCBState) -> bool:
    return state.get("cost_usd", 0.0) < LCB_MAX_COST


def _track_cost(response: Any, state: LCBState) -> float:
    try:
        c = completion_cost(completion_response=response)
    except Exception:
        c = 0.0
    return state.get("cost_usd", 0.0) + c


# ── LLM call helper ───────────────────────────────────────────

def _llm(
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int = 2048,
    max_attempts: int = 3,
) -> Tuple[str, Any]:
    """
    Single LLM call with simple retry.
    Returns (content, raw_response).
    Raises on all attempts exhausted.
    """
    if not _LITELLM_AVAILABLE:
        raise RuntimeError("litellm not installed")

    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            resp = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            # Strip accidental markdown fences
            if content.startswith("```"):
                lines = content.splitlines()
                content = "\n".join(l for l in lines if not l.startswith("```")).strip()
            return content, resp
        except Exception as exc:
            last_exc = exc
            logger.warning("LLM attempt %d/%d failed: %s", attempt + 1, max_attempts, exc)
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= 2.0
    raise RuntimeError(f"LLM call failed after {max_attempts} attempts") from last_exc


def _audit(state: LCBState, node: str, **kwargs) -> List[Dict]:
    entry = {"node": node, "cost_so_far": state.get("cost_usd", 0.0), **kwargs}
    return state.get("audit_log", []) + [entry]


# ── Graph nodes ───────────────────────────────────────────────

def node_classify(state: LCBState) -> LCBState:
    """Classify difficulty and select model."""
    problem: LCBProblem = state["problem"]
    difficulty, score, hits = classify_from_metadata(
        problem.statement, problem.lcb_difficulty
    )
    model = LCB_MODEL_EASY if difficulty == "easy" else LCB_MODEL_HARD
    logger.info("[classify] %s → %s (score=%d)", problem.problem_id, difficulty, score)
    return {
        **state,
        "difficulty": difficulty,
        "model": model,
        "classifier_score": score,
        "classifier_hits": hits,
        "audit_log": _audit(state, "classify", difficulty=difficulty, model=model),
    }


def node_extract_constraints(state: LCBState) -> LCBState:
    """Regex-extract N bounds, structure, and algorithm hints."""
    problem: LCBProblem = state["problem"]
    constraints = extract_constraints(problem.statement)
    hints, complexity = infer_algorithm_hints(constraints)
    logger.info(
        "[constraints] n_max=%s struct=%s hints=%s",
        constraints["n_max"], constraints["structure"], hints[:3],
    )
    return {
        **state,
        "constraints": constraints,
        "algorithm_hints": hints,
        "time_complexity_target": complexity,
        "audit_log": _audit(state, "extract_constraints",
                            constraints=constraints, hints=hints[:5]),
    }


def node_retrieve_templates(state: LCBState) -> LCBState:
    """RAG: fetch algorithm templates matching the problem."""
    problem: LCBProblem = state["problem"]
    # Combine problem statement + algorithm hints for better retrieval
    hints = state.get("algorithm_hints", [])
    query = problem.statement + " " + " ".join(hints[:5])
    templates = retrieve_templates(query, top_k=2)
    context = format_templates_for_prompt(templates)
    names = [t.name for t in templates]
    logger.info("[templates] retrieved: %s", names)
    return {
        **state,
        "template_names": names,
        "template_context": context,
        "audit_log": _audit(state, "retrieve_templates", templates=names),
    }


def node_generate_self_tests(state: LCBState) -> LCBState:
    """
    Ask the LLM to generate edge-case test inputs before solving.
    Falls back to empty list if budget exhausted or LLM fails.
    """
    if not _budget_ok(state):
        logger.warning("[self_tests] budget exceeded, skipping")
        return {**state, "self_tests": [], "self_tests_raw": "budget_exceeded"}

    problem: LCBProblem = state["problem"]
    model = state.get("model", LCB_MODEL_HARD)
    constraints = state.get("constraints", {})

    prompt = build_self_test_prompt(problem, constraints)
    try:
        raw, resp = _llm(model, SELF_TEST_SYSTEM, prompt, temperature=0.3, max_tokens=1024)
        new_cost = _track_cost(resp, state)
    except Exception as exc:
        logger.warning("[self_tests] LLM failed: %s", exc)
        return {**state, "self_tests": [], "self_tests_raw": str(exc)}

    # Parse JSON array of {"stdin": ..., "stdout": ...}
    self_tests: List[ProblemTestCase] = []
    try:
        items = json.loads(raw)
        for item in items:
            if isinstance(item, dict) and "stdin" in item and "stdout" in item:
                self_tests.append(ProblemTestCase(
                    stdin=str(item["stdin"]),
                    expected_stdout=str(item["stdout"]).strip(),
                ))
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("[self_tests] JSON parse failed: %s", exc)

    logger.info("[self_tests] generated %d test cases", len(self_tests))
    return {
        **state,
        "self_tests": self_tests,
        "self_tests_raw": raw,
        "cost_usd": new_cost,
        "audit_log": _audit(state, "generate_self_tests", n_tests=len(self_tests)),
    }


def _build_prompt_for_strategy(
    strategy: str,
    problem: LCBProblem,
    hints: List[str],
    template_context: str,
    constraints_summary: str,
) -> str:
    """Build a generation prompt for the given strategy."""
    if strategy == "direct":
        return build_direct_prompt(problem, template_context, constraints_summary)
    elif strategy == "plan_then_code":
        return build_plan_then_code_prompt(
            problem, hints, template_context, constraints_summary
        )
    else:  # analogy
        return build_analogy_prompt(problem, hints, template_context)


def node_generate_candidates(state: LCBState) -> LCBState:
    """
    Generate N candidate solutions using different prompt strategies.

    easy: 2 direct candidates (GPT-4o-mini, cheap)
    hard: 4 candidates — direct×2, plan_then_code, analogy (GLM 5.1)

    When LCB_PARALLEL_CANDIDATES=true (default), candidates are generated
    concurrently via ThreadPoolExecutor — mirrors SLIME async RL philosophy:
    independent rollouts don't block each other.
    """
    problem: LCBProblem = state["problem"]
    difficulty = state.get("difficulty", "hard")
    model = state.get("model", LCB_MODEL_HARD)
    temperature = EASY_TEMPERATURE if difficulty == "easy" else HARD_TEMPERATURE
    template_context = state.get("template_context", "")
    hints = state.get("algorithm_hints", [])
    complexity = state.get("time_complexity_target", "")
    n_max = state.get("constraints", {}).get("n_max", 0)
    constraints_summary = (
        f"N ≤ {n_max:,}, target {complexity}, "
        f"structure={state.get('constraints', {}).get('structure', '?')}"
        if n_max else f"target {complexity}"
    )

    if not _budget_ok(state):
        logger.warning("[candidates] budget exceeded, skipping generation")
        return {**state, "candidates": [], "candidate_strategies": []}

    strategies = CANDIDATE_PLAN[difficulty]

    def _generate_one(strategy: str) -> Tuple[str, str, float]:
        """Returns (code, strategy, cost). Empty code on failure."""
        prompt = _build_prompt_for_strategy(
            strategy, problem, hints, template_context, constraints_summary
        )
        try:
            code, resp = _llm(model, SYSTEM_PROMPT, prompt, temperature=temperature)
            cost = _track_cost(resp, {"cost_usd": 0.0})
            return code, strategy, cost
        except Exception as exc:
            logger.warning("[candidates] strategy=%s failed: %s", strategy, exc)
            return "", strategy, 0.0

    candidates: List[str] = []
    strategies_used: List[str] = []
    total_cost = state.get("cost_usd", 0.0)

    if _PARALLEL_CANDIDATES and len(strategies) > 1:
        # Parallel generation: all candidates launch simultaneously (SLIME-style)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=len(strategies)) as pool:
            futures = {pool.submit(_generate_one, s): s for s in strategies}
            for fut in as_completed(futures):
                code, strategy, cost = fut.result()
                if code:
                    candidates.append(code)
                    strategies_used.append(strategy)
                    total_cost += cost
                    logger.info("[candidates] parallel %s → %d tokens",
                                strategy, len(code.split()))
    else:
        # Sequential fallback (easier to debug)
        for strategy in strategies:
            code, _, cost = _generate_one(strategy)
            if code:
                candidates.append(code)
                strategies_used.append(strategy)
                total_cost += cost

    logger.info(
        "[candidates] %s: %d/%d candidates generated (parallel=%s)",
        problem.problem_id, len(candidates), len(strategies), _PARALLEL_CANDIDATES,
    )
    return {
        **state,
        "candidates": candidates,
        "candidate_strategies": strategies_used,
        "cost_usd": total_cost,
        "audit_log": _audit(state, "generate_candidates",
                            n_candidates=len(candidates),
                            strategies=strategies_used,
                            parallel=_PARALLEL_CANDIDATES),
    }


def node_evaluate_candidates(state: LCBState) -> LCBState:
    """
    Run every candidate against self-tests + provided test cases.
    Combine: self_tests (edge cases) + problem.test_cases (given examples).
    """
    candidates: List[str] = state.get("candidates", [])
    problem: LCBProblem = state["problem"]
    self_tests: List[ProblemTestCase] = state.get("self_tests", [])
    all_tests = self_tests + problem.test_cases

    results: List[Tuple[int, int]] = []
    errors: List[Optional[str]] = []

    for code in candidates:
        if not code.strip():
            results.append((0, len(all_tests)))
            errors.append("empty code")
            continue
        passed, total, err = evaluate_solution(code, all_tests)
        results.append((passed, total))
        errors.append(err)

    logger.info(
        "[evaluate] scores: %s",
        [f"{p}/{t}" for p, t in results],
    )
    return {
        **state,
        "candidate_results": results,
        "candidate_errors": errors,
        "audit_log": _audit(state, "evaluate_candidates",
                            scores=[f"{p}/{t}" for p, t in results]),
    }


def node_repair_candidates(state: LCBState) -> LCBState:
    """
    For each candidate that failed, attempt up to MAX_REPAIR_ITERATIONS repairs.
    Updates candidates + candidate_results in place.
    """
    if not _budget_ok(state):
        logger.warning("[repair] budget exceeded, skipping repair")
        return state

    problem: LCBProblem = state["problem"]
    model = state.get("model", LCB_MODEL_HARD)
    template_context = state.get("template_context", "")
    complexity = state.get("time_complexity_target", "")

    candidates = list(state.get("candidates", []))
    results = list(state.get("candidate_results", []))
    errors = list(state.get("candidate_errors", []))
    self_tests = state.get("self_tests", [])
    all_tests = self_tests + problem.test_cases
    cost = state.get("cost_usd", 0.0)
    history = list(state.get("repair_history", []))

    for idx, (code, (passed, total), err) in enumerate(zip(candidates, results, errors)):
        if passed == total and total > 0:
            continue   # already perfect
        if not err:
            continue   # no error signal to repair from

        for iteration in range(MAX_REPAIR_ITERATIONS):
            if not _budget_ok({**state, "cost_usd": cost}):
                break

            # Find the failing test case input for better error context
            failing_input = ""
            for tc in all_tests:
                _, _, tc_err = evaluate_solution(code, [tc])
                if tc_err:
                    failing_input = tc.stdin[:500]
                    break

            prompt = build_repair_prompt(
                problem, code, err, failing_input, template_context, complexity
            )
            try:
                new_code, resp = _llm(model, SYSTEM_PROMPT, prompt, temperature=0.3)
                cost += _track_cost(resp, {**state, "cost_usd": 0.0})
            except Exception as exc:
                logger.warning("[repair] idx=%d iter=%d LLM failed: %s", idx, iteration, exc)
                break

            new_passed, new_total, new_err = evaluate_solution(new_code, all_tests)
            history.append({
                "candidate_idx": idx,
                "iteration": iteration,
                "passed_before": passed,
                "passed_after": new_passed,
                "error": err[:200] if err else "",
            })

            if new_passed >= passed:   # accept if not worse
                candidates[idx] = new_code
                results[idx] = (new_passed, new_total)
                errors[idx] = new_err
                code, passed, err = new_code, new_passed, new_err

            if passed == total and total > 0:
                logger.info("[repair] candidate %d fixed at iteration %d", idx, iteration)
                break

    return {
        **state,
        "candidates": candidates,
        "candidate_results": results,
        "candidate_errors": errors,
        "repair_history": history,
        "cost_usd": cost,
        "audit_log": _audit(state, "repair", n_repairs=len(history)),
    }


def node_select_best(state: LCBState) -> LCBState:
    """
    Select best candidate using:
    1. Pass rate (primary)
    2. Majority vote on test outputs (tie-breaking)
    3. Code length (shorter = simpler, final tie-break)
    """
    candidates = state.get("candidates", [])
    results = state.get("candidate_results", [])
    strategies = state.get("candidate_strategies", [])
    problem: LCBProblem = state["problem"]
    self_tests = state.get("self_tests", [])
    all_tests = self_tests + problem.test_cases

    if not candidates:
        return {
            **state,
            "best_code": "",
            "best_passed": 0,
            "best_total": len(all_tests),
            "best_strategy": "none",
            "error": "no candidates generated",
        }

    # --- Primary: highest pass rate ---
    best_idx = max(range(len(candidates)), key=lambda i: results[i][0] if i < len(results) else 0)
    best_passed, best_total = results[best_idx] if best_idx < len(results) else (0, 0)

    # --- Majority vote among candidates with the same pass rate ---
    top_score = best_passed
    top_indices = [
        i for i, (p, _) in enumerate(results) if p == top_score
    ]

    if len(top_indices) > 1 and all_tests:
        # Collect outputs of all top candidates on all tests
        vote_map: Dict[int, Dict[str, int]] = {}  # test_idx → {output → count}
        candidate_outputs: Dict[int, List[str]] = {}
        for ci in top_indices:
            outputs = []
            for tc in all_tests:
                from .runner import _run_code
                out, _ = _run_code(candidates[ci], tc.stdin, timeout=5.0)
                outputs.append(out)
            candidate_outputs[ci] = outputs

        # Build majority per test
        majority_outputs: List[str] = []
        for ti in range(len(all_tests)):
            votes: Dict[str, int] = {}
            for ci in top_indices:
                o = candidate_outputs[ci][ti]
                votes[o] = votes.get(o, 0) + 1
            majority_outputs.append(max(votes, key=lambda k: votes[k]))

        # Score each top candidate by agreement with majority
        best_agreement = -1
        for ci in top_indices:
            agreement = sum(
                1 for ti, out in enumerate(candidate_outputs[ci])
                if out == majority_outputs[ti]
            )
            if agreement > best_agreement:
                best_agreement = agreement
                best_idx = ci
            elif agreement == best_agreement:
                # Tie-break: shorter code (simpler = fewer edge-case bugs)
                if len(candidates[ci]) < len(candidates[best_idx]):
                    best_idx = ci

    best_strategy = strategies[best_idx] if best_idx < len(strategies) else "unknown"
    logger.info(
        "[select_best] candidate %d/%d pass=%d/%d strategy=%s",
        best_idx + 1, len(candidates), best_passed, best_total, best_strategy,
    )

    return {
        **state,
        "best_code": candidates[best_idx],
        "best_passed": best_passed,
        "best_total": best_total,
        "best_strategy": best_strategy,
        "audit_log": _audit(state, "select_best",
                            best_idx=best_idx, pass_rate=f"{best_passed}/{best_total}"),
    }


# ── Graph construction ────────────────────────────────────────

def build_pipeline() -> Any:
    """
    Build and compile the LangGraph StateGraph for the LCB pipeline.
    Returns a compiled graph or raises ImportError if langgraph not installed.
    """
    if not _LANGGRAPH_AVAILABLE:
        raise ImportError("langgraph not installed — run: pip install langgraph")

    graph = StateGraph(LCBState)

    graph.add_node("classify",           node_classify)
    graph.add_node("extract_constraints", node_extract_constraints)
    graph.add_node("retrieve_templates", node_retrieve_templates)
    graph.add_node("generate_self_tests", node_generate_self_tests)
    graph.add_node("generate_candidates", node_generate_candidates)
    graph.add_node("evaluate_candidates", node_evaluate_candidates)
    graph.add_node("repair_candidates",  node_repair_candidates)
    graph.add_node("select_best",        node_select_best)

    graph.add_edge(START,                 "classify")
    graph.add_edge("classify",            "extract_constraints")
    graph.add_edge("extract_constraints", "retrieve_templates")
    graph.add_edge("retrieve_templates",  "generate_self_tests")
    graph.add_edge("generate_self_tests", "generate_candidates")
    graph.add_edge("generate_candidates", "evaluate_candidates")
    graph.add_edge("evaluate_candidates", "repair_candidates")
    graph.add_edge("repair_candidates",   "select_best")
    graph.add_edge("select_best",         END)

    return graph.compile()


# Module-level cached compiled graph (lazy init)
_PIPELINE: Any = None


def get_pipeline() -> Any:
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = build_pipeline()
    return _PIPELINE


def run_pipeline(problem: LCBProblem) -> LCBState:
    """
    Run the full pipeline for a single problem.
    Returns the final LCBState.
    Falls back to direct node calls if langgraph unavailable.
    """
    initial: LCBState = {
        "problem": problem,
        "cost_usd": 0.0,
        "audit_log": [],
    }

    if _LANGGRAPH_AVAILABLE:
        pipe = get_pipeline()
        return pipe.invoke(initial)
    else:
        # Fallback: execute nodes sequentially without LangGraph
        logger.warning("langgraph unavailable, running nodes sequentially")
        state = initial
        for node_fn in [
            node_classify,
            node_extract_constraints,
            node_retrieve_templates,
            node_generate_self_tests,
            node_generate_candidates,
            node_evaluate_candidates,
            node_repair_candidates,
            node_select_best,
        ]:
            state = node_fn(state)
        return state
