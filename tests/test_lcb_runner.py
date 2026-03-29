"""
tests/test_lcb_runner.py
========================
Offline tests for the LCB module.

Coverage:
  - classifier (difficulty routing)
  - algo_templates (RAG retrieval)
  - constraints (extraction + hints)
  - prompts (template rendering)
  - sandbox (code execution)
  - pipeline nodes (mocked LLM)
  - runner (top-level API)
  - eval (metrics + report building)
"""

from __future__ import annotations

import json
import textwrap
from typing import List
from unittest.mock import MagicMock, patch

import pytest

# ── Shared fixtures ───────────────────────────────────────────

from lcb.runner import LCBProblem, ProblemTestCase as TC, LCBRunner, LCBResult


def make_problem(
    stmt: str = "Multiply input by 2.",
    lcb_difficulty: str | None = None,
    test_cases: list | None = None,
    pid: str = "test_001",
) -> LCBProblem:
    tcs = test_cases or [TC(stdin="5", expected_stdout="10")]
    return LCBProblem(
        problem_id=pid,
        title="Test Problem",
        statement=stmt,
        test_cases=tcs,
        lcb_difficulty=lcb_difficulty,
    )


# ═══════════════════════════════════════════════════════════════
# 1. Classifier
# ═══════════════════════════════════════════════════════════════

from lcb.classifier import classify_difficulty, classify_from_metadata


class TestClassifier:
    def test_easy_simulation(self):
        d, _, _ = classify_difficulty("Simulate the process and output the result.")
        assert d == "easy"

    def test_hard_dp(self):
        d, score, hits = classify_difficulty(
            "Use dynamic programming to find the longest common subsequence."
        )
        assert d == "hard"
        assert score >= 3

    def test_hard_graph_shortest_path(self):
        d, _, _ = classify_difficulty(
            "Find the shortest path in a weighted graph using Dijkstra's algorithm."
        )
        assert d == "hard"

    def test_hard_segment_tree(self):
        d, _, hits = classify_difficulty("Support range queries using a segment tree.")
        assert d == "hard"
        assert "segment tree" in hits

    def test_hard_fenwick(self):
        d, _, hits = classify_difficulty("Binary indexed tree prefix sum with point update.")
        assert d == "hard"

    def test_hard_fft(self):
        d, score, _ = classify_difficulty(
            "Multiply two polynomials using fast Fourier transform."
        )
        assert d == "hard"
        assert score >= 3

    def test_metadata_easy_overrides_keywords(self):
        # Normally hard, but metadata says easy
        d, _, _ = classify_from_metadata(
            "Dynamic programming graph theory problem.", lcb_difficulty="easy"
        )
        assert d == "easy"

    def test_metadata_medium_is_hard(self):
        d, _, _ = classify_from_metadata("Simple loop.", lcb_difficulty="medium")
        assert d == "hard"

    def test_metadata_hard(self):
        d, _, _ = classify_from_metadata("Simple loop.", lcb_difficulty="hard")
        assert d == "hard"

    def test_no_metadata_falls_back(self):
        d, _, _ = classify_from_metadata(
            "Use bfs on a graph.", lcb_difficulty=None
        )
        assert d == "hard"

    def test_empty_statement(self):
        d, score, hits = classify_difficulty("")
        assert d == "easy"
        assert score == 0
        assert hits == []

    def test_hits_reported(self):
        _, _, hits = classify_difficulty("Use bfs to traverse the graph.")
        assert len(hits) > 0


# ═══════════════════════════════════════════════════════════════
# 2. Algorithm templates
# ═══════════════════════════════════════════════════════════════

from lcb.algo_templates import retrieve_templates, format_templates_for_prompt, _TEMPLATES
import ast


class TestAlgoTemplates:
    def test_dijkstra_retrieved(self):
        templates = retrieve_templates("shortest path in a weighted graph", top_k=2)
        assert any("Dijkstra" in t.name for t in templates)

    def test_dp_retrieved(self):
        templates = retrieve_templates("dynamic programming coin change", top_k=2)
        assert any("Dynamic" in t.name for t in templates)

    def test_no_match_empty(self):
        templates = retrieve_templates("hello world program", top_k=2)
        assert templates == []

    def test_top_k_respected(self):
        templates = retrieve_templates(
            "graph bfs dfs shortest path dijkstra dynamic programming dp fenwick trie",
            top_k=1,
        )
        assert len(templates) <= 1

    def test_format_nonempty(self):
        templates = retrieve_templates("dijkstra shortest path weighted graph", top_k=1)
        out = format_templates_for_prompt(templates)
        assert "```python" in out
        assert "Dijkstra" in out

    def test_format_empty(self):
        assert format_templates_for_prompt([]) == ""

    def test_segment_tree_retrieved(self):
        templates = retrieve_templates("range queries range updates segment tree", top_k=2)
        assert any("Segment Tree" in t.name for t in templates)

    def test_dsu_retrieved(self):
        templates = retrieve_templates("union find disjoint set connected components", top_k=2)
        assert any("Union" in t.name for t in templates)

    def test_binary_search_retrieved(self):
        templates = retrieve_templates(
            "binary search on the answer smallest feasible value", top_k=3
        )
        assert any("Binary Search" in t.name for t in templates)

    def test_all_templates_valid_python(self):
        for tmpl in _TEMPLATES:
            try:
                ast.parse(tmpl.code)
            except SyntaxError as exc:
                pytest.fail(f"Syntax error in '{tmpl.name}': {exc}")

    def test_templates_have_keywords(self):
        for tmpl in _TEMPLATES:
            assert len(tmpl.keywords) > 0, f"Template '{tmpl.name}' has no keywords"


# ═══════════════════════════════════════════════════════════════
# 3. Constraint extraction
# ═══════════════════════════════════════════════════════════════

from lcb.constraints import extract_constraints, infer_algorithm_hints


class TestConstraints:
    def test_n_max_power(self):
        c = extract_constraints("1 ≤ N ≤ 10^5, 1 ≤ M ≤ 10^4")
        assert c["n_max"] == 100_000
        assert c["m_max"] == 10_000

    def test_n_max_plain_number(self):
        c = extract_constraints("N ≤ 200000")
        assert c["n_max"] == 200_000

    def test_graph_structure(self):
        c = extract_constraints("Given a weighted directed graph with N nodes and M edges.")
        assert c["structure"] == "graph"
        assert c["is_weighted"] is True
        assert c["is_directed"] is True

    def test_string_structure(self):
        c = extract_constraints("Given a string S of length N.")
        assert c["structure"] == "string"

    def test_tree_structure(self):
        c = extract_constraints("Given a rooted tree with N nodes and a parent array.")
        assert c["structure"] == "tree"

    def test_grid_structure(self):
        c = extract_constraints("Given an N×M grid where each cell has a value.")
        assert c["structure"] == "grid"

    def test_interactive(self):
        c = extract_constraints("This is an interactive problem. Query the judge and answer.")
        assert c["is_interactive"] is True

    def test_empty_statement(self):
        c = extract_constraints("")
        assert c["n_max"] == 0
        assert c["structure"] == "unknown"

    def test_hints_large_n(self):
        c = {"n_max": 200_000, "structure": "array", "is_weighted": False, "is_directed": False}
        hints, complexity = infer_algorithm_hints(c)
        assert "O(n log n)" in complexity
        assert len(hints) > 0

    def test_hints_small_n_bitmask(self):
        c = {"n_max": 18, "structure": "array", "is_weighted": False, "is_directed": False}
        hints, complexity = infer_algorithm_hints(c)
        assert "bitmask_dp" in hints or "backtracking" in hints

    def test_hints_graph_weighted(self):
        c = {"n_max": 100_000, "structure": "graph", "is_weighted": True, "is_directed": False}
        hints, _ = infer_algorithm_hints(c)
        assert "dijkstra" in hints

    def test_hints_string(self):
        c = {"n_max": 100_000, "structure": "string", "is_weighted": False, "is_directed": False}
        hints, _ = infer_algorithm_hints(c)
        assert "kmp" in hints or "hashing" in hints

    def test_hints_no_duplicates(self):
        c = {"n_max": 100_000, "structure": "graph", "is_weighted": True, "is_directed": True}
        hints, _ = infer_algorithm_hints(c)
        assert len(hints) == len(set(hints))


# ═══════════════════════════════════════════════════════════════
# 4. Prompts
# ═══════════════════════════════════════════════════════════════

from lcb.prompts import (
    build_direct_prompt, build_plan_then_code_prompt,
    build_analogy_prompt, build_repair_prompt,
    build_self_test_prompt, CANDIDATE_PLAN,
)


class TestPrompts:
    def test_direct_contains_title(self):
        p = make_problem("Find the max subarray sum.", lcb_difficulty="easy")
        prompt = build_direct_prompt(p)
        assert p.title in prompt

    def test_direct_with_template_context(self):
        p = make_problem()
        prompt = build_direct_prompt(p, template_context="## Algorithm Templates\n### BFS\n")
        assert "Algorithm Templates" in prompt

    def test_plan_then_code_contains_hints(self):
        p = make_problem("Find shortest path in a graph.")
        prompt = build_plan_then_code_prompt(p, algorithm_hints=["dijkstra", "bfs"])
        assert "dijkstra" in prompt.lower() or "bfs" in prompt.lower()

    def test_analogy_contains_family(self):
        p = make_problem("Find the longest path in a DAG.")
        prompt = build_analogy_prompt(p, algorithm_hints=["dag_dp", "topological_sort"])
        assert p.statement in prompt

    def test_repair_contains_broken_code(self):
        p = make_problem()
        broken = "print(int(input()) + 1)"
        prompt = build_repair_prompt(p, broken, "Wrong answer: got 6, expected 10")
        assert broken in prompt
        assert "Wrong answer" in prompt

    def test_repair_tle_hint(self):
        p = make_problem()
        prompt = build_repair_prompt(
            p, "for i in range(n): for j in range(n): pass",
            "TLE: exceeded 2s", time_complexity_target="O(n log n)"
        )
        assert "O(n log n)" in prompt

    def test_self_test_prompt_contains_statement(self):
        p = make_problem("Given an array, find the maximum sum subarray.")
        constraints = {"n_max": 100_000, "structure": "array"}
        prompt = build_self_test_prompt(p, constraints)
        assert p.statement in prompt
        assert "JSON" in prompt

    def test_candidate_plan_easy_has_2(self):
        assert len(CANDIDATE_PLAN["easy"]) == 2

    def test_candidate_plan_hard_has_4(self):
        assert len(CANDIDATE_PLAN["hard"]) == 4

    def test_candidate_plan_hard_has_plan_strategy(self):
        assert "plan_then_code" in CANDIDATE_PLAN["hard"]

    def test_candidate_plan_hard_has_analogy(self):
        assert "analogy" in CANDIDATE_PLAN["hard"]


# ═══════════════════════════════════════════════════════════════
# 5. Sandbox execution
# ═══════════════════════════════════════════════════════════════

from lcb.runner import evaluate_solution, _run_code


class TestSandbox:
    def test_correct_solution(self):
        code = "print(int(input()) * 2)"
        passed, total, err = evaluate_solution(code, [TC("5", "10")])
        assert passed == 1 and total == 1 and err is None

    def test_wrong_answer(self):
        code = "print(int(input()) + 1)"
        passed, total, err = evaluate_solution(code, [TC("5", "10")])
        assert passed == 0 and "Wrong answer" in (err or "")

    def test_runtime_error(self):
        _, _, err = evaluate_solution("x = 1/0", [TC("", "anything")])
        assert err is not None

    def test_no_test_cases(self):
        p, t, e = evaluate_solution("print('hi')", [])
        assert p == 0 and t == 0 and e is None

    def test_partial_pass(self):
        code = "n = int(input()); print(n * 2)"
        tcs = [TC("3", "6"), TC("5", "10"), TC("7", "99")]
        passed, total, _ = evaluate_solution(code, tcs)
        assert passed == 2 and total == 3

    def test_timeout_tle(self):
        _, _, err = evaluate_solution("while True: pass", [TC("", "")], timeout=0.3)
        assert err is not None and "TLE" in err

    def test_multiline_output(self):
        code = textwrap.dedent("""\
            n = int(input())
            for i in range(n):
                print(i)
        """)
        passed, _, _ = evaluate_solution(code, [TC("3", "0\n1\n2")])
        assert passed == 1

    def test_run_code_returns_stdout(self):
        out, err = _run_code("print(42)", "", timeout=2.0)
        assert out == "42"
        assert err is None

    def test_run_code_captures_stderr(self):
        _, err = _run_code("raise ValueError('oops')", "", timeout=2.0)
        assert err is not None and "ValueError" in err


# ═══════════════════════════════════════════════════════════════
# 6. Pipeline nodes (mocked LLM)
# ═══════════════════════════════════════════════════════════════

from lcb.pipeline import (
    node_classify, node_extract_constraints, node_retrieve_templates,
    node_generate_self_tests, node_generate_candidates, node_evaluate_candidates,
    node_repair_candidates, node_select_best,
)
from lcb.state import LCBState


def _initial_state(problem: LCBProblem) -> LCBState:
    return {"problem": problem, "cost_usd": 0.0, "audit_log": []}


class TestPipelineNodes:
    def test_node_classify_easy(self):
        s = node_classify(_initial_state(make_problem(lcb_difficulty="easy")))
        assert s["difficulty"] == "easy"
        assert "gpt-4o-mini" in s["model"]

    def test_node_classify_hard(self):
        problem = make_problem(
            "Find shortest path using Dijkstra on a weighted graph.", lcb_difficulty="hard"
        )
        s = node_classify(_initial_state(problem))
        assert s["difficulty"] == "hard"

    def test_node_extract_constraints(self):
        problem = make_problem("Given 1 ≤ N ≤ 10^5 nodes in a graph.")
        s = node_classify(_initial_state(problem))
        s = node_extract_constraints(s)
        assert s["constraints"]["n_max"] == 100_000
        assert len(s["algorithm_hints"]) > 0

    def test_node_retrieve_templates(self):
        problem = make_problem("Find shortest path using dijkstra algorithm.")
        s = node_classify(_initial_state(problem))
        s = node_extract_constraints(s)
        s = node_retrieve_templates(s)
        assert "template_names" in s

    def test_node_generate_self_tests_budget_exceeded(self):
        problem = make_problem()
        s = {**_initial_state(problem), "cost_usd": 999.0, "model": "gpt-4o-mini"}
        s = node_generate_self_tests(s)
        assert s["self_tests"] == []
        assert s.get("self_tests_raw") == "budget_exceeded"

    def test_node_generate_self_tests_llm_mock(self):
        tests_json = json.dumps([
            {"stdin": "1", "stdout": "2"},
            {"stdin": "0", "stdout": "0"},
        ])
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = tests_json
        with patch("lcb.pipeline._llm", return_value=(tests_json, mock_resp)):
            problem = make_problem()
            s = {**_initial_state(problem), "model": "gpt-4o-mini", "constraints": {}}
            s = node_generate_self_tests(s)
        assert len(s["self_tests"]) == 2
        assert s["self_tests"][0].stdin == "1"

    def test_node_generate_candidates_mocked(self):
        correct_code = "print(int(input()) * 2)"
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = correct_code
        with patch("lcb.pipeline._llm", return_value=(correct_code, mock_resp)):
            problem = make_problem(lcb_difficulty="easy")
            s = node_classify(_initial_state(problem))
            s = node_extract_constraints(s)
            s = node_retrieve_templates(s)
            s = {**s, "self_tests": []}
            s = node_generate_candidates(s)
        assert len(s["candidates"]) == 2  # easy = 2 candidates
        assert all(correct_code in c for c in s["candidates"])

    def test_node_evaluate_candidates(self):
        problem = make_problem(test_cases=[TC("5", "10"), TC("3", "6")])
        s = {
            **_initial_state(problem),
            "candidates": ["print(int(input()) * 2)", "print(int(input()) + 1)"],
            "self_tests": [],
        }
        s = node_evaluate_candidates(s)
        results = s["candidate_results"]
        assert results[0] == (2, 2)   # correct
        assert results[1] == (0, 2)   # wrong

    def test_node_repair_with_mock(self):
        fixed_code = "print(int(input()) * 2)"
        mock_resp = MagicMock()
        mock_resp.choices[0].message.content = fixed_code
        with patch("lcb.pipeline._llm", return_value=(fixed_code, mock_resp)):
            problem = make_problem(test_cases=[TC("5", "10")])
            s = {
                **_initial_state(problem),
                "model": "gpt-4o-mini",
                "candidates": ["print(42)"],         # wrong
                "candidate_results": [(0, 1)],
                "candidate_errors": ["Wrong answer: got 42, expected 10"],
                "self_tests": [],
                "template_context": "",
                "time_complexity_target": "O(n)",
                "repair_history": [],
            }
            s = node_repair_candidates(s)
        assert s["candidate_results"][0][0] == 1    # fixed

    def test_node_select_best_picks_highest_pass(self):
        problem = make_problem(test_cases=[TC("5", "10")])
        s = {
            **_initial_state(problem),
            "candidates": ["print(int(input()) + 1)", "print(int(input()) * 2)"],
            "candidate_results": [(0, 1), (1, 1)],
            "candidate_errors": ["wrong", None],
            "candidate_strategies": ["direct", "plan_then_code"],
            "self_tests": [],
        }
        s = node_select_best(s)
        assert s["best_passed"] == 1
        assert s["best_code"] == "print(int(input()) * 2)"
        assert s["best_strategy"] == "plan_then_code"

    def test_node_select_best_no_candidates(self):
        problem = make_problem()
        s = {**_initial_state(problem), "candidates": [], "candidate_results": [],
             "candidate_errors": [], "candidate_strategies": [], "self_tests": []}
        s = node_select_best(s)
        assert s["best_code"] == ""
        assert s["error"] is not None


# ═══════════════════════════════════════════════════════════════
# 7. Runner (top-level API)
# ═══════════════════════════════════════════════════════════════

import os


class TestRunner:
    def _mock_llm_call(self, code: str = "print(int(input()) * 2)", cost: float = 0.001):
        mock = MagicMock(return_value=(code, cost))
        return patch("lcb.runner._llm_call", mock)

    def test_easy_uses_mini_model(self):
        problem = make_problem(lcb_difficulty="easy")
        with self._mock_llm_call():
            result = LCBRunner(easy_candidates=1).solve(problem)
        assert result.difficulty == "easy"
        assert result.model == os.getenv("LCB_MODEL_EASY", "gpt-4o-mini")

    def test_hard_uses_deepseek(self):
        problem = make_problem(
            "Dijkstra shortest path on weighted graph.", lcb_difficulty="hard"
        )
        with self._mock_llm_call():
            result = LCBRunner(hard_candidates=1, max_repair=0).solve(problem)
        assert result.difficulty == "hard"
        assert result.model == os.getenv("LCB_MODEL_HARD", "deepseek/deepseek-chat")

    def test_success_when_all_pass(self):
        problem = make_problem(test_cases=[TC("3", "6"), TC("5", "10")])
        with self._mock_llm_call("print(int(input()) * 2)"):
            result = LCBRunner(easy_candidates=1).solve(problem)
        assert result.success is True and result.passed == 2

    def test_fail_when_wrong_answer(self):
        problem = make_problem(test_cases=[TC("5", "10")])
        with self._mock_llm_call("print(int(input()) + 1)"):
            result = LCBRunner(easy_candidates=1).solve(problem)
        assert result.success is False and result.passed == 0

    def test_cost_accumulated(self):
        problem = make_problem(test_cases=[TC("5", "10")])
        with self._mock_llm_call(cost=0.002):
            result = LCBRunner(easy_candidates=2).solve(problem)
        assert result.cost_usd > 0

    def test_repair_invoked(self):
        """Repair should be called when initial solution is wrong."""
        problem = make_problem(
            "Dijkstra shortest path on weighted graph.",
            lcb_difficulty="hard",
            test_cases=[TC("5", "10")],
        )
        calls = [
            ("print(42)", 0.01),                    # initial: wrong
            ("print(int(input()) * 2)", 0.01),       # repair 1: correct
        ]
        call_iter = iter(calls)

        def fake(model, prompt, temperature):
            return next(call_iter)

        with patch("lcb.runner._llm_call", side_effect=fake):
            result = LCBRunner(hard_candidates=1, max_repair=2).solve(problem)

        assert result.repair_iterations >= 1
        assert result.success is True

    def test_summarise_empty(self):
        assert LCBRunner.summarise([]) == {}

    def test_summarise_full(self):
        results = [
            LCBResult("p1", "easy", "gpt-4o-mini", "", 1, 1, True, 0.001, 0.5, 2, 0, 0, []),
            LCBResult("p2", "easy", "gpt-4o-mini", "", 0, 1, False, 0.001, 0.5, 2, 0, 0, []),
            LCBResult("p3", "hard", "deepseek/deepseek-chat", "", 1, 1, True, 0.02, 2.0, 4, 2, 3, []),
            LCBResult("p4", "hard", "deepseek/deepseek-chat", "", 0, 1, False, 0.02, 2.0, 4, 1, 3, []),
        ]
        s = LCBRunner.summarise(results)
        assert s["total_problems"] == 4
        assert s["solved"] == 2
        assert s["easy_count"] == 2
        assert s["hard_count"] == 2
        assert s["avg_repair_iterations_hard"] == pytest.approx(1.5)


# ═══════════════════════════════════════════════════════════════
# 8. Eval (metrics + report)
# ═══════════════════════════════════════════════════════════════

from lcb.eval import LCBEvaluator, pass_at_k, ProblemRecord, _load_local_dataset
import tempfile


class TestEval:
    def test_pass_at_k_all_pass(self):
        assert pass_at_k(5, 5, 1) == pytest.approx(1.0)

    def test_pass_at_k_none_pass(self):
        assert pass_at_k(5, 0, 1) == pytest.approx(0.0)

    def test_pass_at_k_partial(self):
        # 3 out of 5 pass, k=1: should be > 0 and < 1
        v = pass_at_k(5, 3, 1)
        assert 0.0 < v < 1.0

    def test_pass_at_k_k_equals_n(self):
        # If k = n and c > 0, should be 1.0
        assert pass_at_k(5, 1, 5) == pytest.approx(1.0)

    def test_load_local_dataset(self):
        data = [
            {
                "id": "p1",
                "title": "T1",
                "statement": "Find max.",
                "difficulty": "easy",
                "test_cases": [{"stdin": "5", "stdout": "5"}],
            },
            {
                "id": "p2",
                "title": "T2",
                "statement": "Find min.",
                "difficulty": "hard",
                "test_cases": [],
            },
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            path = f.name

        problems = _load_local_dataset(path)
        assert len(problems) == 2
        assert problems[0].problem_id == "p1"
        assert len(problems[0].test_cases) == 1
        assert problems[1].lcb_difficulty == "hard"

    def test_build_report(self):
        records = [
            ProblemRecord("p1", "T1", "easy", "gpt-4o-mini", "direct",
                          1, 1, True, 0.001, 0.5, 0, 0, None),
            ProblemRecord("p2", "T2", "easy", "gpt-4o-mini", "direct",
                          0, 1, False, 0.001, 0.5, 0, 0, "wrong"),
            ProblemRecord("p3", "T3", "hard", "deepseek/deepseek-chat", "plan_then_code",
                          1, 1, True, 0.02, 2.0, 2, 3, None),
        ]
        report = LCBEvaluator._build_report(records)
        assert report.total_problems == 3
        assert report.solved == 2
        assert report.easy_count == 2
        assert report.hard_count == 1
        assert report.easy_accuracy == pytest.approx(0.5)
        assert report.hard_accuracy == pytest.approx(1.0)

    def test_save_report_json(self):
        records = [
            ProblemRecord("p1", "T1", "easy", "gpt-4o-mini", "direct",
                          1, 1, True, 0.001, 0.5, 0, 0, None),
        ]
        report = LCBEvaluator._build_report(records)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        LCBEvaluator.save_report(report, path)
        with open(path) as f:
            data = json.load(f)
        assert data["summary"]["total_problems"] == 1
        assert len(data["records"]) == 1

    def test_save_csv(self):
        import csv as csv_mod
        records = [
            ProblemRecord("p1", "T1", "easy", "gpt-4o-mini", "direct",
                          1, 1, True, 0.001, 0.5, 0, 0, None),
        ]
        report = LCBEvaluator._build_report(records)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            path = f.name
        LCBEvaluator.save_csv(report, path)
        with open(path) as f:
            rows = list(csv_mod.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["problem_id"] == "p1"

    def test_evaluator_run_with_mock(self):
        """Full evaluator.run() with mocked LCBRunner.solve."""
        problems = [
            make_problem("Simple task.", lcb_difficulty="easy", pid="p1"),
            make_problem("Graph problem.", lcb_difficulty="hard", pid="p2"),
        ]

        def fake_solve(problem: LCBProblem) -> LCBResult:
            return LCBResult(
                problem_id=problem.problem_id,
                difficulty=problem.lcb_difficulty or "easy",
                model="mock",
                best_code="print(42)",
                passed=1,
                total=1,
                success=True,
                cost_usd=0.001,
                elapsed_sec=0.1,
                candidates_tried=1,
                repair_iterations=0,
                classifier_score=0,
                classifier_hits=[],
            )

        evaluator = LCBEvaluator(use_pipeline=False)
        with patch.object(evaluator._runner, "solve", side_effect=fake_solve):
            report = evaluator.run(problems=problems)

        assert report.total_problems == 2
        assert report.solved == 2
        assert report.accuracy == pytest.approx(1.0)
