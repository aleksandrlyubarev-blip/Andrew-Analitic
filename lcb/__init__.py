"""
lcb — Andrew LCB Runner
=======================
Difficulty-routed competitive programming solver for LiveCodeBench.

Pipeline (LangGraph):
  classify → extract_constraints → retrieve_templates → generate_self_tests
    → generate_candidates → evaluate_candidates → repair_candidates → select_best

Routing:
  easy (55%)  → GPT-4o-mini   1-2 candidates, no repair
  hard (45%)  → DeepSeek V3   4 candidates + 3 repair iterations + RAG templates

Usage (full pipeline):
    from lcb.pipeline import run_pipeline
    from lcb.runner import LCBProblem, ProblemTestCase
    problem = LCBProblem(...)
    state = run_pipeline(problem)

Usage (simple runner):
    from lcb.runner import LCBRunner
    runner = LCBRunner()
    result = runner.solve(problem)

Usage (batch evaluation):
    from lcb.eval import LCBEvaluator
    ev = LCBEvaluator(dataset_path="problems.json")
    report = ev.run()
    ev.print_summary(report)
"""
from .runner import LCBRunner, LCBResult, LCBProblem, ProblemTestCase
from .eval import LCBEvaluator, EvalReport

__all__ = [
    "LCBRunner", "LCBResult", "LCBProblem", "ProblemTestCase",
    "LCBEvaluator", "EvalReport",
]
