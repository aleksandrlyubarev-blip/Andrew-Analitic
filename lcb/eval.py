"""
lcb/eval.py
===========
Evaluation harness for LiveCodeBench.

Supports:
  - Loading from HuggingFace: livecodebench/code_generation_lite
  - Loading from local JSON
  - pass@1 and pass@k metrics
  - Per-difficulty breakdown
  - CSV / JSON result export

Usage:
    # Run against HF dataset
    from lcb.eval import LCBEvaluator
    evaluator = LCBEvaluator()
    report = evaluator.run(max_problems=100)
    evaluator.save_report(report, "results.json")

    # Run against local file
    evaluator = LCBEvaluator(dataset_path="problems.json")
    report = evaluator.run()
"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

from .pipeline import run_pipeline
from .runner import LCBProblem, LCBResult, ProblemTestCase, LCBRunner
from .state import LCBState

logger = logging.getLogger("lcb.eval")

# ── Dataset loading ───────────────────────────────────────────

def _load_hf_dataset(split: str = "test", max_problems: Optional[int] = None) -> List[LCBProblem]:
    """
    Load problems from HuggingFace livecodebench/code_generation_lite.

    Dataset fields (may vary by version):
      question_content / problem_statement — problem text
      difficulty                           — easy/medium/hard
      public_test_cases                    — [{"input": ..., "output": ...}]
      question_id / id                     — unique identifier
      title                                — problem name
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "HuggingFace datasets not installed. "
            "Run: pip install datasets"
        )

    ds = load_dataset("livecodebench/code_generation_lite", split=split, trust_remote_code=True)
    problems: List[LCBProblem] = []

    for row in ds:
        if max_problems and len(problems) >= max_problems:
            break

        problem_id = str(row.get("question_id", row.get("id", f"lcb_{len(problems):04d}")))
        title = str(row.get("title", row.get("question_title", problem_id)))
        statement = str(row.get("question_content", row.get("problem_statement", "")))
        difficulty = str(row.get("difficulty", "")).lower() or None

        # Parse test cases
        raw_tests = row.get("public_test_cases", row.get("test_cases", []))
        test_cases: List[ProblemTestCase] = []
        if isinstance(raw_tests, str):
            try:
                raw_tests = json.loads(raw_tests)
            except json.JSONDecodeError:
                raw_tests = []
        for tc in (raw_tests or []):
            if isinstance(tc, dict):
                stdin = str(tc.get("input", tc.get("stdin", "")))
                stdout = str(tc.get("output", tc.get("stdout", ""))).strip()
                test_cases.append(ProblemTestCase(stdin=stdin, expected_stdout=stdout))

        problems.append(LCBProblem(
            problem_id=problem_id,
            title=title,
            statement=statement,
            test_cases=test_cases,
            lcb_difficulty=difficulty,
        ))

    logger.info("Loaded %d problems from HuggingFace LCB dataset", len(problems))
    return problems


def _load_local_dataset(path: str) -> List[LCBProblem]:
    """
    Load problems from a local JSON file.

    Expected format (list of objects):
    [
      {
        "id": "lcb_001",
        "title": "Two Sum",
        "statement": "...",
        "difficulty": "easy",
        "test_cases": [{"stdin": "...", "stdout": "..."}]
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    problems: List[LCBProblem] = []
    for i, item in enumerate(data):
        test_cases = [
            ProblemTestCase(stdin=str(tc["stdin"]), expected_stdout=str(tc["stdout"]).strip())
            for tc in item.get("test_cases", [])
            if "stdin" in tc and "stdout" in tc
        ]
        problems.append(LCBProblem(
            problem_id=str(item.get("id", item.get("problem_id", f"local_{i:04d}"))),
            title=str(item.get("title", f"Problem {i}")),
            statement=str(item.get("statement", item.get("question_content", ""))),
            test_cases=test_cases,
            lcb_difficulty=item.get("difficulty"),
        ))

    logger.info("Loaded %d problems from %s", len(problems), path)
    return problems


# ── Metrics ───────────────────────────────────────────────────

def pass_at_k(n: int, c: int, k: int) -> float:
    """
    Unbiased estimator of pass@k.
    n = number of samples, c = number that pass, k = k in pass@k.
    Formula from Codex paper (Chen et al. 2021):
      pass@k = 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    if n < k:
        return 0.0
    # Use log to avoid overflow
    def log_comb(a: int, b: int) -> float:
        if b > a or b < 0:
            return float("-inf")
        return sum(math.log(a - i) - math.log(i + 1) for i in range(b))

    return 1.0 - math.exp(log_comb(n - c, k) - log_comb(n, k))


# ── Result types ──────────────────────────────────────────────

@dataclass
class ProblemRecord:
    problem_id: str
    title: str
    difficulty: str
    model: str
    strategy: str
    passed: int
    total: int
    success: bool
    cost_usd: float
    elapsed_sec: float
    repair_iterations: int
    classifier_score: int
    error: Optional[str]


@dataclass
class EvalReport:
    timestamp: str
    total_problems: int
    solved: int
    accuracy: float
    pass_at_1: float
    easy_accuracy: float
    hard_accuracy: float
    easy_count: int
    hard_count: int
    total_cost_usd: float
    avg_cost_per_problem: float
    avg_repair_iterations: float
    records: List[ProblemRecord]


# ── Evaluator ─────────────────────────────────────────────────

class LCBEvaluator:
    """
    End-to-end evaluation harness.

    Args:
        dataset_path: Path to local JSON dataset. If None, loads from HuggingFace.
        hf_split: HuggingFace dataset split (default: 'test').
        on_result: Optional callback called after each problem with its ProblemRecord.
        use_pipeline: If True, use full LangGraph pipeline. If False, use simpler
                      LCBRunner (faster for ablations).
    """

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        hf_split: str = "test",
        on_result: Optional[Callable[[ProblemRecord], None]] = None,
        use_pipeline: bool = True,
    ):
        self.dataset_path = dataset_path
        self.hf_split = hf_split
        self.on_result = on_result
        self.use_pipeline = use_pipeline
        self._runner = LCBRunner() if not use_pipeline else None

    def load_problems(self, max_problems: Optional[int] = None) -> List[LCBProblem]:
        if self.dataset_path:
            problems = _load_local_dataset(self.dataset_path)
        else:
            problems = _load_hf_dataset(self.hf_split, max_problems)
        if max_problems:
            problems = problems[:max_problems]
        return problems

    def _solve_one(self, problem: LCBProblem) -> ProblemRecord:
        t0 = time.monotonic()

        if self.use_pipeline:
            final_state: LCBState = run_pipeline(problem)
            passed = final_state.get("best_passed", 0)
            total = final_state.get("best_total", 0)
            cost = final_state.get("cost_usd", 0.0)
            difficulty = final_state.get("difficulty", "unknown")
            model = final_state.get("model", "unknown")
            strategy = final_state.get("best_strategy", "unknown")
            classifier_score = final_state.get("classifier_score", 0)
            repair_iters = sum(
                1 for h in final_state.get("repair_history", [])
            )
            error = final_state.get("error")
        else:
            result: LCBResult = self._runner.solve(problem)
            passed = result.passed
            total = result.total
            cost = result.cost_usd
            difficulty = result.difficulty
            model = result.model
            strategy = "direct"
            classifier_score = result.classifier_score
            repair_iters = result.repair_iterations
            error = result.error

        elapsed = time.monotonic() - t0
        success = (total > 0 and passed == total)

        return ProblemRecord(
            problem_id=problem.problem_id,
            title=problem.title,
            difficulty=difficulty,
            model=model,
            strategy=strategy,
            passed=passed,
            total=total,
            success=success,
            cost_usd=cost,
            elapsed_sec=elapsed,
            repair_iterations=repair_iters,
            classifier_score=classifier_score,
            error=error,
        )

    def run(
        self,
        problems: Optional[List[LCBProblem]] = None,
        max_problems: Optional[int] = None,
    ) -> EvalReport:
        """
        Run evaluation over the dataset.
        Returns an EvalReport with per-problem records and aggregate metrics.
        """
        if problems is None:
            problems = self.load_problems(max_problems)
        elif max_problems:
            problems = problems[:max_problems]

        records: List[ProblemRecord] = []

        for i, problem in enumerate(problems):
            logger.info(
                "Solving %d/%d: %s (%s)",
                i + 1, len(problems), problem.problem_id, problem.title[:40],
            )
            try:
                record = self._solve_one(problem)
            except Exception as exc:
                logger.error("Problem %s crashed: %s", problem.problem_id, exc)
                record = ProblemRecord(
                    problem_id=problem.problem_id,
                    title=problem.title,
                    difficulty="unknown",
                    model="unknown",
                    strategy="unknown",
                    passed=0,
                    total=len(problem.test_cases),
                    success=False,
                    cost_usd=0.0,
                    elapsed_sec=0.0,
                    repair_iterations=0,
                    classifier_score=0,
                    error=str(exc),
                )
            records.append(record)

            if self.on_result:
                self.on_result(record)

            logger.info(
                "  → %s pass=%d/%d cost=$%.4f",
                "✓" if record.success else "✗",
                record.passed, record.total, record.cost_usd,
            )

        return self._build_report(records)

    @staticmethod
    def _build_report(records: List[ProblemRecord]) -> EvalReport:
        from datetime import datetime, timezone
        total = len(records)
        solved = sum(1 for r in records if r.success)
        easy = [r for r in records if r.difficulty == "easy"]
        hard = [r for r in records if r.difficulty == "hard"]
        total_cost = sum(r.cost_usd for r in records)
        avg_repair = (
            sum(r.repair_iterations for r in records) / total if total else 0.0
        )

        # pass@1 = fraction solved (since we pick 1 best answer)
        p1 = solved / total if total else 0.0

        return EvalReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            total_problems=total,
            solved=solved,
            accuracy=p1,
            pass_at_1=p1,
            easy_accuracy=sum(1 for r in easy if r.success) / len(easy) if easy else 0.0,
            hard_accuracy=sum(1 for r in hard if r.success) / len(hard) if hard else 0.0,
            easy_count=len(easy),
            hard_count=len(hard),
            total_cost_usd=total_cost,
            avg_cost_per_problem=total_cost / total if total else 0.0,
            avg_repair_iterations=avg_repair,
            records=records,
        )

    @staticmethod
    def save_report(report: EvalReport, path: str) -> None:
        """Save full report to JSON."""
        data = {
            "timestamp": report.timestamp,
            "summary": {
                "total_problems": report.total_problems,
                "solved": report.solved,
                "accuracy": report.accuracy,
                "pass_at_1": report.pass_at_1,
                "easy_accuracy": report.easy_accuracy,
                "hard_accuracy": report.hard_accuracy,
                "easy_count": report.easy_count,
                "hard_count": report.hard_count,
                "total_cost_usd": report.total_cost_usd,
                "avg_cost_per_problem": report.avg_cost_per_problem,
                "avg_repair_iterations": report.avg_repair_iterations,
            },
            "records": [asdict(r) for r in report.records],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Report saved to %s", path)

    @staticmethod
    def save_csv(report: EvalReport, path: str) -> None:
        """Save per-problem records to CSV for analysis."""
        if not report.records:
            return
        fields = list(asdict(report.records[0]).keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in report.records:
                writer.writerow(asdict(r))
        logger.info("CSV saved to %s", path)

    @staticmethod
    def print_summary(report: EvalReport) -> None:
        """Pretty-print the evaluation summary."""
        print(f"\n{'='*60}")
        print(f"Andrew LCB Evaluation Report  {report.timestamp}")
        print(f"{'='*60}")
        print(f"  Total problems : {report.total_problems}")
        print(f"  Solved         : {report.solved}  ({report.accuracy:.1%})")
        print(f"  pass@1         : {report.pass_at_1:.1%}")
        print(f"  Easy  ({report.easy_count:3d})    : {report.easy_accuracy:.1%}")
        print(f"  Hard  ({report.hard_count:3d})    : {report.hard_accuracy:.1%}")
        print(f"  Total cost     : ${report.total_cost_usd:.2f}")
        print(f"  Avg/problem    : ${report.avg_cost_per_problem:.4f}")
        print(f"  Avg repairs    : {report.avg_repair_iterations:.2f}")
        print(f"{'='*60}\n")
