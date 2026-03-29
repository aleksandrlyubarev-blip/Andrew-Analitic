"""
lcb/runner.py
=============
Andrew LCB Runner — difficulty-routed competitive programming solver.

Routing:
  easy (55%)  → GPT-4o-mini   1-2 candidates, no repair
  hard (45%)  → GLM 5.1       4 candidates + 3 repair iterations + RAG templates

GLM 5.1 (Z.ai / ZhipuAI):
  - 744B total params, ~40B active per token (sparse MoE, 256 experts)
  - 204K context, trained with SLIME async RL for agentic tasks
  - Interleaved Thinking = natural fit for repair loop pattern
  - MIT licence, zero vendor lock-in

Cost model (approximate):
  easy problem: ~$0.001  (2 candidates × GPT-4o-mini)
  hard problem: ~$0.02   (4 candidates + up to 12 repair calls × GLM 5.1)
  800-problem run: ~$18-20 total

Sandbox backends:
  subprocess (default) — local process, no dependencies
  moltis               — ZeroClaw/Moltis Docker sandbox (set LCB_SANDBOX_BACKEND=moltis)
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

try:
    import litellm
    from litellm import completion, completion_cost
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False

from .classifier import classify_from_metadata
from .algo_templates import retrieve_templates, format_templates_for_prompt

logger = logging.getLogger("lcb.runner")

# ── Model registry ────────────────────────────────────────────

LCB_MODEL_EASY = os.getenv("LCB_MODEL_EASY", "gpt-4o-mini")
# GLM 5.1: 744B MoE, ~40B active, SLIME-trained for agentic/coding tasks.
# Fallback: deepseek/deepseek-chat if ZhipuAI key not set.
LCB_MODEL_HARD = os.getenv("LCB_MODEL_HARD", "zhipuai/glm-5.1")

EASY_CANDIDATES = int(os.getenv("LCB_EASY_CANDIDATES", "2"))
HARD_CANDIDATES = int(os.getenv("LCB_HARD_CANDIDATES", "4"))
MAX_REPAIR_ITERATIONS = int(os.getenv("LCB_MAX_REPAIR", "3"))
SANDBOX_TIMEOUT = int(os.getenv("LCB_SANDBOX_TIMEOUT", "5"))  # seconds per test

# "subprocess" (default, local) | "moltis" (ZeroClaw Docker sandbox)
SANDBOX_BACKEND = os.getenv("LCB_SANDBOX_BACKEND", "subprocess")

# Temperature for candidate generation (diverse sampling)
EASY_TEMPERATURE = 0.2
HARD_TEMPERATURE = 0.7

# ── Data types ────────────────────────────────────────────────


@dataclass
class ProblemTestCase:
    stdin: str
    expected_stdout: str

# Backwards-compatible alias
TestCase = ProblemTestCase


@dataclass
class LCBProblem:
    """A single LiveCodeBench problem."""
    problem_id: str
    title: str
    statement: str
    test_cases: List[ProblemTestCase] = field(default_factory=list)
    lcb_difficulty: Optional[str] = None   # 'easy'|'medium'|'hard' from LCB metadata
    time_limit_sec: float = 2.0
    memory_limit_mb: int = 256


@dataclass
class SolveAttempt:
    code: str
    passed: int        # test cases passed
    total: int         # total test cases
    error: Optional[str] = None
    cost_usd: float = 0.0
    repair_iterations: int = 0


@dataclass
class LCBResult:
    problem_id: str
    difficulty: str          # 'easy' | 'hard'
    model: str
    best_code: str
    passed: int
    total: int
    success: bool            # all test cases passed
    cost_usd: float
    elapsed_sec: float
    candidates_tried: int
    repair_iterations: int
    classifier_score: int
    classifier_hits: List[str]
    error: Optional[str] = None


# ── Sandbox execution ─────────────────────────────────────────


def _run_code(code: str, stdin: str, timeout: float) -> Tuple[str, Optional[str]]:
    """
    Execute Python code in a subprocess with stdin/timeout.
    Returns (stdout, error_message_or_None).
    """
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        fname = f.name

    try:
        proc = subprocess.run(
            [sys.executable, fname],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            return "", proc.stderr.strip()
        return proc.stdout.strip(), None
    except subprocess.TimeoutExpired:
        return "", f"TLE: exceeded {timeout}s"
    except Exception as exc:
        return "", str(exc)
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


def _run_code_moltis(code: str, stdin: str, timeout: float) -> Tuple[str, Optional[str]]:
    """
    Execute code via Moltis/ZeroClaw Docker sandbox.
    Bridges async Moltis client into sync pipeline with asyncio.run().
    Falls back to subprocess if Moltis is unreachable.

    Moltis = ZeroClaw equivalent in the Andrew stack:
      - Rust binary ~5 MB, <10 ms startup
      - Strict allowlist sandboxing (workspace scoping)
      - Lifecycle hooks intercept dangerous commands before shell execution
    """
    try:
        import asyncio
        # Import here to avoid circular import and hard dependency
        from bridge.client import MoltisClient

        async def _exec() -> Tuple[str, Optional[str]]:
            client = MoltisClient()
            try:
                # Inject stdin by prepending sys.stdin override
                # This is safe because we control the code string entirely
                injected = (
                    "import sys as _sys, io as _io\n"
                    f"_sys.stdin = _io.StringIO({stdin!r})\n"
                    + code
                )
                result = await client.execute_in_sandbox(injected, language="python")
                out = (result.get("output") or "").strip()
                err = result.get("error") or None
                if result.get("exitCode", 0) != 0 and not err:
                    err = f"Exit code {result.get('exitCode')}"
                return out, err
            finally:
                await client.close()

        return asyncio.run(_exec())
    except Exception as exc:
        logger.warning("Moltis sandbox failed (%s), falling back to subprocess", exc)
        return _run_code(code, stdin, timeout)


def _dispatch_run(code: str, stdin: str, timeout: float) -> Tuple[str, Optional[str]]:
    """Route to the configured sandbox backend."""
    if SANDBOX_BACKEND == "moltis":
        return _run_code_moltis(code, stdin, timeout)
    return _run_code(code, stdin, timeout)


def evaluate_solution(
    code: str,
    test_cases: List[TestCase],
    timeout: float = SANDBOX_TIMEOUT,
) -> Tuple[int, int, Optional[str]]:
    """
    Run code against all test cases.
    Returns (passed, total, first_error_or_None).
    """
    if not test_cases:
        return 0, 0, None

    passed = 0
    first_error: Optional[str] = None

    for tc in test_cases:
        actual, err = _dispatch_run(code, tc.stdin, timeout)
        if err:
            if first_error is None:
                first_error = err
            continue
        if actual == tc.expected_stdout.strip():
            passed += 1
        elif first_error is None:
            first_error = f"Wrong answer: got {actual!r}, expected {tc.expected_stdout.strip()!r}"

    return passed, len(test_cases), first_error


# ── Prompt builders ───────────────────────────────────────────

_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert competitive programmer. Solve the given problem with a complete,
    correct Python solution. Output ONLY the Python code — no explanation, no markdown
    fences, no imports beyond what is needed. The code must:
    - Read all input from stdin
    - Write all output to stdout
    - Handle all edge cases
    - Be efficient enough to pass within the time limit
""")


def _build_generation_prompt(
    problem: LCBProblem,
    rag_context: str = "",
) -> str:
    parts = [f"## Problem: {problem.title}\n\n{problem.statement}"]
    if rag_context:
        parts.append(f"\n{rag_context}")
    return "\n\n".join(parts)


def _build_repair_prompt(
    problem: LCBProblem,
    broken_code: str,
    error: str,
    rag_context: str = "",
) -> str:
    parts = [
        f"## Problem: {problem.title}\n\n{problem.statement}",
        f"## Your previous (incorrect) solution:\n```python\n{broken_code}\n```",
        f"## Error / failure:\n{error}",
        "## Fix the solution. Output ONLY corrected Python code.",
    ]
    if rag_context:
        parts.insert(1, rag_context)
    return "\n\n".join(parts)


# ── LLM call wrapper ──────────────────────────────────────────

def _llm_call(
    model: str,
    prompt: str,
    temperature: float,
) -> Tuple[str, float]:
    """
    Single LLM call via LiteLLM.
    Returns (generated_code, cost_usd).
    Raises on hard errors; caller handles retries.
    """
    if not _LITELLM_AVAILABLE:
        raise RuntimeError("litellm not installed — run: pip install litellm")

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    resp = completion(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=2048,
    )

    code = resp.choices[0].message.content or ""
    # Strip markdown fences if the model adds them despite instructions
    if code.startswith("```"):
        lines = code.splitlines()
        code = "\n".join(
            l for l in lines
            if not l.startswith("```")
        ).strip()

    try:
        cost = completion_cost(completion_response=resp)
    except Exception:
        cost = 0.0

    return code, cost


# ── Core solving logic ────────────────────────────────────────


def _solve_easy(
    problem: LCBProblem,
    n_candidates: int = EASY_CANDIDATES,
) -> Tuple[SolveAttempt, float]:
    """
    Easy lane: GPT-4o-mini, 1-2 candidates, no repair.
    Returns (best_attempt, total_cost).
    """
    model = LCB_MODEL_EASY
    prompt = _build_generation_prompt(problem)
    total_cost = 0.0
    best: Optional[SolveAttempt] = None

    for _ in range(n_candidates):
        try:
            code, cost = _llm_call(model, prompt, EASY_TEMPERATURE)
        except Exception as exc:
            logger.warning("Easy LLM call failed: %s", exc)
            continue
        total_cost += cost

        passed, total, err = evaluate_solution(code, problem.test_cases)
        attempt = SolveAttempt(code=code, passed=passed, total=total, error=err, cost_usd=cost)

        if best is None or passed > best.passed:
            best = attempt
        if best.passed == best.total and best.total > 0:
            break   # perfect solution found early

    if best is None:
        best = SolveAttempt(code="", passed=0, total=len(problem.test_cases),
                            error="All LLM calls failed")
    return best, total_cost


def _repair_loop(
    problem: LCBProblem,
    initial_code: str,
    initial_error: Optional[str],
    initial_passed: int,
    model: str,
    rag_context: str,
    max_iters: int = MAX_REPAIR_ITERATIONS,
) -> Tuple[str, int, int, Optional[str], float, int]:
    """
    Iteratively repair a failing solution.
    Returns (best_code, passed, total, error, cost, repair_count).
    """
    code = initial_code
    error = initial_error
    passed = initial_passed
    total = len(problem.test_cases)
    total_cost = 0.0
    repair_count = 0

    for _ in range(max_iters):
        if passed == total and total > 0:
            break
        if not error:
            break

        prompt = _build_repair_prompt(problem, code, error, rag_context)
        try:
            new_code, cost = _llm_call(model, prompt, temperature=0.4)
        except Exception as exc:
            logger.warning("Repair LLM call failed: %s", exc)
            break
        total_cost += cost
        repair_count += 1

        new_passed, new_total, new_error = evaluate_solution(new_code, problem.test_cases)
        if new_passed >= passed:   # accept if not worse
            code = new_code
            error = new_error
            passed = new_passed
            total = new_total

    return code, passed, total, error, total_cost, repair_count


def _solve_hard(
    problem: LCBProblem,
    n_candidates: int = HARD_CANDIDATES,
    max_repair: int = MAX_REPAIR_ITERATIONS,
) -> Tuple[SolveAttempt, float]:
    """
    Hard lane: DeepSeek V3, 4 candidates, 3 repair iterations each, + RAG templates.
    Returns (best_attempt, total_cost).
    """
    model = LCB_MODEL_HARD
    templates = retrieve_templates(problem.statement, top_k=2)
    rag_context = format_templates_for_prompt(templates)
    prompt = _build_generation_prompt(problem, rag_context)
    total_cost = 0.0
    best: Optional[SolveAttempt] = None

    for _ in range(n_candidates):
        try:
            code, cost = _llm_call(model, prompt, HARD_TEMPERATURE)
        except Exception as exc:
            logger.warning("Hard LLM call failed: %s", exc)
            continue
        total_cost += cost

        passed, total, err = evaluate_solution(code, problem.test_cases)

        # Repair loop for this candidate
        if err and passed < (total or 1):
            fixed_code, fixed_passed, fixed_total, fixed_err, repair_cost, n_repairs = (
                _repair_loop(problem, code, err, passed, model, rag_context, max_repair)
            )
            total_cost += repair_cost
            attempt = SolveAttempt(
                code=fixed_code,
                passed=fixed_passed,
                total=fixed_total,
                error=fixed_err,
                cost_usd=cost + repair_cost,
                repair_iterations=n_repairs,
            )
        else:
            attempt = SolveAttempt(
                code=code, passed=passed, total=total, error=err, cost_usd=cost
            )

        if best is None or attempt.passed > best.passed:
            best = attempt
        if best.passed == best.total and best.total > 0:
            break   # perfect solution found

    if best is None:
        best = SolveAttempt(code="", passed=0, total=len(problem.test_cases),
                            error="All LLM calls failed")
    return best, total_cost


# ── Public API ────────────────────────────────────────────────


class LCBRunner:
    """
    Main runner entry point.

    Examples:
        runner = LCBRunner()
        result = runner.solve(problem)
        print(result.success, result.cost_usd)

        # Batch
        results = runner.solve_batch(problems)
        summary = runner.summarise(results)
    """

    def __init__(
        self,
        easy_candidates: int = EASY_CANDIDATES,
        hard_candidates: int = HARD_CANDIDATES,
        max_repair: int = MAX_REPAIR_ITERATIONS,
    ):
        self.easy_candidates = easy_candidates
        self.hard_candidates = hard_candidates
        self.max_repair = max_repair

    def solve(self, problem: LCBProblem) -> LCBResult:
        """Solve a single problem with automatic difficulty routing."""
        t0 = time.monotonic()

        difficulty, score, hits = classify_from_metadata(
            problem.statement, problem.lcb_difficulty
        )
        logger.info(
            "Problem %s → %s (score=%d, hits=%s)",
            problem.problem_id, difficulty, score, hits
        )

        if difficulty == "easy":
            best, total_cost = _solve_easy(problem, self.easy_candidates)
            model = LCB_MODEL_EASY
        else:
            best, total_cost = _solve_hard(problem, self.hard_candidates, self.max_repair)
            model = LCB_MODEL_HARD

        elapsed = time.monotonic() - t0
        success = (best.total > 0 and best.passed == best.total)

        return LCBResult(
            problem_id=problem.problem_id,
            difficulty=difficulty,
            model=model,
            best_code=best.code,
            passed=best.passed,
            total=best.total,
            success=success,
            cost_usd=total_cost,
            elapsed_sec=elapsed,
            candidates_tried=self.easy_candidates if difficulty == "easy" else self.hard_candidates,
            repair_iterations=best.repair_iterations,
            classifier_score=score,
            classifier_hits=hits,
            error=best.error,
        )

    def solve_batch(
        self,
        problems: List[LCBProblem],
        on_result: Any = None,
    ) -> List[LCBResult]:
        """
        Solve a list of problems sequentially.
        Calls on_result(result) after each solve if provided.
        """
        results: List[LCBResult] = []
        for i, problem in enumerate(problems):
            logger.info("Solving %d/%d: %s", i + 1, len(problems), problem.problem_id)
            result = self.solve(problem)
            results.append(result)
            if on_result is not None:
                on_result(result)
        return results

    @staticmethod
    def summarise(results: List[LCBResult]) -> Dict[str, Any]:
        """Aggregate statistics over a batch of results."""
        if not results:
            return {}

        total = len(results)
        solved = sum(1 for r in results if r.success)
        easy = [r for r in results if r.difficulty == "easy"]
        hard = [r for r in results if r.difficulty == "hard"]
        total_cost = sum(r.cost_usd for r in results)
        avg_repair = (
            sum(r.repair_iterations for r in hard) / len(hard) if hard else 0.0
        )

        return {
            "total_problems": total,
            "solved": solved,
            "accuracy": solved / total,
            "easy_count": len(easy),
            "easy_solved": sum(1 for r in easy if r.success),
            "easy_accuracy": sum(1 for r in easy if r.success) / len(easy) if easy else 0,
            "hard_count": len(hard),
            "hard_solved": sum(1 for r in hard if r.success),
            "hard_accuracy": sum(1 for r in hard if r.success) / len(hard) if hard else 0,
            "total_cost_usd": total_cost,
            "avg_cost_per_problem": total_cost / total,
            "avg_repair_iterations_hard": avg_repair,
        }
