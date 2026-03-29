"""
lcb/prompts.py
==============
Competitive programming prompt strategies for the Andrew LCB pipeline.

Three generation strategies, each targeting a different reasoning path:
  direct          — solve directly, experienced coder mode
  plan_then_code  — force explicit algorithm identification + pseudocode first
  analogy         — recognise a known problem class, then adapt

One self-test generation prompt:
  self_test       — generate edge cases from constraints before solving
"""

from __future__ import annotations

import textwrap
from typing import Dict, List

from .runner import LCBProblem

# ── System prompt ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert competitive programmer ranked in the top 0.1% on Codeforces.
    You write correct, efficient Python 3 solutions that run within time and memory limits.

    Rules:
    - Output ONLY Python 3 code — no markdown fences, no explanations.
    - Read all input from stdin. Write all output to stdout.
    - Handle every edge case (empty input, n=1, max N, negative values, etc.).
    - Choose the most efficient algorithm that fits within the time limit.
    - Use sys.stdin for fast input when N > 10^4.
""")

SELF_TEST_SYSTEM = textwrap.dedent("""\
    You are a test-case engineer for competitive programming.
    Your job is to generate diverse test cases that expose bugs in solutions.
    Focus on: boundary values, edge cases, large inputs, special structures.
    Output ONLY valid JSON — no explanations, no markdown.
""")

# ── Prompt builders ───────────────────────────────────────────

def build_direct_prompt(
    problem: LCBProblem,
    template_context: str = "",
    constraints_summary: str = "",
) -> str:
    """Direct solve: experienced coder picks algorithm implicitly."""
    parts = [f"## {problem.title}\n\n{problem.statement}"]
    if constraints_summary:
        parts.append(f"**Constraints summary:** {constraints_summary}")
    if template_context:
        parts.append(template_context)
    parts.append("Implement the solution in Python 3.")
    return "\n\n".join(parts)


def build_plan_then_code_prompt(
    problem: LCBProblem,
    algorithm_hints: List[str],
    template_context: str = "",
    constraints_summary: str = "",
) -> str:
    """
    Force the model to reason about algorithm before coding.
    Structured as a chain-of-thought in the user message.
    """
    hints_str = ", ".join(algorithm_hints[:4]) if algorithm_hints else "unknown"
    parts = [
        f"## {problem.title}\n\n{problem.statement}",
    ]
    if constraints_summary:
        parts.append(f"**Constraints:** {constraints_summary}")
    if template_context:
        parts.append(template_context)
    parts.append(textwrap.dedent(f"""\
        Solve step by step:
        1. **Algorithm class**: Which algorithm applies? (Hints: {hints_str})
        2. **Complexity**: State time and space complexity of your approach.
        3. **Corner cases**: List edge cases to handle.
        4. **Implementation**: Write the complete Python 3 solution.

        Output ONLY the Python 3 code (step 4). Skip the analysis — it is implicit.
    """))
    return "\n\n".join(parts)


def build_analogy_prompt(
    problem: LCBProblem,
    algorithm_hints: List[str],
    template_context: str = "",
) -> str:
    """
    Ask the model to name a canonical problem this resembles, then adapt.
    Useful when the problem is a disguised classic.
    """
    hints_str = ", ".join(algorithm_hints[:3]) if algorithm_hints else "unknown"
    parts = [
        f"## {problem.title}\n\n{problem.statement}",
    ]
    if template_context:
        parts.append(template_context)
    parts.append(textwrap.dedent(f"""\
        This problem belongs to the following algorithm family: {hints_str}.
        Identify the canonical form of this problem (e.g., "This is coin change with a twist"),
        then implement the adapted solution in Python 3.
        Output ONLY the Python 3 code.
    """))
    return "\n\n".join(parts)


def build_repair_prompt(
    problem: LCBProblem,
    broken_code: str,
    error_message: str,
    failing_input: str = "",
    template_context: str = "",
    time_complexity_target: str = "",
) -> str:
    """
    Repair prompt: receives failing code + error, asks for fix.
    Includes TLE hint if the error suggests timeout.
    """
    is_tle = "TLE" in error_message or "Time Limit" in error_message or "timeout" in error_message.lower()

    parts = [f"## {problem.title}\n\n{problem.statement}"]

    if template_context:
        parts.append(template_context)

    parts.append(f"## Your previous solution (INCORRECT):\n```python\n{broken_code}\n```")

    error_section = f"## Error:\n{error_message}"
    if failing_input:
        error_section += f"\n\n**Failing input:**\n```\n{failing_input}\n```"
    parts.append(error_section)

    if is_tle and time_complexity_target:
        parts.append(
            f"The solution exceeded the time limit. "
            f"You need an algorithm with complexity {time_complexity_target} or better. "
            "Rewrite with a more efficient approach."
        )

    parts.append("Fix the solution. Output ONLY corrected Python 3 code.")
    return "\n\n".join(parts)


def build_self_test_prompt(problem: LCBProblem, constraints: Dict) -> str:
    """
    Ask the model to generate edge-case test inputs + expected outputs.
    Returns a prompt that expects JSON: [{"stdin": "...", "stdout": "..."}]
    """
    n_max = constraints.get("n_max", 0)
    struct = constraints.get("structure", "unknown")

    guidance_parts = []
    if n_max:
        guidance_parts.append(f"- N can be up to {n_max:,}")
    if struct != "unknown":
        guidance_parts.append(f"- Input structure: {struct}")
    guidance_parts += [
        "- Include: minimum input (n=1 or empty), maximum input, all-equal elements,",
        "  negative values (if applicable), already sorted / reverse sorted.",
    ]
    guidance = "\n".join(guidance_parts)

    return textwrap.dedent(f"""\
        ## Problem: {problem.title}

        {problem.statement}

        Generate 6 test cases (stdin + expected stdout) that cover edge cases.
        {guidance}

        Respond with ONLY a JSON array:
        [
          {{"stdin": "<input lines>", "stdout": "<expected output>"}},
          ...
        ]
    """)


# ── Strategy registry ─────────────────────────────────────────

STRATEGIES = {
    "direct": build_direct_prompt,
    "plan_then_code": build_plan_then_code_prompt,
    "analogy": build_analogy_prompt,
}

# How many candidates per strategy per difficulty lane
CANDIDATE_PLAN: Dict[str, List[str]] = {
    # easy: 2 candidates total — both direct (cheap)
    "easy": ["direct", "direct"],
    # hard: 4 candidates — diversity of reasoning paths
    "hard": ["direct", "plan_then_code", "direct", "analogy"],
}
