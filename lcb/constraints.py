"""
lcb/constraints.py
==================
Regex-based constraint extraction from competitive programming problem statements.

Extracts:
  n_max, m_max, value_range  — from "1 ≤ N ≤ 10^5" style lines
  structure                  — array / graph / string / tree / grid
  is_weighted, is_directed   — graph properties

Maps constraints to:
  algorithm_hints            — ['dp', 'bfs', 'segment_tree', ...]
  time_complexity_target     — 'O(n)', 'O(n log n)', 'O(n^2)'
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

# ── Constraint value parsing ──────────────────────────────────

_POWER_RE = re.compile(r"10\s*\^?\s*(\d+)")   # 10^5, 10 ^ 5
_MULT_RE  = re.compile(r"(\d+)\s*[×x\*]\s*10\s*\^?\s*(\d+)")  # 2×10^5


def _parse_bound(token: str) -> int:
    """Parse a bound token like '10^5', '200000', '1e5', '2×10^5' → int."""
    token = token.strip().replace(",", "")
    m = _MULT_RE.search(token)
    if m:
        return int(m.group(1)) * (10 ** int(m.group(2)))
    m = _POWER_RE.search(token)
    if m:
        return 10 ** int(m.group(1))
    try:
        return int(float(token))
    except ValueError:
        return 0


# Pattern: "1 ≤ N ≤ 10^5" or "N, M ≤ 2×10^5" or "n ≤ 10^9"
_BOUND_PATTERN = re.compile(
    r"(?:[1-9]\d*\s*[≤<=]\s*)?"        # optional lower bound
    r"([nNmM])\s*[≤<=,]\s*"            # variable name
    r"([\d\s\^\*×x\.eE]+)",            # upper bound value
    re.IGNORECASE,
)

# Explicit "up to X" style
_UPTO_PATTERN = re.compile(
    r"up\s*to\s+([\d\s\^\*×x\.eE]+)",
    re.IGNORECASE,
)

_VALUE_BOUND_PATTERN = re.compile(
    r"(?:[−\-]?\d+\s*[≤<=]\s*)?[a-zA-Z_]\w*\s*[≤<=]\s*([\d\s\^\*×x\.eE]+)",
    re.IGNORECASE,
)


def extract_constraints(statement: str) -> Dict[str, Any]:
    """
    Parse a problem statement and return a constraints dict.

    Returns:
      {
        n_max: int,           # largest N found (0 if not detected)
        m_max: int,           # largest M found (0 if not detected)
        value_range: int,     # largest value bound (0 if not detected)
        structure: str,       # 'array'|'graph'|'string'|'tree'|'grid'|'unknown'
        is_weighted: bool,
        is_directed: bool,
        is_interactive: bool,
      }
    """
    text = statement.lower()
    result: Dict[str, Any] = {
        "n_max": 0,
        "m_max": 0,
        "value_range": 0,
        "structure": "unknown",
        "is_weighted": False,
        "is_directed": False,
        "is_interactive": False,
    }

    # --- Extract N and M bounds ---
    for m in _BOUND_PATTERN.finditer(statement):
        var = m.group(1).lower()
        val = _parse_bound(m.group(2))
        if var == "n" and val > result["n_max"]:
            result["n_max"] = val
        elif var == "m" and val > result["m_max"]:
            result["m_max"] = val

    # Fallback: "up to X"
    for m in _UPTO_PATTERN.finditer(statement):
        val = _parse_bound(m.group(1))
        if val > result["n_max"]:
            result["n_max"] = val

    # --- Structure detection ---
    structure_signals = [
        ("grid",   ["grid", "matrix", "row", "column", "cell", "2d array"]),
        ("tree",   ["tree", "rooted tree", "binary tree", "parent", "children"]),
        ("graph",  ["graph", "node", "edge", "vertex", "vertices", "adjacent"]),
        ("string", ["string", "substring", "character", "palindrome", "lexicograph"]),
        ("array",  ["array", "sequence", "list", "elements", "subarray"]),
    ]
    for struct, signals in structure_signals:
        if any(s in text for s in signals):
            result["structure"] = struct
            break

    # --- Graph properties ---
    if result["structure"] in ("graph", "tree"):
        result["is_weighted"] = any(w in text for w in ["weight", "cost", "distance", "capacity"])
        result["is_directed"] = any(w in text for w in ["directed", "direction", "dag", "topological"])

    # --- Interactive ---
    result["is_interactive"] = "interactive" in text or "query" in text and "answer" in text

    return result


# ── Constraint → algorithm hints ─────────────────────────────

def infer_algorithm_hints(constraints: Dict[str, Any]) -> Tuple[List[str], str]:
    """
    Map extracted constraints to algorithm hints and time complexity target.

    Returns:
      (algorithm_hints, time_complexity_target)
    """
    n = constraints.get("n_max", 0)
    struct = constraints.get("structure", "unknown")
    weighted = constraints.get("is_weighted", False)
    directed = constraints.get("is_directed", False)
    hints: List[str] = []

    # --- Time complexity from N ---
    if n == 0 or n <= 20:
        complexity = "O(2^n) or O(n!)"
        hints += ["bitmask_dp", "backtracking", "brute_force"]
    elif n <= 300:
        complexity = "O(n^2) or O(n^3)"
        hints += ["dp_2d", "floyd_warshall", "interval_dp"]
    elif n <= 5_000:
        complexity = "O(n^2)"
        hints += ["dp_1d", "two_pointers", "brute_force"]
    elif n <= 100_000:
        complexity = "O(n log n)"
        hints += ["binary_search", "merge_sort", "segment_tree", "bfs", "dp_1d"]
    elif n <= 1_000_000:
        complexity = "O(n log n) or O(n)"
        hints += ["linear_dp", "two_pointers", "sliding_window", "prefix_sum"]
    else:
        complexity = "O(log n) or O(1)"
        hints += ["math", "binary_search_answer", "number_theory"]

    # --- Structure-specific hints ---
    if struct == "graph":
        if weighted:
            hints += ["dijkstra", "bellman_ford"]
        else:
            hints += ["bfs", "dfs"]
        if directed:
            hints += ["topological_sort", "dag_dp"]
        hints += ["dsu"]

    elif struct == "tree":
        hints += ["tree_dp", "dfs", "lca"]
        if n > 100_000:
            hints += ["heavy_light_decomposition", "centroid_decomposition"]

    elif struct == "string":
        hints += ["two_pointers", "hashing"]
        if n > 5_000:
            hints += ["kmp", "z_function", "suffix_array"]

    elif struct == "grid":
        hints += ["bfs", "dp_2d"]

    elif struct == "array":
        hints += ["prefix_sum", "binary_search", "two_pointers", "monotonic_stack"]

    # Deduplicate while preserving order
    seen = set()
    unique_hints = []
    for h in hints:
        if h not in seen:
            seen.add(h)
            unique_hints.append(h)

    return unique_hints, complexity
