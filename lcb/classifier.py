"""
lcb/classifier.py
=================
Difficulty classifier for LiveCodeBench competitive programming problems.

Routes problems into two model lanes based on keyword analysis:
  easy  → GPT-4o-mini (implementation, simulation, basic math)
  hard  → DeepSeek V3 (DP, graphs, advanced DS, number theory, geometry)

Design mirrors core/routing.py: pure keyword scoring, zero LLM calls,
deterministic and offline-testable.
"""

import re
from typing import Dict, List, Tuple

# ── Hard-problem keyword table ────────────────────────────────
# Weight 1 = moderate complexity indicator
# Weight 2 = strong complexity indicator
# Weight 3 = heavy/advanced algorithm indicator

HARD_KEYWORDS: Dict[str, int] = {
    # Dynamic programming
    "dynamic programming": 3, "dp": 2, "memoization": 2, "bottom-up": 2,
    "top-down": 2, "knapsack": 3, "longest common subsequence": 3,
    "longest increasing subsequence": 3, "edit distance": 3,
    "bitmask dp": 3, "digit dp": 3, "interval dp": 3, "tree dp": 3,

    # Graph algorithms
    "graph": 2, "shortest path": 3, "dijkstra": 3, "bellman-ford": 3,
    "floyd": 3, "bfs": 1, "dfs": 1, "topological sort": 2, "cycle detection": 2,
    "strongly connected": 3, "bipartite": 2, "minimum spanning tree": 3,
    "network flow": 3, "max flow": 3, "min cut": 3, "eulerian": 3,
    "hamiltonian": 3, "tarjan": 3, "kosaraju": 3,

    # Advanced data structures
    "segment tree": 3, "fenwick tree": 3, "bit": 2, "binary indexed tree": 3,
    "sparse table": 3, "suffix array": 3, "suffix automaton": 3,
    "trie": 2, "treap": 3, "splay tree": 3, "link cut tree": 3,
    "sqrt decomposition": 3, "sqrt": 2, "heavy light decomposition": 3,
    "centroid decomposition": 3, "persistent": 3,

    # Number theory / combinatorics
    "modular arithmetic": 2, "modular inverse": 3, "fermat": 3,
    "chinese remainder theorem": 3, "crt": 2, "euler's totient": 3,
    "prime factorization": 2, "sieve": 2, "miller-rabin": 3,
    "extended euclidean": 3, "matrix exponentiation": 3, "fast exponentiation": 2,
    "combinatorics": 2, "permutation": 1, "combination": 1, "ntt": 3,
    "fast fourier transform": 3, "fft": 3, "polynomial": 2,

    # Geometry
    "convex hull": 3, "computational geometry": 3, "line intersection": 3,
    "polygon": 2, "voronoi": 3, "delaunay": 3, "geometry": 2,

    # String algorithms
    "kmp": 3, "z-function": 3, "aho-corasick": 3, "manacher": 3,
    "rolling hash": 2, "rabin-karp": 3, "palindrome": 1,

    # Other advanced
    "two pointers": 1, "sliding window": 1, "monotonic stack": 2,
    "monotonic deque": 2, "binary search on answer": 2, "meet in the middle": 3,
    "divide and conquer": 2, "game theory": 3, "sprague grundy": 3,
    "probability": 2, "expected value": 2, "interactive": 2,
}

# Keywords that strongly suggest easy implementation problems
EASY_KEYWORDS = {
    "simulate", "simulation", "brute force", "iterate", "increment",
    "decrement", "traverse", "scan", "check each", "for each",
    "simple", "straightforward", "basic", "find the", "count the",
    "output the", "print the", "single pass", "one pass",
}

# Threshold: total weight >= this → hard
HARD_THRESHOLD = 3


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def classify_difficulty(problem_statement: str) -> Tuple[str, int, List[str]]:
    """
    Classify a competitive programming problem as 'easy' or 'hard'.

    Returns:
        (difficulty, score, matched_keywords)
        difficulty: 'easy' | 'hard'
        score: cumulative keyword weight
        matched_keywords: list of matched terms
    """
    text = _normalize(problem_statement)
    padded = f" {text} "

    hits: List[Tuple[str, int]] = []
    for kw, weight in HARD_KEYWORDS.items():
        if f" {kw} " in padded or kw in text:
            hits.append((kw, weight))

    score = sum(w for _, w in hits)
    matched = [kw for kw, _ in hits]

    # Easy override: explicit easy signals and low score
    has_easy_signal = any(t in text for t in EASY_KEYWORDS)
    if has_easy_signal and score < 2:
        return "easy", score, matched

    difficulty = "hard" if score >= HARD_THRESHOLD else "easy"
    return difficulty, score, matched


def classify_from_metadata(
    problem_statement: str,
    lcb_difficulty: str | None = None,
) -> Tuple[str, int, List[str]]:
    """
    Classify using LCB metadata difficulty if available, falling back to
    keyword analysis.

    LCB difficulty levels: 'easy' → easy, 'medium'/'hard' → hard.
    """
    if lcb_difficulty:
        lvl = lcb_difficulty.lower().strip()
        if lvl == "easy":
            return "easy", 0, ["metadata:easy"]
        if lvl in ("medium", "hard"):
            return "hard", 0, [f"metadata:{lvl}"]

    return classify_difficulty(problem_statement)
