#!/usr/bin/env python3
"""
scripts/lcb_calibrate.py
========================
Calibration tool for the Andrew LCB Runner.

Runs BEFORE spending API credits to validate:
  1. Routing distribution — does 55/45 split hold on real LCB data?
  2. Constraint extraction quality — n_max, structure detection accuracy
  3. Template retrieval hit rate — how often does RAG return something useful?
  4. Cost projection — expected total cost for N problems
  5. Dry-run solve (optional) — a small sample with real LLM calls

Usage:
    # Full analysis on 200 problems (no API calls)
    python scripts/lcb_calibrate.py --max 200

    # Analysis + dry-run on 10 problems (requires API keys)
    python scripts/lcb_calibrate.py --max 200 --dry-run 10

    # Use local dataset file instead of HuggingFace
    python scripts/lcb_calibrate.py --dataset problems.json --max 200

    # Save JSON report
    python scripts/lcb_calibrate.py --max 500 --out calibration.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lcb.classifier import classify_from_metadata
from lcb.constraints import extract_constraints, infer_algorithm_hints
from lcb.algo_templates import retrieve_templates, _TEMPLATES


# ── Cost model ────────────────────────────────────────────────

# Prices in USD per 1M tokens (as of 2025)
MODEL_PRICES = {
    "gpt-4o-mini":        {"input": 0.15,  "output": 0.60},
    "zhipuai/glm-5.1":    {"input": 0.28,  "output": 1.12},
    "deepseek/deepseek-chat": {"input": 0.27, "output": 1.10},
}

# Typical token counts per call
EASY_TOKENS  = {"input": 900,  "output": 400}  # 2 candidates
HARD_TOKENS  = {"input": 1600, "output": 600}  # per call (4 cands + 3 repair avg)
SELF_TEST_TOKENS = {"input": 800, "output": 300}


def estimate_cost(n_easy: int, n_hard: int,
                  easy_model: str = "gpt-4o-mini",
                  hard_model: str = "zhipuai/glm-5.1",
                  easy_cands: int = 2,
                  hard_cands: int = 4,
                  avg_repairs: float = 1.5) -> Dict[str, float]:
    """Project total cost for a batch run."""
    ep = MODEL_PRICES.get(easy_model, {"input": 0.15, "output": 0.60})
    hp = MODEL_PRICES.get(hard_model, {"input": 0.28, "output": 1.12})

    def cost_call(tokens, prices):
        return (tokens["input"] * prices["input"] + tokens["output"] * prices["output"]) / 1e6

    easy_total = n_easy * easy_cands * cost_call(EASY_TOKENS, ep)
    hard_self_tests = n_hard * cost_call(SELF_TEST_TOKENS, hp)
    hard_candidates = n_hard * hard_cands * cost_call(HARD_TOKENS, hp)
    hard_repairs = n_hard * hard_cands * avg_repairs * cost_call(
        {"input": 1800, "output": 500}, hp
    )

    return {
        "easy_generation":  round(easy_total, 3),
        "hard_self_tests":  round(hard_self_tests, 3),
        "hard_generation":  round(hard_candidates, 3),
        "hard_repairs":     round(hard_repairs, 3),
        "total":            round(easy_total + hard_self_tests + hard_candidates + hard_repairs, 2),
    }


# ── Analysis functions ────────────────────────────────────────

@dataclass
class ProblemAnalysis:
    problem_id: str
    lcb_difficulty: Optional[str]
    predicted_difficulty: str
    classifier_score: int
    classifier_hits: List[str]
    n_max: int
    structure: str
    algorithm_hints: List[str]
    time_complexity: str
    templates_retrieved: List[str]
    template_hit: bool   # at least one template matched


def analyse_problem(problem) -> ProblemAnalysis:
    difficulty, score, hits = classify_from_metadata(
        problem.statement, problem.lcb_difficulty
    )
    constraints = extract_constraints(problem.statement)
    hints, complexity = infer_algorithm_hints(constraints)
    query = problem.statement + " " + " ".join(hints[:5])
    templates = retrieve_templates(query, top_k=2)

    return ProblemAnalysis(
        problem_id=problem.problem_id,
        lcb_difficulty=problem.lcb_difficulty,
        predicted_difficulty=difficulty,
        classifier_score=score,
        classifier_hits=hits[:5],
        n_max=constraints["n_max"],
        structure=constraints["structure"],
        algorithm_hints=hints[:6],
        time_complexity=complexity,
        templates_retrieved=[t.name for t in templates],
        template_hit=len(templates) > 0,
    )


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_calibration(
    problems,
    dry_run_n: int = 0,
    easy_model: str = "gpt-4o-mini",
    hard_model: str = "zhipuai/glm-5.1",
) -> Dict:
    total = len(problems)
    print(f"\n[calibrate] Analysing {total} problems (no LLM calls)…")
    t0 = time.monotonic()

    analyses: List[ProblemAnalysis] = []
    for p in problems:
        analyses.append(analyse_problem(p))

    elapsed = time.monotonic() - t0
    print(f"[calibrate] Done in {elapsed:.2f}s ({total/elapsed:.0f} problems/s)")

    # ── 1. Routing distribution ───────────────────────────────
    print_section("1. ROUTING DISTRIBUTION")
    easy = [a for a in analyses if a.predicted_difficulty == "easy"]
    hard = [a for a in analyses if a.predicted_difficulty == "hard"]
    pct_easy = len(easy) / total * 100
    pct_hard = len(hard) / total * 100
    print(f"  Easy  : {len(easy):4d}  ({pct_easy:.1f}%)  ← target 55%")
    print(f"  Hard  : {len(hard):4d}  ({pct_hard:.1f}%)  ← target 45%")

    # Breakdown by LCB metadata vs predicted
    if any(a.lcb_difficulty for a in analyses):
        meta_counts = Counter(
            (a.lcb_difficulty or "unknown", a.predicted_difficulty)
            for a in analyses
        )
        print("\n  LCB metadata → predicted routing:")
        for (meta, pred), cnt in sorted(meta_counts.items()):
            print(f"    {meta:8s} → {pred:4s} : {cnt:4d}")

    # ── 2. Classifier score distribution ─────────────────────
    print_section("2. CLASSIFIER SCORE DISTRIBUTION")
    scores = [a.classifier_score for a in analyses]
    for threshold in [0, 1, 2, 3, 4, 5, 6, 10]:
        cnt = sum(1 for s in scores if s >= threshold)
        print(f"  score ≥ {threshold:2d} : {cnt:4d}  ({cnt/total*100:.1f}%)")

    print(f"\n  Mean score  : {sum(scores)/len(scores):.2f}")
    print(f"  Score = 0   : {scores.count(0):4d}  (no keyword match → easy)")
    print(f"  Score = 3   : {scores.count(3):4d}  (exact boundary)")

    # ── 3. Top classifier keywords ────────────────────────────
    print_section("3. TOP CLASSIFIER KEYWORDS")
    keyword_counter: Counter = Counter()
    for a in analyses:
        for kw in a.classifier_hits:
            keyword_counter[kw] += 1
    for kw, cnt in keyword_counter.most_common(15):
        bar = "█" * (cnt * 30 // max(keyword_counter.values()))
        print(f"  {kw:30s} {cnt:4d}  {bar}")

    # ── 4. Constraint extraction stats ───────────────────────
    print_section("4. CONSTRAINT EXTRACTION")
    n_max_values = [a.n_max for a in analyses]
    detected = sum(1 for n in n_max_values if n > 0)
    print(f"  n_max detected    : {detected}/{total}  ({detected/total*100:.1f}%)")
    print(f"  n_max = 0         : {n_max_values.count(0):4d}  (not found in statement)")

    buckets = [
        ("≤ 20 (bitmask DP)",      lambda n: 0 < n <= 20),
        ("≤ 300 (O(n^2) DP)",      lambda n: 20 < n <= 300),
        ("≤ 5K (O(n^2) ok)",       lambda n: 300 < n <= 5_000),
        ("≤ 100K (O(n log n))",    lambda n: 5_000 < n <= 100_000),
        ("≤ 1M (O(n log n)/O(n))", lambda n: 100_000 < n <= 1_000_000),
        ("> 1M (O(log n)/math)",   lambda n: n > 1_000_000),
    ]
    print()
    for label, cond in buckets:
        cnt = sum(1 for n in n_max_values if cond(n))
        print(f"  {label:30s} : {cnt:4d}  ({cnt/total*100:.1f}%)")

    struct_counter = Counter(a.structure for a in analyses)
    print("\n  Structure distribution:")
    for struct, cnt in struct_counter.most_common():
        bar = "█" * (cnt * 25 // max(struct_counter.values()))
        print(f"    {struct:10s} {cnt:4d}  {bar}")

    # ── 5. Algorithm hints distribution ──────────────────────
    print_section("5. ALGORITHM HINTS (top-15)")
    hint_counter: Counter = Counter()
    for a in analyses:
        for h in a.algorithm_hints:
            hint_counter[h] += 1
    for hint, cnt in hint_counter.most_common(15):
        bar = "█" * (cnt * 25 // max(hint_counter.values()))
        print(f"  {hint:30s} {cnt:4d}  {bar}")

    # ── 6. RAG template hit rate ──────────────────────────────
    print_section("6. RAG TEMPLATE RETRIEVAL")
    hit_rate = sum(1 for a in analyses if a.template_hit) / total * 100
    print(f"  Hard problems with template match: "
          f"{sum(1 for a in hard if a.template_hit)}/{len(hard)}  "
          f"({sum(1 for a in hard if a.template_hit)/max(len(hard),1)*100:.1f}%)")
    print(f"  All problems hit rate:              {hit_rate:.1f}%")

    tmpl_counter: Counter = Counter()
    for a in analyses:
        for t in a.templates_retrieved:
            tmpl_counter[t] += 1
    print("\n  Most retrieved templates:")
    for tmpl, cnt in tmpl_counter.most_common(10):
        bar = "█" * (cnt * 25 // max(tmpl_counter.values(), default=1))
        print(f"    {tmpl:40s} {cnt:4d}  {bar}")

    unused = {t.name for t in _TEMPLATES} - set(tmpl_counter.keys())
    if unused:
        print(f"\n  Unused templates ({len(unused)}): {', '.join(sorted(unused))}")

    # ── 7. Cost projection ────────────────────────────────────
    print_section("7. COST PROJECTION")
    projection = estimate_cost(len(easy), len(hard), easy_model, hard_model)
    print(f"  Models : easy={easy_model}, hard={hard_model}")
    print(f"  Easy generation    : ${projection['easy_generation']:.2f}")
    print(f"  Hard self-tests    : ${projection['hard_self_tests']:.2f}")
    print(f"  Hard generation    : ${projection['hard_generation']:.2f}")
    print(f"  Hard repair loops  : ${projection['hard_repairs']:.2f}")
    print(f"  ─────────────────────────")
    print(f"  TOTAL {total} problems  : ${projection['total']:.2f}")
    if total < 800:
        scale_factor = 800 / total
        scaled = round(projection['total'] * scale_factor, 2)
        print(f"  Scaled to 800 probs: ${scaled:.2f}  (×{scale_factor:.1f})")

    # ── 8. Dry-run (optional real LLM calls) ─────────────────
    dry_run_results = []
    if dry_run_n > 0:
        print_section(f"8. DRY RUN ({dry_run_n} problems with real LLM calls)")
        try:
            from lcb.runner import LCBRunner
            runner = LCBRunner(easy_candidates=1, hard_candidates=2, max_repair=1)
            sample = problems[:dry_run_n]
            for i, p in enumerate(sample):
                print(f"  [{i+1}/{dry_run_n}] {p.problem_id} ({p.lcb_difficulty or '?'})… ", end="", flush=True)
                result = runner.solve(p)
                status = "✓" if result.success else "✗"
                print(f"{status}  pass={result.passed}/{result.total}  ${result.cost_usd:.4f}")
                dry_run_results.append({
                    "problem_id": result.problem_id,
                    "difficulty": result.difficulty,
                    "success": result.success,
                    "passed": result.passed,
                    "total": result.total,
                    "cost_usd": result.cost_usd,
                })
            total_cost_dry = sum(r["cost_usd"] for r in dry_run_results)
            solved_dry = sum(1 for r in dry_run_results if r["success"])
            print(f"\n  Results: {solved_dry}/{dry_run_n} solved  "
                  f"({solved_dry/dry_run_n*100:.0f}%)  total cost ${total_cost_dry:.4f}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            print("  (Set OPENAI_API_KEY / ZHIPUAI_API_KEY and retry)")

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CALIBRATION SUMMARY  ({total} problems)")
    print(f"{'='*60}")
    print(f"  Routing   : easy {pct_easy:.0f}% / hard {pct_hard:.0f}%  "
          f"(target 55/45)")
    print(f"  n_max detected : {detected/total*100:.0f}%")
    print(f"  RAG hit rate   : {hit_rate:.0f}% (all) / "
          f"{sum(1 for a in hard if a.template_hit)/max(len(hard),1)*100:.0f}% (hard only)")
    print(f"  Projected cost : ${projection['total']:.2f} for {total} problems")
    print(f"{'='*60}\n")

    return {
        "total_problems": total,
        "routing": {"easy": len(easy), "hard": len(hard),
                    "pct_easy": round(pct_easy, 1), "pct_hard": round(pct_hard, 1)},
        "n_max_detected_pct": round(detected / total * 100, 1),
        "rag_hit_rate_all": round(hit_rate, 1),
        "rag_hit_rate_hard": round(
            sum(1 for a in hard if a.template_hit) / max(len(hard), 1) * 100, 1
        ),
        "cost_projection": projection,
        "top_keywords": dict(keyword_counter.most_common(20)),
        "top_hints": dict(hint_counter.most_common(20)),
        "structure_distribution": dict(struct_counter),
        "dry_run_results": dry_run_results,
    }


# ── CLI ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Andrew LCB calibration — pre-flight analysis before API spending"
    )
    parser.add_argument("--dataset", default=None,
                        help="Path to local JSON dataset (default: HuggingFace LCB)")
    parser.add_argument("--split", default="test",
                        help="HuggingFace dataset split (default: test)")
    parser.add_argument("--max", type=int, default=200,
                        help="Maximum problems to analyse (default: 200)")
    parser.add_argument("--dry-run", type=int, default=0, metavar="N",
                        help="Also solve N problems with real LLM calls (default: 0)")
    parser.add_argument("--easy-model", default="gpt-4o-mini")
    parser.add_argument("--hard-model", default="zhipuai/glm-5.1")
    parser.add_argument("--out", default=None,
                        help="Save JSON report to this path")
    args = parser.parse_args()

    # Load problems
    if args.dataset:
        from lcb.eval import _load_local_dataset
        problems = _load_local_dataset(args.dataset)
    else:
        print(f"[calibrate] Loading up to {args.max} problems from HuggingFace…")
        try:
            from lcb.eval import _load_hf_dataset
            problems = _load_hf_dataset(args.split, max_problems=args.max)
        except ImportError:
            print("ERROR: Install HuggingFace datasets: pip install datasets")
            sys.exit(1)

    if args.max:
        problems = problems[:args.max]

    # Run calibration
    report = run_calibration(
        problems,
        dry_run_n=args.dry_run,
        easy_model=args.easy_model,
        hard_model=args.hard_model,
    )

    # Save if requested
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[calibrate] Report saved to {args.out}")


if __name__ == "__main__":
    main()
