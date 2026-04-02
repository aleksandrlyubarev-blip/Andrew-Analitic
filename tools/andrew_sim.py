#!/usr/bin/env python3
"""
AndrewSim CLI — MiroFish swarm simulator for Physical AI / RoboQC.

Usage:
    python -m tools.andrew_sim --query "predict defects next 3 shifts" --data ./evidence.json
    python -m tools.andrew_sim --query "..." --data '{"anomaly_score": 0.87, "temperature": 42.1}'
    python -m tools.andrew_sim --query "..." --scenarios 5000

Options:
    --query       TEXT       Forecast question (required)
    --data        PATH|JSON  Production data: file path or inline JSON
    --scenarios   INT        Number of simulation scenarios [default: 1000]
    --profiles    TEXT       Comma-separated personality profiles
    --output      FORMAT     Output format: text|json [default: text]
    --save        PATH       Save simulation result as JSON file
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="AndrewSim — MiroFish swarm forecaster for RoboQC / Physical AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--query", "-q", required=True, help="Forecast question")
    parser.add_argument("--data", "-d", default="{}", help="Production data: file path or inline JSON string")
    parser.add_argument("--scenarios", "-n", type=int, default=1000, help="Number of simulation scenarios")
    parser.add_argument("--profiles", "-p", default="", help="Comma-separated personality profiles")
    parser.add_argument("--output", "-o", choices=["text", "json"], default="text")
    parser.add_argument("--save", "-s", default=None, help="Save result to JSON file")
    args = parser.parse_args()

    # Load production data
    production_data = {}
    if args.data and args.data.strip() != "{}":
        p = Path(args.data)
        if p.exists():
            with open(p) as f:
                production_data = json.load(f)
        else:
            try:
                production_data = json.loads(args.data)
            except json.JSONDecodeError:
                print(f"[ERROR] --data must be a valid JSON file path or JSON string", file=sys.stderr)
                sys.exit(1)

    profiles = [p.strip() for p in args.profiles.split(",") if p.strip()] if args.profiles else []

    from core.mirofish_swarm import MiroFishExecutor

    print("=" * 60, file=sys.stderr)
    print("AndrewSim v1.0.0 — MiroFish Swarm Forecaster", file=sys.stderr)
    print(f"Query    : {args.query}", file=sys.stderr)
    print(f"Scenarios: {args.scenarios}", file=sys.stderr)
    print(f"Profiles : {profiles or 'default'}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    t0 = time.time()
    executor = MiroFishExecutor()
    result = executor.execute(
        query=args.query,
        production_data=production_data,
        scenario_count=args.scenarios,
        personality_profiles=profiles,
    )
    elapsed = time.time() - t0

    if args.output == "json":
        out = {
            "query": args.query,
            "defect_probability": result.defect_probability,
            "recommended_action": result.recommended_action,
            "risk_level": result.risk_level,
            "forecast_horizon": result.forecast_horizon,
            "confidence": result.confidence,
            "cost_usd": result.cost_usd,
            "success": result.success,
            "elapsed_seconds": round(elapsed, 2),
            "final_report": result.output,
            "simulation_stats": result.simulation_stats,
            "visual_evidence": result.visual_evidence,
            "warnings": result.warnings,
            "error": result.error,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        print(result.to_roma_output())
        print(f"\n[Elapsed: {elapsed:.1f}s | Cost: ${result.cost_usd:.4f}]")

    if args.save:
        save_path = Path(args.save)
        out = {
            "query": args.query,
            "defect_probability": result.defect_probability,
            "recommended_action": result.recommended_action,
            "risk_level": result.risk_level,
            "forecast_horizon": result.forecast_horizon,
            "confidence": result.confidence,
            "final_report": result.output,
            "simulation_stats": result.simulation_stats,
            "warnings": result.warnings,
        }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"[Saved to {save_path}]", file=sys.stderr)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
