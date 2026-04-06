"""
bridge/cli.py
=============
Command-line interface for the LTX video pipeline.

Usage
-----
    python -m bridge.cli render  <scenario.md> [options]
    python -m bridge.cli estimate <scenario.md> [options]
    python -m bridge.cli status  <job_id>      [options]

Examples
--------
    python -m bridge.cli estimate demo/pinoblanco_vs_greywald.scenario.md

    python -m bridge.cli render demo/pinoblanco_vs_greywald.scenario.md \\
        --google-key $GOOGLE_API_KEY \\
        --xai-key    $XAI_API_KEY

    # Async: submit and poll later
    python -m bridge.cli render demo/pinoblanco_vs_greywald.scenario.md --async
    python -m bridge.cli status <job_id> --server http://localhost:8100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bridge.cli",
        description="LTX video pipeline — render / estimate / status",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── render ────────────────────────────────────────────────────────────────
    render = sub.add_parser("render", help="Render a scenario (dispatch all scenes)")
    render.add_argument("scenario", help="Path to scenario .md file")
    render.add_argument("--project-id", default=None,
                        help="Project identifier (default: stem of scenario file)")
    render.add_argument("--google-key", default=None,
                        help="Google AI Studio API key [env: GOOGLE_API_KEY]")
    render.add_argument("--xai-key", default=None,
                        help="xAI API key for Grok 4.2 [env: XAI_API_KEY]")
    render.add_argument("--comfyui-host", default="http://127.0.0.1:8188",
                        help="Local ComfyUI URL (default: http://127.0.0.1:8188)")
    render.add_argument("--no-fallback", action="store_true",
                        help="Disable ComfyUI fallback on cloud backend failure")
    render.add_argument("--output", default=None,
                        help="Write JSON result to this file")

    # ── estimate ──────────────────────────────────────────────────────────────
    estimate = sub.add_parser("estimate", help="Estimate cost/time without rendering")
    estimate.add_argument("scenario", help="Path to scenario .md file")
    estimate.add_argument("--project-id", default=None)
    estimate.add_argument("--no-fallback", action="store_true")
    estimate.add_argument("--json", action="store_true",
                          help="Output raw JSON instead of formatted report")

    # ── status ────────────────────────────────────────────────────────────────
    status = sub.add_parser("status", help="Poll an async job from a running bridge server")
    status.add_argument("job_id", help="Job ID returned by --async render")
    status.add_argument("--server", default="http://localhost:8100",
                        help="Bridge server URL (default: http://localhost:8100)")
    status.add_argument("--watch", action="store_true",
                        help="Keep polling every 10 s until job finishes")

    return parser


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_scenario(path: str) -> str:
    p = Path(path)
    if not p.exists():
        print(f"error: scenario file not found: {p}", file=sys.stderr)
        sys.exit(1)
    return p.read_text(encoding="utf-8")


def _project_id(args) -> str:
    if args.project_id:
        return args.project_id
    return Path(args.scenario).stem


# ── subcommand handlers ───────────────────────────────────────────────────────

def _cmd_estimate(args) -> None:
    from bridge.cost_estimator import CostEstimator

    scenario_text = _read_scenario(args.scenario)
    project_id = _project_id(args)
    fallback = not args.no_fallback

    estimator = CostEstimator(fallback_to_comfyui=fallback)
    est = estimator.estimate_scenario(scenario_text, project_id=project_id)

    if getattr(args, "json", False):
        print(json.dumps(est.model_dump(), indent=2))
    else:
        print(f"\n  Scenario : {args.scenario}")
        print(f"  Project  : {project_id}")
        print(f"  Scenes   : {len(est.scenes)}")
        print(estimator.format_report(est))


def _cmd_render(args) -> None:
    from bridge.schemas import LtxGenerationConfig, VideoDispatchRequest
    from bridge.video_dispatcher import dispatch_scenario

    scenario_text = _read_scenario(args.scenario)
    project_id = _project_id(args)

    req = VideoDispatchRequest(
        project_id=project_id,
        scenario_text=scenario_text,
        google_api_key=args.google_key or os.getenv("GOOGLE_API_KEY"),
        xai_api_key=args.xai_key or os.getenv("XAI_API_KEY"),
        comfyui_host=args.comfyui_host,
        fallback_to_comfyui=not args.no_fallback,
    )

    # Print estimate first
    from bridge.cost_estimator import CostEstimator
    estimator = CostEstimator(fallback_to_comfyui=req.fallback_to_comfyui)
    est = estimator.estimate_scenario(scenario_text, project_id=project_id)
    print(f"\n  Scenario : {args.scenario}")
    print(f"  Project  : {project_id}")
    print(f"  Scenes   : {len(est.scenes)}")
    print(estimator.format_report(est))

    scenes_done = 0
    total_scenes = len(est.scenes)

    def _on_scene_done(result):
        nonlocal scenes_done
        scenes_done += 1
        icon = "✓" if result.status.value == "SUCCESS" else "✗"
        print(f"  [{scenes_done}/{total_scenes}] {icon} {result.scene_id}  "
              f"({result.status.value})")

    print("  Dispatching...\n")
    t0 = time.monotonic()

    batch = asyncio.run(_async_render(req, _on_scene_done))

    elapsed = time.monotonic() - t0
    print(f"\n  Done in {elapsed:.1f}s — "
          f"{batch.succeeded} succeeded, {batch.failed} failed\n")

    if args.output:
        out = Path(args.output)
        out.write_text(json.dumps(batch.model_dump(), indent=2), encoding="utf-8")
        print(f"  Results written to {out}\n")

    if batch.failed:
        sys.exit(1)


async def _async_render(req, on_scene_done):
    from bridge.video_dispatcher import dispatch_scenario
    return await dispatch_scenario(req, on_scene_done=on_scene_done)


def _cmd_status(args) -> None:
    import httpx

    url = f"{args.server.rstrip('/')}/video/jobs/{args.job_id}"

    def _fetch():
        try:
            r = httpx.get(url, timeout=10)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPError as e:
            print(f"  error: {e}", file=sys.stderr)
            return None

    def _print_record(rec):
        status = rec.get("status", "?")
        done   = rec.get("scenes_done", 0)
        total  = rec.get("scenes_total", 0)
        print(f"  Job     : {rec['job_id']}")
        print(f"  Status  : {status}")
        print(f"  Progress: {done}/{total} scenes")
        if rec.get("error"):
            print(f"  Error   : {rec['error']}")
        if rec.get("result"):
            r = rec["result"]
            print(f"  Result  : {r.get('succeeded',0)} succeeded, "
                  f"{r.get('failed',0)} failed")

    if args.watch:
        terminal_states = {"done", "error"}
        while True:
            rec = _fetch()
            if rec is None:
                break
            print(f"\r  [{rec.get('status','?')}] "
                  f"{rec.get('scenes_done',0)}/{rec.get('scenes_total',0)} scenes",
                  end="", flush=True)
            if rec.get("status") in terminal_states:
                print()
                _print_record(rec)
                break
            time.sleep(10)
    else:
        rec = _fetch()
        if rec:
            _print_record(rec)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "render":
        _cmd_render(args)
    elif args.command == "estimate":
        _cmd_estimate(args)
    elif args.command == "status":
        _cmd_status(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
