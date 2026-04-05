"""
bridge/mcp_server.py
====================
Andrew MCP Server — Sprint 10.

Exposes Andrew analytical, educational, and swarm-simulation capabilities
as Model Context Protocol (MCP) tools.  Any MCP-compatible client (Claude
Desktop, Cursor, Continue, etc.) can discover and call these tools directly.

Tools exposed:
  andrew_analyze          Analytical query → SQL + Python + narrative
  andrew_educate          Educational question → structured explanation
  andrew_swarm_simulate   AndrewSim MiroFish swarm forecast (Physical AI / RoboQC)
  andrew_get_metrics      Runtime metrics: cost, routing, confidence stats
  andrew_health           Health check for Andrew + Moltis

Run:
  python -m bridge.mcp_server                 (stdio transport, default)
  python -m bridge.mcp_server --transport sse --port 8200

Configure in Claude Desktop (claude_desktop_config.json):
  {
    "mcpServers": {
      "andrew": {
        "command": "python",
        "args": ["-m", "bridge.mcp_server"],
        "cwd": "/path/to/Andrew-Analitic",
        "env": { "DATABASE_URL": "...", "OPENAI_API_KEY": "..." }
      }
    }
  }
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server
from mcp.server.models import InitializationOptions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s | %(levelname)s | %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("andrew_mcp")

# ── Lazy bridge singleton ────────────────────────────────────
_bridge = None


def _get_bridge():
    global _bridge
    if _bridge is None:
        from bridge.service import AndrewMoltisBridge
        _bridge = AndrewMoltisBridge()
        logger.info("AndrewMoltisBridge initialized for MCP server")
    return _bridge


# ── Cost + metrics tracking ──────────────────────────────────
_metrics: Dict[str, Any] = {
    "total_calls": 0,
    "total_cost_usd": 0.0,
    "tool_calls": {},
    "errors": 0,
    "start_time": time.time(),
}


def _track(tool: str, cost: float = 0.0, error: bool = False) -> None:
    _metrics["total_calls"] += 1
    _metrics["total_cost_usd"] += cost
    _metrics["tool_calls"][tool] = _metrics["tool_calls"].get(tool, 0) + 1
    if error:
        _metrics["errors"] += 1


# ============================================================
# MCP Server definition
# ============================================================

server = Server("andrew-analitic")


@server.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name="andrew_analyze",
            description=(
                "Run an analytical query against the connected database using the Andrew Swarm engine. "
                "Generates SQL, executes it, optionally runs Python analysis, and returns a structured narrative. "
                "Use for: data queries, revenue analysis, trend forecasting, aggregations, reports."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language analytical question (e.g. 'total revenue by region last quarter')",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Optional session ID for memory recall context",
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="andrew_educate",
            description=(
                "Get a structured educational explanation from Romeo PhD. "
                "Use for: concept explanations, tutorials, comparisons, mathematical derivations, "
                "'what is X', 'how does Y work', 'difference between A and B'."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Educational question or topic to explain",
                    },
                },
                "required": ["question"],
            },
        ),
        types.Tool(
            name="andrew_swarm_simulate",
            description=(
                "Run AndrewSim — a multi-agent Monte Carlo swarm simulation for Physical AI and RoboQC forecasting. "
                "Feed production data (PatchCore anomaly scores, PLC logs, defect rates, temperatures) and get: "
                "defect probability forecast, risk level, recommended PLC action, and visual evidence references. "
                "Use for: factory QC forecasting, defect prediction, production line risk assessment."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Forecast question (e.g. 'predict defects next 3 shifts at station 4')",
                    },
                    "production_data": {
                        "type": "object",
                        "description": (
                            "Production data dict. Supported keys: "
                            "defect_rate (float 0-1), anomaly_score (float 0-1), "
                            "patchcore_score (float 0-1), temperature (float, Celsius), "
                            "machine_ids (list[str]), image_paths (list[str]), "
                            "forecast_hours (int), description (str), operator_log (str)"
                        ),
                        "additionalProperties": True,
                    },
                    "scenario_count": {
                        "type": "integer",
                        "description": "Number of parallel simulation scenarios (10–10000, default 1000)",
                        "minimum": 10,
                        "maximum": 10000,
                        "default": 1000,
                    },
                    "personality_profiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Agent personality profiles to include. "
                            "Valid values: cautious_operator, qc_expert, fast_worker, "
                            "maintenance_tech, shift_supervisor. Default: all three base profiles."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        types.Tool(
            name="andrew_get_metrics",
            description=(
                "Return Andrew MCP server runtime metrics: total tool calls, total LLM cost (USD), "
                "per-tool call counts, error count, and uptime. Useful for monitoring and cost tracking."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        types.Tool(
            name="andrew_health",
            description=(
                "Check the health of the Andrew stack: Python bridge, Moltis runtime, and database connection. "
                "Returns status for each component."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    logger.info(f"Tool call: {name} args={list(arguments.keys())}")

    if name == "andrew_analyze":
        return await _tool_analyze(arguments)
    if name == "andrew_educate":
        return await _tool_educate(arguments)
    if name == "andrew_swarm_simulate":
        return await _tool_swarm_simulate(arguments)
    if name == "andrew_get_metrics":
        return await _tool_metrics(arguments)
    if name == "andrew_health":
        return await _tool_health(arguments)

    return [types.TextContent(type="text", text=f"Unknown tool: {name}")]


# ============================================================
# Tool implementations
# ============================================================

async def _tool_analyze(args: Dict[str, Any]) -> List[types.TextContent]:
    query = (args.get("query") or "").strip()
    if not query:
        _track("andrew_analyze", error=True)
        return [types.TextContent(type="text", text="Error: `query` is required.")]

    t0 = time.time()
    try:
        bridge = _get_bridge()
        result = await bridge.handle_query(
            query,
            context={"channel": "mcp", "session_id": args.get("session_id")},
        )
        elapsed = round(time.time() - t0, 2)
        cost = result.get("cost_usd", 0.0)
        _track("andrew_analyze", cost=cost)

        output = _format_analyze_result(result, elapsed)
        return [types.TextContent(type="text", text=output)]
    except Exception as exc:
        _track("andrew_analyze", error=True)
        logger.error(f"andrew_analyze failed: {exc}")
        return [types.TextContent(type="text", text=f"Analysis failed: {exc}")]


async def _tool_educate(args: Dict[str, Any]) -> List[types.TextContent]:
    question = (args.get("question") or "").strip()
    if not question:
        _track("andrew_educate", error=True)
        return [types.TextContent(type="text", text="Error: `question` is required.")]

    t0 = time.time()
    try:
        bridge = _get_bridge()
        result = await bridge.handle_query(
            question,
            context={"channel": "mcp"},
        )
        elapsed = round(time.time() - t0, 2)
        cost = result.get("cost_usd", 0.0)
        _track("andrew_educate", cost=cost)

        narrative = result.get("narrative") or result.get("formatted_message", "")
        lines = [
            f"## Educational Response",
            narrative,
            f"\n_Cost: ${cost:.4f} | Time: {elapsed}s_",
        ]
        return [types.TextContent(type="text", text="\n".join(lines))]
    except Exception as exc:
        _track("andrew_educate", error=True)
        logger.error(f"andrew_educate failed: {exc}")
        return [types.TextContent(type="text", text=f"Education query failed: {exc}")]


async def _tool_swarm_simulate(args: Dict[str, Any]) -> List[types.TextContent]:
    query = (args.get("query") or "").strip()
    if not query:
        _track("andrew_swarm_simulate", error=True)
        return [types.TextContent(type="text", text="Error: `query` is required.")]

    production_data = args.get("production_data") or {}
    scenario_count = int(args.get("scenario_count") or 1000)
    scenario_count = max(10, min(10000, scenario_count))
    personality_profiles = args.get("personality_profiles") or []

    t0 = time.time()
    try:
        bridge = _get_bridge()
        result = await bridge.handle_swarm_simulation(
            query=query,
            production_data=production_data,
            scenario_count=scenario_count,
            personality_profiles=personality_profiles,
            context={"channel": "mcp"},
        )
        elapsed = round(time.time() - t0, 2)
        cost = result.get("cost_usd", 0.0)
        _track("andrew_swarm_simulate", cost=cost)

        output = _format_swarm_result(result, elapsed)
        return [types.TextContent(type="text", text=output)]
    except Exception as exc:
        _track("andrew_swarm_simulate", error=True)
        logger.error(f"andrew_swarm_simulate failed: {exc}")
        return [types.TextContent(type="text", text=f"Swarm simulation failed: {exc}")]


async def _tool_metrics(_args: Dict[str, Any]) -> List[types.TextContent]:
    _track("andrew_get_metrics")
    uptime = round(time.time() - _metrics["start_time"], 1)
    data = {
        **_metrics,
        "uptime_seconds": uptime,
        "avg_cost_per_call": (
            round(_metrics["total_cost_usd"] / max(_metrics["total_calls"], 1), 5)
        ),
    }
    return [types.TextContent(type="text", text=json.dumps(data, indent=2))]


async def _tool_health(_args: Dict[str, Any]) -> List[types.TextContent]:
    _track("andrew_health")
    health: Dict[str, Any] = {"bridge": "ok", "moltis": "unknown", "database": "unknown"}
    try:
        bridge = _get_bridge()
        moltis_status = await bridge.moltis.health_check()
        health["moltis"] = moltis_status.get("status", "unknown")
    except Exception as exc:
        health["moltis"] = f"error: {exc}"

    try:
        db_url = os.getenv("DATABASE_URL", "")
        if db_url:
            from sqlalchemy import create_engine, text
            eng = create_engine(db_url)
            with eng.connect() as conn:
                conn.execute(text("SELECT 1"))
            health["database"] = "ok"
        else:
            health["database"] = "no DATABASE_URL configured"
    except Exception as exc:
        health["database"] = f"error: {exc}"

    return [types.TextContent(type="text", text=json.dumps(health, indent=2))]


# ============================================================
# Output formatters
# ============================================================

def _format_analyze_result(result: Dict[str, Any], elapsed: float) -> str:
    lines = []
    success = result.get("success", False)
    confidence = result.get("confidence", 0.0)
    cost = result.get("cost_usd", 0.0)
    routing = result.get("routing", "?")

    if success:
        lines.append(f"## Analysis (confidence: {confidence:.0%})")
        lines.append(result.get("narrative", ""))
        sql = result.get("sql_query")
        if sql:
            lines.append(f"\n**SQL:**\n```sql\n{sql}\n```")
    else:
        lines.append("## Analysis failed")
        lines.append(f"Error: {result.get('error', 'unknown')}")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("\n**Warnings:**")
        for w in warnings[:3]:
            lines.append(f"- {w}")

    lines.append(
        f"\n_Route: {routing} | Cost: ${cost:.4f} | Time: {elapsed}s_"
    )
    return "\n".join(lines)


def _format_swarm_result(result: Dict[str, Any], elapsed: float) -> str:
    risk = result.get("risk_level", "unknown")
    risk_tag = {"low": "[LOW]", "medium": "[MEDIUM]", "high": "[HIGH]", "critical": "[CRITICAL]"}.get(risk, "[?]")
    defect_p = result.get("defect_probability", 0.0)
    action = result.get("recommended_action", "inspect_batch")
    confidence = result.get("confidence", 0.0)
    cost = result.get("cost_usd", 0.0)
    horizon = result.get("forecast_horizon", "?")
    stats = result.get("simulation_stats") or {}

    lines = [
        f"## AndrewSim Forecast {risk_tag}",
        f"",
        f"**Defect Probability:** {defect_p:.1%}",
        f"**Risk Level:** {risk.upper()}",
        f"**Recommended Action:** `{action}`",
        f"**Forecast Horizon:** {horizon}",
        f"",
    ]

    report = result.get("final_report") or ""
    if report:
        lines.append(report[:2000])
        lines.append("")

    evidence = result.get("visual_evidence") or []
    if evidence:
        lines.append("**Visual Evidence:**")
        for path in evidence[:5]:
            lines.append(f"- `{path}`")
        lines.append("")

    if stats:
        lines.append("**Simulation Statistics:**")
        lines.append(f"- Scenarios run: {stats.get('scenario_count', '?')}")
        if "mean_defect_rate" in stats:
            lines.append(f"- Mean defect rate: {stats['mean_defect_rate']:.1%}")
        if "p95_defect_rate" in stats:
            lines.append(f"- p95 defect rate: {stats['p95_defect_rate']:.1%}")
        if "std_defect_rate" in stats:
            lines.append(f"- Std deviation: {stats['std_defect_rate']:.4f}")
        lines.append("")

    warnings = result.get("warnings") or []
    if warnings:
        lines.append("**Warnings:**")
        for w in warnings[:3]:
            lines.append(f"- {w}")
        lines.append("")

    lines.append(
        f"_Confidence: {confidence:.0%} | Cost: ${cost:.4f} | Time: {elapsed}s_"
    )
    return "\n".join(lines)


# ============================================================
# Entry point
# ============================================================

async def _run_stdio():
    logger.info("Andrew MCP Server v1.0.0 starting (stdio)")
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="andrew-analitic",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities={},
                ),
            ),
        )


async def _run_sse(port: int = 8200):
    """SSE transport for network MCP clients."""
    try:
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route, Mount
        import uvicorn

        sse_transport = SseServerTransport("/messages")

        async def handle_sse(scope, receive, send):
            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0],
                    streams[1],
                    InitializationOptions(
                        server_name="andrew-analitic",
                        server_version="1.0.0",
                        capabilities=server.get_capabilities(
                            notification_options=None,
                            experimental_capabilities={},
                        ),
                    ),
                )

        starlette_app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
                Mount("/messages", app=sse_transport.handle_post_message),
            ]
        )
        logger.info(f"Andrew MCP Server v1.0.0 (SSE) on port {port}")
        config = uvicorn.Config(starlette_app, host="0.0.0.0", port=port, log_level="warning")
        uvicorn_server = uvicorn.Server(config)
        await uvicorn_server.serve()
    except ImportError as exc:
        logger.error(f"SSE transport requires starlette + uvicorn: {exc}")
        sys.exit(1)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Andrew MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="Transport mode: stdio (default, for Claude Desktop) or sse (network clients)",
    )
    parser.add_argument("--port", type=int, default=8200, help="Port for SSE transport (default 8200)")
    args = parser.parse_args()

    if args.transport == "sse":
        asyncio.run(_run_sse(port=args.port))
    else:
        asyncio.run(_run_stdio())


if __name__ == "__main__":
    main()
