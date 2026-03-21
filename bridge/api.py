"""
bridge/api.py
=============
FastAPI application — route handlers, rate limiting, deployment utilities,
and CLI entry point.

Rate limits (per source IP):
  POST /analyze           10 req/min   (LLM calls are expensive)
  POST /webhook/moltis    30 req/min   (webhook bursts allowed)
  POST /schedule           5 req/min   (scheduling is rare)
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from bridge.client import MoltisConfig
from bridge.scene_ops import build_demo_scene_ops_request
from bridge.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    HealthResponse,
    SceneOpsAggregateRequest,
    SceneOpsSnapshotResponse,
    SceneReviewRequest,
    SceneReviewResponse,
    ScheduleRequest,
)
from bridge.service import AndrewMoltisBridge

logger = logging.getLogger("bridge_api")

# ── Rate limiter ─────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)

# ── Singleton bridge instance ────────────────────────────────

_bridge: Optional[AndrewMoltisBridge] = None


def get_cors_origins() -> list[str]:
    raw = os.getenv(
        "CORS_ALLOW_ORIGINS",
        "https://romeoflexvision.com,https://www.romeoflexvision.com,http://localhost:5173,http://127.0.0.1:5173,http://localhost:4173,http://127.0.0.1:4173",
    )
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def get_bridge() -> AndrewMoltisBridge:
    global _bridge
    if _bridge is None:
        _bridge = AndrewMoltisBridge()
    return _bridge


# ── Lifespan ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    global _bridge
    if _bridge:
        await _bridge.close()


# ── FastAPI app ──────────────────────────────────────────────

app = FastAPI(
    title="Andrew Swarm — Moltis Bridge",
    version="1.0.0-rc1",
    description="Connects Andrew's analytical brain to Moltis's Rust runtime",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ── Endpoints ────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Andrew, the Moltis runtime, and the bridge itself."""
    bridge = get_bridge()
    moltis_health = await bridge.moltis.health_check()
    return HealthResponse(andrew="ok", moltis=moltis_health, bridge="ok")


@app.post("/analyze", response_model=AnalyzeResponse)
@limiter.limit("10/minute")
async def analyze(request: Request, req: AnalyzeRequest):
    """
    Submit an analytical or educational query.

    Moltis hooks call this when a user message matches analytical intent.
    The query is routed to Andrew (analytics), Romeo (education), or both.
    """
    bridge = get_bridge()
    result = await bridge.handle_query(
        req.query,
        context={"channel": req.channel, "user_id": req.user_id, "session_id": req.session_id},
    )
    return AnalyzeResponse(**{k: v for k, v in result.items() if k in AnalyzeResponse.model_fields})


@app.post("/scene/review", response_model=SceneReviewResponse)
@limiter.limit("10/minute")
async def review_scene(request: Request, req: SceneReviewRequest):
    """
    Review a PinoCut scene bundle and return Andrew-style QA output.

    Intended for Pinnocat / PinoCut scene-level validation before final export.
    """
    bridge = get_bridge()
    result = await bridge.handle_scene_review(req.model_dump(), context={"channel": "api"})
    return SceneReviewResponse(
        **{k: v for k, v in result.items() if k in SceneReviewResponse.model_fields}
    )


@app.post("/scene/ops", response_model=SceneOpsSnapshotResponse)
@limiter.limit("10/minute")
async def aggregate_scene_ops(request: Request, req: SceneOpsAggregateRequest):
    """
    Aggregate a frontend-ready SceneOps snapshot for RomeoFlexVision.

    This contract adapts PinoCut scene state, Andrew QA, and Bassito job status
    into one response model that the React frontend can render directly.
    """
    bridge = get_bridge()
    result = await bridge.handle_scene_ops(req.model_dump(), context={"channel": "api"})
    return SceneOpsSnapshotResponse(
        **{k: v for k, v in result.items() if k in SceneOpsSnapshotResponse.model_fields}
    )


@app.get("/scene/ops", response_model=SceneOpsSnapshotResponse)
async def get_scene_ops():
    """
    Stable GET seam for RomeoFlexVision.

    Until a persisted scene-state store is imported into this repo, this route
    returns the deterministic demo snapshot used for frontend integration.
    """
    bridge = get_bridge()
    result = await bridge.handle_scene_ops(
        build_demo_scene_ops_request().model_dump(),
        context={"channel": "api"},
    )
    return SceneOpsSnapshotResponse(
        **{k: v for k, v in result.items() if k in SceneOpsSnapshotResponse.model_fields}
    )


@app.get("/scene/ops/demo", response_model=SceneOpsSnapshotResponse)
async def demo_scene_ops():
    """
    Deterministic demo snapshot for frontend wiring and design review.
    """
    bridge = get_bridge()
    result = await bridge.handle_scene_ops(
        build_demo_scene_ops_request().model_dump(),
        context={"channel": "api"},
    )
    return SceneOpsSnapshotResponse(
        **{k: v for k, v in result.items() if k in SceneOpsSnapshotResponse.model_fields}
    )


@app.post("/webhook/moltis")
@limiter.limit("30/minute")
async def moltis_webhook(request: Request):
    """
    Webhook endpoint for the Moltis hook system.

    Configure in Moltis HOOK.md:
        name: andrew-analytics
        event: MessageReceived
        handler: http://localhost:8100/webhook/moltis
    """
    body = await request.json()
    message = body.get("message", {}).get("content", "")
    channel = body.get("channel", "unknown")
    user_id = body.get("user", {}).get("id", "")

    if not message:
        return {"status": "skipped", "reason": "empty message"}

    # Only process messages that look like analytical or educational queries
    analytical_signals = [
        "analyze", "analysis", "report", "revenue", "sales", "forecast",
        "trend", "calculate", "query", "data", "statistics", "compare",
        "predict", "average", "total", "breakdown",
        "explain", "what is", "how does", "tutorial", "define",
    ]
    if not any(signal in message.lower() for signal in analytical_signals):
        return {"status": "skipped", "reason": "not analytical — Moltis handles this"}

    bridge = get_bridge()
    result = await bridge.handle_query(
        message,
        context={"channel": channel, "user_id": user_id},
    )
    return {
        "status": "completed",
        "response": result.get("formatted_message", "Analysis complete."),
        "confidence": result.get("confidence", 0),
    }


@app.post("/schedule")
@limiter.limit("5/minute")
async def schedule_analysis(request: Request, req: ScheduleRequest):
    """Schedule a recurring analytical task via Moltis cron."""
    bridge = get_bridge()
    success = await bridge.moltis.add_cron_job(
        schedule=req.cron_schedule,
        task=req.query,
        name=req.name,
    )
    if success:
        return {"status": "scheduled", "schedule": req.cron_schedule, "query": req.query}
    raise HTTPException(status_code=500, detail="Failed to create cron job in Moltis")


# ── Deployment utilities ─────────────────────────────────────

def generate_moltis_hook_config(bridge_port: int = 8100) -> str:
    """
    Generate HOOK.md content for Moltis to connect to Andrew.
    Place at: ~/.moltis/hooks/andrew-analytics/HOOK.md
    """
    return f"""---
name: andrew-analytics
version: 1.0.0
description: Routes analytical queries to Andrew Swarm's LangGraph brain
events:
  - MessageReceived
priority: 10
enabled: true
---

# Andrew Analytics Hook

When a user message contains analytical intent (revenue, forecast, trend, etc.),
this hook forwards it to Andrew Swarm for deep analysis.

## Handler

```bash
curl -s -X POST http://localhost:{bridge_port}/webhook/moltis \\
  -H "Content-Type: application/json" \\
  -d '{{"message": {{"content": "${{MESSAGE}}"}}, "channel": "${{CHANNEL}}", "user": {{"id": "${{USER_ID}}"}}}}'
```

## Configuration

The bridge runs on port {bridge_port} alongside Moltis (port 13131).
Set MOLTIS_HOST and DATABASE_URL in the bridge's environment.
"""


def generate_docker_compose(bridge_port: int = 8100) -> str:
    """Generate docker-compose.yml for the full Andrew + Moltis stack."""
    return f"""version: "3.8"

services:
  # Moltis: Rust runtime (channels, memory, sandbox, cron)
  moltis:
    image: ghcr.io/moltis-org/moltis:latest
    container_name: andrew-moltis
    restart: unless-stopped
    ports:
      - "13131:13131"   # Gateway (Web UI + API)
      - "13132:13132"   # WebSocket
    volumes:
      - moltis-config:/home/moltis/.config/moltis
      - moltis-data:/home/moltis/.moltis
      # docker.sock needed for per-session sandbox containers.
      # Restrict with a socket proxy (e.g. tecnativa/docker-socket-proxy)
      # in production.
      - /var/run/docker.sock:/var/run/docker.sock
    environment:
      - MOLTIS_PASSWORD=${{MOLTIS_PASSWORD:?Set MOLTIS_PASSWORD in .env}}
    healthcheck:
      test: ["CMD", "moltis", "doctor"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Andrew Bridge: Python FastAPI service
  andrew-bridge:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: andrew-bridge
    restart: unless-stopped
    ports:
      - "{bridge_port}:8100"
    environment:
      - MOLTIS_HOST=moltis
      - MOLTIS_PORT=13131
      - MOLTIS_PASSWORD=${{MOLTIS_PASSWORD:?Set MOLTIS_PASSWORD in .env}}
      - DATABASE_URL=${{DATABASE_URL:-sqlite:///data/andrew.db}}
      - OPENAI_API_KEY=${{OPENAI_API_KEY}}
      - ANTHROPIC_API_KEY=${{ANTHROPIC_API_KEY}}
      - ANDREW_MAX_COST=1.00
    volumes:
      - andrew-data:/data
    depends_on:
      moltis:
        condition: service_healthy

  # PostgreSQL (optional — start with: docker compose --profile production up)
  postgres:
    image: postgres:16-alpine
    container_name: andrew-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=andrew
      - POSTGRES_USER=andrew
      - POSTGRES_PASSWORD=${{PG_PASSWORD:?Set PG_PASSWORD in .env}}
    ports:
      - "5432:5432"
    volumes:
      - pg-data:/var/lib/postgresql/data
    profiles: [production]

volumes:
  moltis-config:
  moltis-data:
  andrew-data:
  pg-data:
"""


# ── CLI entry point ──────────────────────────────────────────

def main():
    import uvicorn
    if len(sys.argv) > 1 and sys.argv[1] == "generate-hook":
        print(generate_moltis_hook_config())
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "generate-compose":
        print(generate_docker_compose())
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        import asyncio

        async def _test():
            bridge = AndrewMoltisBridge()
            health = await bridge.moltis.health_check()
            print(f"Moltis health: {health}")
            if health.get("status") != "unreachable":
                result = await bridge.handle_query("What is the total revenue by region?")
                print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
            else:
                print("\nMoltis not running. Start with: docker compose up -d")

        asyncio.run(_test())
        sys.exit(0)

    port = int(os.getenv("BRIDGE_PORT", "8100"))
    uvicorn.run("bridge.api:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
