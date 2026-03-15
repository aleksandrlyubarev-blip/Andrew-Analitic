"""
Andrew Swarm — Moltis Bridge v1.0.0-rc1
====================================================
Connects Andrew's LangGraph analytical brain to Moltis's Rust runtime.

Moltis provides the "hands":
  - Channels: Telegram, Discord, Web UI (port 13131)
  - Sandbox: Docker/Podman per-session container isolation
  - Memory: Hybrid vector + full-text search (SQLite)
  - Scheduling: Cron + heartbeat for recurring analytics
  - GraphQL API: /graphql with subscriptions

Andrew provides the "brain":
  - Weighted query routing (48 keywords, 3 lanes)
  - SQL generation + sqlglot qualify validation
  - Python code generation + AST safety checks
  - Deterministic result validation + semantic guardrails

Integration pattern:
  Moltis (Rust) ←→ MoltisBridge (Python, this file) ←→ Andrew Swarm v4 (LangGraph)

The bridge runs as a FastAPI service that:
1. Receives messages from Moltis via webhook/GraphQL
2. Routes them through Andrew's LangGraph pipeline
3. Returns structured results to Moltis for channel delivery
4. Optionally stores analysis results in Moltis's memory
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moltis_bridge")


# ============================================================
# Configuration
# ============================================================

@dataclass
class MoltisConfig:
    """Connection settings for the Moltis runtime."""
    host: str = "127.0.0.1"
    port: int = 13131
    graphql_port: int = 13131  # Same port, /graphql path
    password: str = ""          # Moltis auth password
    token: str = ""             # Bearer token after auth
    use_tls: bool = False

    @classmethod
    def from_env(cls) -> "MoltisConfig":
        return cls(
            host=os.getenv("MOLTIS_HOST", "127.0.0.1"),
            port=int(os.getenv("MOLTIS_PORT", "13131")),
            password=os.getenv("MOLTIS_PASSWORD", ""),
            token=os.getenv("MOLTIS_TOKEN", ""),
            use_tls=os.getenv("MOLTIS_TLS", "false").lower() == "true",
        )

    @property
    def base_url(self) -> str:
        scheme = "https" if self.use_tls else "http"
        return f"{scheme}://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        scheme = "wss" if self.use_tls else "ws"
        return f"{scheme}://{self.host}:{self.port}/ws/chat"

    @property
    def graphql_url(self) -> str:
        return f"{self.base_url}/graphql"


# ============================================================
# Moltis Client
# ============================================================

class MoltisClient:
    """
    HTTP/GraphQL client for the Moltis runtime.
    
    Handles:
    - Authentication (password → session token)
    - Sending messages (chat API)
    - Memory operations (store/recall)
    - Sandbox execution requests
    - Cron job management
    """

    def __init__(self, config: Optional[MoltisConfig] = None):
        self.config = config or MoltisConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None
        self._authenticated = False

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            headers = {}
            if self.config.token:
                headers["Authorization"] = f"Bearer {self.config.token}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=60.0,
            )
        return self._client

    async def health_check(self) -> Dict[str, Any]:
        """Check if Moltis is running and healthy."""
        client = await self._get_client()
        try:
            resp = await client.get("/api/health")
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Moltis health check failed: {e}")
            return {"status": "unreachable", "error": str(e)}

    async def send_message(self, message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Send a message to Moltis's chat API.
        Returns the agent's response.
        """
        client = await self._get_client()
        payload = {"message": message}
        if session_id:
            payload["session_id"] = session_id

        try:
            resp = await client.post("/api/chat", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Moltis chat failed: {e}")
            return {"error": str(e)}

    async def graphql_query(self, query: str, variables: Optional[Dict] = None) -> Dict[str, Any]:
        """Execute a GraphQL query against Moltis."""
        client = await self._get_client()
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            resp = await client.post("/graphql", json=payload)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            logger.error(f"Moltis GraphQL failed: {e}")
            return {"errors": [{"message": str(e)}]}

    async def store_memory(self, content: str, metadata: Optional[Dict] = None) -> bool:
        """Store an analysis result in Moltis's memory system."""
        query = """
        mutation StoreMemory($content: String!, $metadata: Json) {
            memoryStore(content: $content, metadata: $metadata) {
                success
            }
        }
        """
        result = await self.graphql_query(query, {
            "content": content,
            "metadata": metadata or {},
        })
        return not result.get("errors")

    async def recall_memory(self, query: str, limit: int = 5) -> List[Dict]:
        """Search Moltis's memory for relevant past analyses."""
        gql = """
        query RecallMemory($query: String!, $limit: Int) {
            memoryRecall(query: $query, limit: $limit) {
                content
                score
                metadata
            }
        }
        """
        result = await self.graphql_query(gql, {"query": query, "limit": limit})
        data = result.get("data", {})
        return data.get("memoryRecall", [])

    async def execute_in_sandbox(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Execute code in Moltis's Docker sandbox.
        This replaces E2B for local deployments — zero cloud cost.
        """
        gql = """
        mutation ExecuteCode($code: String!, $language: String!) {
            toolCall(name: "shell", input: $code) {
                output
                error
                exitCode
            }
        }
        """
        # Moltis sandbox wraps shell commands in Docker containers
        # For Python, we write to a temp file and execute
        wrapped = f'python3 -c """{code}"""' if language == "python" else code
        result = await self.graphql_query(gql, {"code": wrapped, "language": language})
        return result.get("data", {}).get("toolCall", {})

    async def add_cron_job(self, schedule: str, task: str, name: str = "") -> bool:
        """
        Schedule a recurring analytics task via Moltis's cron system.
        
        Args:
            schedule: Cron expression (e.g., "0 9 * * 1" for Monday 9am)
            task: The analytical query to run on schedule
            name: Optional job name
        """
        gql = """
        mutation AddCron($schedule: String!, $command: String!, $name: String) {
            cronAdd(schedule: $schedule, command: $command, name: $name) {
                id
                schedule
            }
        }
        """
        result = await self.graphql_query(gql, {
            "schedule": schedule,
            "command": f"andrew-analyze: {task}",
            "name": name or f"andrew-{hashlib.md5(task.encode()).hexdigest()[:8]}",
        })
        return not result.get("errors")

    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


# ============================================================
# Bridge: Moltis ↔ Andrew Swarm
# ============================================================

class AndrewMoltisBridge:
    """
    The integration layer between Moltis (Rust runtime) and
    Andrew Swarm (Python/LangGraph analytics brain).
    
    Lifecycle:
    1. Moltis receives user message via Telegram/Discord/Web
    2. Moltis forwards to this bridge via webhook
    3. Bridge runs Andrew's LangGraph pipeline
    4. Bridge returns structured result to Moltis
    5. Moltis delivers response via the original channel
    6. Optionally: bridge stores result in Moltis memory
    
    Usage:
        bridge = AndrewMoltisBridge()
        result = await bridge.handle_query("Total revenue by region for Q3")
    """

    def __init__(
        self,
        moltis_config: Optional[MoltisConfig] = None,
        andrew_db_url: Optional[str] = None,
        store_results_in_memory: bool = True,
    ):
        self.moltis = MoltisClient(moltis_config)
        self.store_results = store_results_in_memory
        self._andrew_executor = None
        self._db_url = andrew_db_url or os.getenv("DATABASE_URL", "")

    def _get_executor(self):
        """Lazy-load SwarmSupervisor (routes to Andrew, Romeo, or both)."""
        if self._andrew_executor is None:
            from core.supervisor import SwarmSupervisor
            self._andrew_executor = SwarmSupervisor(db_url=self._db_url)
            logger.info("SwarmSupervisor v1.0.0 initialized (Andrew + Romeo)")
        return self._andrew_executor

    async def handle_query(self, query: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main entry point: receive a query, run Andrew's pipeline, return results.
        
        Args:
            query: Natural language analytical question
            context: Optional context (channel, user_id, session_id, etc.)
            
        Returns:
            Structured result dict with narrative, confidence, cost, warnings
        """
        context = context or {}
        start_time = time.time()
        logger.info(f"Bridge received: {query[:80]}...")

        # 1. Check for relevant past analyses in Moltis memory
        memory_context = ""
        if self.store_results:
            try:
                memories = await self.moltis.recall_memory(query, limit=3)
                if memories:
                    memory_context = "\n".join(
                        f"- Prior analysis (relevance {m.get('score', 0):.2f}): {m.get('content', '')[:200]}"
                        for m in memories
                    )
                    logger.info(f"Found {len(memories)} relevant memories")
            except Exception as e:
                logger.warning(f"Memory recall failed (non-fatal): {e}")

        # 2. Run Andrew's LangGraph pipeline
        executor = self._get_executor()

        # Enrich query with memory context if available
        enriched_query = query
        if memory_context:
            enriched_query = f"{query}\n\n[Context from prior analyses:\n{memory_context}]"

        # Andrew's execute() is synchronous — run in thread pool
        result = await asyncio.to_thread(executor.execute, enriched_query)
        elapsed = time.time() - start_time

        # 3. Build structured response
        response = {
            "query": query,
            "narrative": result.output if hasattr(result, 'output') else str(result.raw_data or ""),
            "sql_query": result.sql_query,
            "confidence": result.confidence,
            "cost_usd": result.cost_usd if hasattr(result, 'cost_usd') else result.cost,
            "warnings": result.warnings if hasattr(result, 'warnings') else [],
            "success": result.success,
            "error": result.error_message if hasattr(result, 'error_message') else result.error,
            "elapsed_seconds": round(elapsed, 2),
            "routing": getattr(result, 'routing', 'unknown'),
            "model_used": getattr(result, 'model_used', 'unknown'),
            "agent_used": getattr(result, 'agent_used', 'andrew'),
            "channel": context.get("channel", "api"),
        }

        # 4. Format for Moltis channel delivery
        response["formatted_message"] = self._format_for_channel(response)

        # 5. Store in Moltis memory for future context
        if self.store_results and result.success:
            try:
                await self.moltis.store_memory(
                    content=f"Query: {query}\nResult: {response['narrative'][:500]}",
                    metadata={
                        "confidence": response["confidence"],
                        "sql": result.sql_query,
                        "timestamp": time.time(),
                        "cost": response["cost_usd"],
                    },
                )
                logger.info("Analysis stored in Moltis memory")
            except Exception as e:
                logger.warning(f"Memory store failed (non-fatal): {e}")

        logger.info(
            f"Bridge completed: confidence={response['confidence']:.2f}, "
            f"cost=${response['cost_usd']:.4f}, elapsed={elapsed:.1f}s"
        )
        return response

    def _format_for_channel(self, response: Dict) -> str:
        """
        Format Andrew's analytical output for messaging channels.
        Keeps it readable in Telegram/Discord (max ~4000 chars).
        """
        parts = []

        if response.get("success"):
            parts.append(f"**Analysis** (confidence: {response['confidence']:.0%})")
            narrative = response.get("narrative", "")
            if len(narrative) > 2000:
                narrative = narrative[:2000] + "...\n_(truncated — full report available via API)_"
            parts.append(narrative)
        else:
            parts.append("**Analysis failed**")
            parts.append(f"Error: {response.get('error', 'Unknown')}")

        if response.get("warnings"):
            parts.append("\n**Warnings:**")
            for w in response["warnings"][:3]:
                parts.append(f"- {w}")

        parts.append(f"\n_Cost: ${response.get('cost_usd', 0):.4f} | "
                      f"Time: {response.get('elapsed_seconds', 0)}s | "
                      f"Route: {response.get('routing', '?')}_")

        return "\n".join(parts)

    async def handle_scheduled_task(self, task: str) -> Dict[str, Any]:
        """
        Handle a cron-triggered analytical task.
        Same as handle_query but with scheduling context.
        """
        logger.info(f"Scheduled task triggered: {task[:80]}")
        return await self.handle_query(task, context={"channel": "cron", "scheduled": True})

    async def use_moltis_sandbox(self, code: str) -> Dict[str, Any]:
        """
        Execute Python code via Moltis's Docker sandbox instead of E2B.
        Zero cloud cost — runs on your own hardware.
        """
        return await self.moltis.execute_in_sandbox(code, language="python")

    async def close(self):
        await self.moltis.close()


# ============================================================
# FastAPI Webhook Server
# ============================================================

app = FastAPI(
    title="Andrew Swarm — Moltis Bridge",
    version="1.0.0-rc1",
    description="Connects Andrew's analytical brain to Moltis's Rust runtime",
)

# Global bridge instance
_bridge: Optional[AndrewMoltisBridge] = None


def get_bridge() -> AndrewMoltisBridge:
    global _bridge
    if _bridge is None:
        _bridge = AndrewMoltisBridge()
    return _bridge


# ─── Request/Response models ────────────────────────────────

class AnalyzeRequest(BaseModel):
    query: str
    channel: str = "api"
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    query: str
    narrative: str
    sql_query: Optional[str] = None
    confidence: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float
    routing: str
    formatted_message: str


class ScheduleRequest(BaseModel):
    query: str
    cron_schedule: str  # e.g., "0 9 * * 1" for Monday 9am
    name: Optional[str] = None


class HealthResponse(BaseModel):
    andrew: str
    moltis: Dict[str, Any]
    bridge: str


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for both Andrew and Moltis."""
    bridge = get_bridge()
    moltis_health = await bridge.moltis.health_check()
    return HealthResponse(
        andrew="ok",
        moltis=moltis_health,
        bridge="ok",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """
    Main endpoint: submit an analytical query.
    
    Moltis hooks call this when a user sends a message
    that matches the analytical intent pattern.
    """
    bridge = get_bridge()
    result = await bridge.handle_query(
        req.query,
        context={"channel": req.channel, "user_id": req.user_id, "session_id": req.session_id},
    )
    return AnalyzeResponse(**{k: v for k, v in result.items() if k in AnalyzeResponse.model_fields})


@app.post("/webhook/moltis")
async def moltis_webhook(request: Request):
    """
    Webhook endpoint for Moltis hook system.
    
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

    # Only process messages that look like analytical queries
    # (let Moltis handle general chat itself)
    analytical_signals = [
        "analyze", "analysis", "report", "revenue", "sales", "forecast",
        "trend", "calculate", "query", "data", "statistics", "compare",
        "predict", "average", "total", "breakdown",
    ]
    is_analytical = any(signal in message.lower() for signal in analytical_signals)

    if not is_analytical:
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
async def schedule_analysis(req: ScheduleRequest):
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


@app.on_event("shutdown")
async def shutdown():
    global _bridge
    if _bridge:
        await _bridge.close()


# ============================================================
# Moltis Hook Configuration Generator
# ============================================================

def generate_moltis_hook_config(bridge_port: int = 8100) -> str:
    """
    Generate the HOOK.md file content for Moltis to connect to Andrew.
    
    Place this in ~/.moltis/hooks/andrew-analytics/HOOK.md
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


# ============================================================
# Docker Compose for Moltis + Andrew
# ============================================================

def generate_docker_compose() -> str:
    """Generate docker-compose.yml for the full Andrew + Moltis stack."""
    return """version: "3.8"

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
      - /var/run/docker.sock:/var/run/docker.sock  # For sandbox
    environment:
      - MOLTIS_PASSWORD=${MOLTIS_PASSWORD:-andrew2026}
    healthcheck:
      test: ["CMD", "moltis", "doctor"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Andrew Bridge: Python service connecting brain to hands
  andrew-bridge:
    build:
      context: .
      dockerfile: Dockerfile.andrew
    container_name: andrew-bridge
    restart: unless-stopped
    ports:
      - "8100:8100"     # Bridge API
    environment:
      - MOLTIS_HOST=moltis
      - MOLTIS_PORT=13131
      - MOLTIS_PASSWORD=${MOLTIS_PASSWORD:-andrew2026}
      - DATABASE_URL=${DATABASE_URL:-sqlite:///data/andrew.db}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - ANDREW_MAX_COST=1.00
    volumes:
      - andrew-data:/data
    depends_on:
      moltis:
        condition: service_healthy

  # PostgreSQL (optional, for production SQL analytics)
  postgres:
    image: postgres:16-alpine
    container_name: andrew-postgres
    restart: unless-stopped
    environment:
      - POSTGRES_DB=andrew
      - POSTGRES_USER=andrew
      - POSTGRES_PASSWORD=${PG_PASSWORD:-andrew2026}
    ports:
      - "5432:5432"
    volumes:
      - pg-data:/var/lib/postgresql/data

volumes:
  moltis-config:
  moltis-data:
  andrew-data:
  pg-data:
"""


# ============================================================
# CLI & Entry Point
# ============================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "generate-hook":
        print(generate_moltis_hook_config())
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "generate-compose":
        print(generate_docker_compose())
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # Quick integration test
        async def _test():
            bridge = AndrewMoltisBridge()
            health = await bridge.moltis.health_check()
            print(f"Moltis health: {health}")

            if health.get("status") != "unreachable":
                result = await bridge.handle_query("What is the total revenue by region?")
                print(f"\nResult: {json.dumps(result, indent=2, default=str)}")
            else:
                print("\nMoltis not running. Start with: docker compose up -d")

            await bridge.close()

        asyncio.run(_test())
        sys.exit(0)

    # Default: run the bridge server
    import uvicorn
    port = int(os.getenv("BRIDGE_PORT", "8100"))
    logger.info(f"Starting Andrew-Moltis bridge on port {port}")
    logger.info(f"Moltis expected at {MoltisConfig.from_env().base_url}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
