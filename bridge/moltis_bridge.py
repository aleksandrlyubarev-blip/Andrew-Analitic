"""
bridge/moltis_bridge.py
=======================
Thin compatibility shim and uvicorn entry point.

The bridge has been split into focused modules:
  bridge/client.py   — MoltisConfig + MoltisClient (HTTP/GraphQL)
  bridge/schemas.py  — Pydantic request/response models
  bridge/service.py  — AndrewMoltisBridge orchestration
  bridge/hitl.py     — Human-in-the-loop gate
  bridge/api.py      — FastAPI app, endpoints, rate limiting, utils, CLI

Andrew provides the "brain":
  - Semantic routing (embedding cosine scoring, Capability Registry, Sprint 5)
  - SQL generation + sqlglot qualify validation
  - Python code generation + AST safety checks
  - Deterministic result validation + semantic guardrails
  - HITL escalation for low-confidence results (Sprint 7)

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
import threading
import uuid
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")
logger = logging.getLogger("moltis_bridge")


# ============================================================
# Rate Limiting (Sprint 6 security hardening)
# ============================================================
# Per-IP sliding-window rate limiter.  Fully in-process — no Redis needed.
# Configure via env vars:
#   RATE_LIMIT_ANALYZE   default "10/60"   → 10 requests per 60 s per IP
#   RATE_LIMIT_EDUCATE   default "20/60"   → 20 requests per 60 s per IP
#   RATE_LIMIT_WEBHOOK   default "30/60"   → 30 requests per 60 s per IP
# Set to "0/60" to disable a limiter entirely.


def _parse_rate_spec(spec: str, default_req: int, default_win: int) -> Tuple[int, int]:
    """Parse 'N/W' into (max_requests, window_seconds). Returns defaults on error."""
    try:
        req_str, win_str = spec.split("/", 1)
        return int(req_str), int(win_str)
    except Exception:
        return default_req, default_win


class SlidingWindowRateLimiter:
    """
    Thread-safe per-key sliding-window rate limiter.

    Each key (typically a client IP) gets an independent window.
    is_allowed() is O(k) where k is the number of requests in the window —
    typically very small.
    """

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._windows: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> Tuple[bool, int]:
        """
        Check whether the key may make another request right now.

        Returns:
            (True, 0)           — request allowed
            (False, retry_after) — denied; caller should wait retry_after seconds
        """
        if self.max_requests == 0:
            return True, 0                  # limiter disabled

        now = time.time()
        cutoff = now - self.window_seconds

        with self._lock:
            window = self._windows[key]
            # Evict timestamps older than the window
            while window and window[0] <= cutoff:
                window.popleft()

            if len(window) >= self.max_requests:
                # How many seconds until the oldest slot expires
                retry_after = max(1, int(window[0] - cutoff) + 1)
                return False, retry_after

            window.append(now)
            return True, 0

    def _prune(self) -> None:
        """Remove keys whose entire window has expired. Prevents unbounded dict growth."""
        cutoff = time.time() - self.window_seconds
        with self._lock:
            stale = [k for k, v in self._windows.items() if not v or v[-1] <= cutoff]
            for k in stale:
                del self._windows[k]

    def reset(self, key: str) -> None:
        """Clear all recorded timestamps for a key (test helper)."""
        with self._lock:
            self._windows.pop(key, None)

    def __repr__(self) -> str:
        return (
            f"SlidingWindowRateLimiter(max={self.max_requests}, "
            f"window={self.window_seconds}s)"
        )


def _build_limiter(env_var: str, default_req: int, default_win: int) -> SlidingWindowRateLimiter:
    spec = os.getenv(env_var, f"{default_req}/{default_win}")
    req, win = _parse_rate_spec(spec, default_req, default_win)
    limiter = SlidingWindowRateLimiter(req, win)
    logger.info(f"RateLimit {env_var}: {limiter}")
    return limiter


# Module-level limiter instances (one per rate-limited endpoint)
_limiter_analyze = _build_limiter("RATE_LIMIT_ANALYZE", 10, 60)
_limiter_educate = _build_limiter("RATE_LIMIT_EDUCATE", 20, 60)
_limiter_webhook = _build_limiter("RATE_LIMIT_WEBHOOK", 30, 60)


def _client_ip(request: Request) -> str:
    """Extract the real client IP, honoring X-Forwarded-For from reverse proxies."""
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def _check_rate_limit(limiter: SlidingWindowRateLimiter, request: Request) -> None:
    """Raise HTTP 429 if the client has exceeded its rate limit."""
    ip = _client_ip(request)
    allowed, retry_after = limiter.is_allowed(ip)
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


# ============================================================
# Episodic Memory Store (Sprint 5 §5.1)
# ============================================================
# Session-scoped in-process store: {session_id -> [{"role", "content", "ts"}]}
# Cleared on process restart; a durable store (Redis / Moltis) can replace this.

_episodic_store: Dict[str, List[Dict[str, Any]]] = {}
SEMANTIC_RELEVANCE_THRESHOLD = float(os.getenv("SEMANTIC_RELEVANCE_THRESHOLD", "0.65"))
EPISODIC_MAX_ENTRIES = int(os.getenv("EPISODIC_MAX_ENTRIES", "20"))

# Shared memory + router singletons (Sprint 5 §5.4)
from core.memory import _semantic_store, _consolidation_engine, SweepResult  # noqa: E402
from core.semantic_router import _semantic_router                              # noqa: E402
from core.andrew_swarm import get_stored_result                                # noqa: E402

# ── Bridge uptime + last sweep tracking ───────────────────────────────────────
_bridge_start_time: float = 0.0
_last_sweep: Optional[SweepResult] = None

# ── Operational counters (in-process; reset on restart) ───────────────────────
_metrics_lock = threading.Lock()
_metrics: Dict[str, Any] = {
    "queries_total": 0,
    "queries_success": 0,
    "queries_hitl": 0,
    "queries_error": 0,
    "educate_total": 0,
    "total_cost_usd": 0.0,
    "confidence_sum": 0.0,
    "confidence_count": 0,
    "routing_lanes": {"reasoning_math": 0, "analytics_fastlane": 0, "standard": 0},
}

# ── In-process job scheduler ───────────────────────────────────────────────────
# _job_store: job_id → {"id", "name", "query", "cron_expr",
#                        "next_run", "last_run", "run_count", "created_at"}
_job_store: Dict[str, Dict] = {}
_job_store_lock = threading.Lock()

try:
    from croniter import croniter as _croniter
    _CRONITER_AVAILABLE = True
except ImportError:
    _CRONITER_AVAILABLE = False
    logger.warning("croniter not installed — in-process scheduler disabled; pip install croniter")


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
        """Lazy-load Andrew executor to avoid import-time side effects."""
        if self._andrew_executor is None:
            # Import Andrew Swarm v4 — the full LangGraph pipeline
            from core.andrew_swarm import AndrewExecutor
            self._andrew_executor = AndrewExecutor(db_url=self._db_url)
            logger.info("Andrew Swarm v4 executor initialized")
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
        session_id = context.get("session_id") or "default"
        logger.info(f"Bridge received: {query[:80]}... [session={session_id}]")

        # ── 1. Episodic memory (§5.1) — session-scoped facts ─────────────────
        episodic = _episodic_store.setdefault(session_id, [])
        episodic.append({"role": "user", "content": query, "ts": time.time()})
        # Trim to max window
        if len(episodic) > EPISODIC_MAX_ENTRIES:
            _episodic_store[session_id] = episodic[-EPISODIC_MAX_ENTRIES:]
            episodic = _episodic_store[session_id]

        episodic_context = ""
        if len(episodic) > 1:
            episodic_context = "\n".join(
                f"[{e['role']}] {e['content'][:200]}" for e in episodic[:-1]
            )

        # Session summary for semantic router δ term (last N=10 turns joined)
        session_summary = " | ".join(e["content"][:80] for e in episodic[-10:]) or None

        # ── 2. Semantic memory (§5.2) — persistent knowledge, 0.65 threshold ─
        semantic_memories: List[Dict] = []
        memory_context = ""
        if self.store_results:
            try:
                raw_memories = await self.moltis.recall_memory(query, limit=5)
                # Apply §5.3 relevance threshold
                semantic_memories = [
                    m for m in (raw_memories or [])
                    if m.get("score", 0.0) >= SEMANTIC_RELEVANCE_THRESHOLD
                ]
                if semantic_memories:
                    memory_context = "\n".join(
                        f"- Prior analysis (relevance {m['score']:.2f}): {m.get('content', '')[:200]}"
                        for m in semantic_memories
                    )
                    logger.info(
                        f"Semantic memory: {len(semantic_memories)}/{len(raw_memories or [])} "
                        f"records above threshold {SEMANTIC_RELEVANCE_THRESHOLD}"
                    )
            except Exception as e:
                logger.warning(f"Memory recall failed (non-fatal): {e}")

        # ── 3. Context assembly (§6 step 3) ──────────────────────────────────
        enriched_query = query
        context_parts = []
        if episodic_context:
            context_parts.append(f"[Session history:\n{episodic_context}]")
        if memory_context:
            context_parts.append(f"[Long-term context:\n{memory_context}]")
        if context_parts:
            enriched_query = query + "\n\n" + "\n".join(context_parts)

        # ── 4. Run Andrew's LangGraph pipeline ───────────────────────────────
        executor = self._get_executor()
        result = await asyncio.to_thread(executor.execute, enriched_query)
        elapsed = time.time() - start_time

        # ── 5. Write result to episodic memory (§6 step 6a) ──────────────────
        output_snippet = (result.output or result.error or "")[:300]
        episodic.append({"role": "assistant", "content": output_snippet, "ts": time.time()})

        # ── 6. Build structured response ──────────────────────────────────────
        response = {
            "query": query,
            "narrative": result.output if hasattr(result, "output") else str(result.raw_data or ""),
            "sql_query": result.sql_query,
            "query_results": getattr(result, "query_results", []) or [],
            "confidence": result.confidence,
            "cost_usd": result.cost_usd if hasattr(result, "cost_usd") else result.cost,
            "warnings": result.warnings if hasattr(result, "warnings") else [],
            "success": result.success,
            "error": result.error_message if hasattr(result, "error_message") else result.error,
            "elapsed_seconds": round(elapsed, 2),
            "routing": getattr(result, "routing", "unknown"),
            "model_used": getattr(result, "model_used", "unknown"),
            "channel": context.get("channel", "api"),
            "hitl_required": getattr(result, "hitl_required", False),
            "hitl_reason": getattr(result, "hitl_reason", None),
            "session_id": session_id,
            "session_length": len(episodic),
            "memory_records_retrieved": len(semantic_memories),
        }

        # ── 7. Format for Moltis channel delivery ─────────────────────────────
        response["formatted_message"] = self._format_for_channel(response)

        # ── 8. Procedural feedback (§5.4 post-execution) ─────────────────────
        # Fire-and-forget: does not block the response path.
        routing_decision = result.routing or "standard_analytics"
        asyncio.create_task(
            asyncio.to_thread(
                _semantic_router.record_outcome, query, routing_decision, result.success
            )
        )

        # ── 9. Store in semantic memory (§5.4, §6 step 6b) ───────────────────
        if self.store_results and result.success:
            mem_content = f"Query: {query}\nResult: {response['narrative'][:500]}"
            mem_meta = {
                "confidence": response["confidence"],
                "sql": result.sql_query,
                "timestamp": time.time(),
                "cost": response["cost_usd"],
                "session_id": session_id,
            }
            try:
                await self.moltis.store_memory(content=mem_content, metadata=mem_meta)
                logger.info("Analysis stored in Moltis semantic memory")
            except Exception as e:
                logger.warning(f"Moltis memory store failed (non-fatal): {e}")
            # Mirror into in-process store for consolidation + staleness tracking
            emb = await asyncio.to_thread(_semantic_router.embed, query)
            if emb:
                _semantic_store.upsert(mem_content, emb, metadata=mem_meta)

        logger.info(
            f"Bridge completed: confidence={response['confidence']:.2f}, "
            f"cost=${response['cost_usd']:.4f}, elapsed={elapsed:.1f}s, "
            f"session_len={response['session_length']}, "
            f"semantic_memories={response['memory_records_retrieved']}"
        )
        return response

    async def end_session(self, session_id: str) -> Optional[str]:
        """
        Explicitly close a session: consolidate episodic memory into the
        semantic store (§5.4 "End of session").

        Returns the record_id of the stored/merged semantic record, or None.
        Called by the /end_session API endpoint or a channel disconnect hook.
        """
        episodic = _episodic_store.pop(session_id, [])
        if not episodic:
            logger.debug(f"end_session: nothing to consolidate for {session_id}")
            return None
        record_id = await asyncio.to_thread(
            _consolidation_engine.consolidate_session, session_id, episodic
        )
        logger.info(f"end_session: consolidated {len(episodic)} entries → {record_id}")
        return record_id

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

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown for the bridge."""
    global _bridge_start_time, _last_sweep, _bridge
    _bridge_start_time = time.time()
    interval = int(os.getenv("STALENESS_SWEEP_INTERVAL_S", str(24 * 3600)))

    async def _sweep_loop():
        while True:
            await asyncio.sleep(interval)
            try:
                _last_sweep = await asyncio.to_thread(_consolidation_engine.staleness_sweep)
                logger.info(f"Daily staleness sweep: {_last_sweep}")
                for lim in (_limiter_analyze, _limiter_educate, _limiter_webhook):
                    lim._prune()
            except Exception as exc:
                logger.warning(f"Staleness sweep failed: {exc}")

    async def _scheduler_loop():
        """Run due in-process scheduled jobs every 60 s."""
        while True:
            await asyncio.sleep(60)
            if not _CRONITER_AVAILABLE:
                continue
            now = time.time()
            with _job_store_lock:
                due = [j for j in _job_store.values() if j["next_run"] <= now]
            for job in due:
                try:
                    bridge = get_bridge()
                    result = await bridge.handle_query(
                        job["query"],
                        context={"channel": "scheduler", "scheduled": True, "job_id": job["id"]},
                    )
                    logger.info(f"Scheduled job {job['name']} ran; success={result.get('success')}")
                    with _job_store_lock:
                        if job["id"] in _job_store:
                            _job_store[job["id"]]["last_run"] = now
                            _job_store[job["id"]]["run_count"] += 1
                            _job_store[job["id"]]["next_run"] = _croniter(
                                job["cron_expr"], now
                            ).get_next(float)
                except Exception as exc:
                    logger.warning(f"Scheduled job {job['name']} failed: {exc}")

    sweep_task     = asyncio.create_task(_sweep_loop())
    scheduler_task = asyncio.create_task(_scheduler_loop())
    try:
        yield
    finally:
        sweep_task.cancel()
        scheduler_task.cancel()
        for t in (sweep_task, scheduler_task):
            try:
                await t
            except asyncio.CancelledError:
                pass
        if _bridge:
            await _bridge.close()


app = FastAPI(
    title="Andrew Swarm — Moltis Bridge",
    version="1.4.0",
    description="Connects Andrew's analytical brain to Moltis's Rust runtime",
    lifespan=lifespan,
)

# Serve the compiled Vue frontend from bridge/static/ (built by `npm run build`)
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=_STATIC_DIR / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    async def serve_spa():
        return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa_fallback(full_path: str):
        # Let API routes handle themselves; everything else → SPA index
        candidate = _STATIC_DIR / full_path
        if candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(_STATIC_DIR / "index.html")

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
    query_results: Optional[List[Dict[str, Any]]] = None
    confidence: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float
    routing: str
    formatted_message: str
    hitl_required: bool = False
    hitl_reason: Optional[str] = None
    session_id: Optional[str] = None
    session_length: int = 0
    memory_records_retrieved: int = 0
    data_profile: Optional[Dict[str, Any]] = None


class EducateRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class EducateResponse(BaseModel):
    question: str
    answer: str
    cost_usd: float
    model: str
    elapsed_seconds: float
    success: bool
    error: Optional[str] = None


class ScheduleRequest(BaseModel):
    query: str
    cron_schedule: str  # e.g., "0 9 * * 1" for Monday 9am
    name: Optional[str] = None


class HealthResponse(BaseModel):
    andrew: str
    moltis: Dict[str, Any]
    bridge: str
    uptime_seconds: float
    semantic_store_records: int
    procedural_store_records: int
    active_sessions: int
    rate_limits: Dict[str, Any]
    last_sweep: Optional[Dict[str, Any]]


# ─── Endpoints ──────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check — Andrew, Moltis, and bridge internal state."""
    bridge = get_bridge()
    moltis_health = await bridge.moltis.health_check()
    proc_store = _semantic_router.procedural_store
    return HealthResponse(
        andrew="ok",
        moltis=moltis_health,
        bridge="ok",
        uptime_seconds=round(time.time() - _bridge_start_time, 1),
        semantic_store_records=len(_semantic_store),
        procedural_store_records=len(proc_store) if proc_store else 0,
        active_sessions=len(_episodic_store),
        rate_limits={
            name: {"max": lim.max_requests, "window_s": lim.window_seconds}
            for name, lim in (
                ("analyze", _limiter_analyze),
                ("educate", _limiter_educate),
                ("webhook", _limiter_webhook),
            )
        },
        last_sweep=asdict(_last_sweep) if _last_sweep else None,
    )


@app.get("/metrics")
async def metrics():
    """
    Operational metrics — queries processed, cost, confidence, routing distribution.
    Counters reset on bridge restart (in-process only).
    """
    with _metrics_lock:
        snap = dict(_metrics)
    avg_confidence = (
        round(snap["confidence_sum"] / snap["confidence_count"], 3)
        if snap["confidence_count"] > 0 else None
    )
    success_rate = (
        round(snap["queries_success"] / snap["queries_total"], 3)
        if snap["queries_total"] > 0 else None
    )
    return {
        "uptime_seconds": round(time.time() - _bridge_start_time, 1),
        "queries": {
            "total":   snap["queries_total"],
            "success": snap["queries_success"],
            "hitl":    snap["queries_hitl"],
            "error":   snap["queries_error"],
            "success_rate": success_rate,
        },
        "educate_total": snap["educate_total"],
        "cost": {
            "total_usd":      round(snap["total_cost_usd"], 4),
            "avg_per_query":  round(snap["total_cost_usd"] / snap["queries_total"], 4)
                              if snap["queries_total"] > 0 else None,
        },
        "confidence": {
            "average": avg_confidence,
            "samples": snap["confidence_count"],
        },
        "routing_lanes": snap["routing_lanes"],
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, response: Response, request: Request):
    """
    Main endpoint: submit an analytical query.

    Returns HTTP 200 on success.
    Returns HTTP 202 when confidence is below the HITL threshold — the
    result is still populated but `hitl_required=true` signals that a
    human should review before acting on the output.
    Returns HTTP 429 when the per-IP rate limit is exceeded.

    Moltis hooks call this when a user sends a message
    that matches the analytical intent pattern.
    """
    _check_rate_limit(_limiter_analyze, request)
    bridge = get_bridge()
    result = await bridge.handle_query(
        req.query,
        context={"channel": req.channel, "user_id": req.user_id, "session_id": req.session_id},
    )
    body = AnalyzeResponse(**{k: v for k, v in result.items() if k in AnalyzeResponse.model_fields})
    with _metrics_lock:
        _metrics["queries_total"] += 1
        if body.error:
            _metrics["queries_error"] += 1
        elif body.hitl_required:
            _metrics["queries_hitl"] += 1
        else:
            _metrics["queries_success"] += 1
        _metrics["total_cost_usd"] += body.cost_usd or 0.0
        if body.confidence is not None:
            _metrics["confidence_sum"] += body.confidence
            _metrics["confidence_count"] += 1
        lane = (body.routing or "").lower()
        for key in ("reasoning_math", "analytics_fastlane", "standard"):
            if key in lane:
                _metrics["routing_lanes"][key] += 1
                break
    if body.hitl_required:
        response.status_code = 202
    return body


@app.post("/webhook/moltis")
async def moltis_webhook(request: Request):
    """
    Webhook endpoint for Moltis hook system.
    Returns HTTP 429 when the per-IP rate limit is exceeded.

    Configure in Moltis HOOK.md:
        name: andrew-analytics
        event: MessageReceived
        handler: http://localhost:8100/webhook/moltis
    """
    _check_rate_limit(_limiter_webhook, request)
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


@app.post("/educate", response_model=EducateResponse)
async def educate(req: EducateRequest, request: Request):
    """
    Ask Romeo PhD an educational question.
    Returns HTTP 429 when the per-IP rate limit is exceeded.

    Romeo handles conceptual, theoretical, and explanatory queries about
    data science, ML, statistics, mathematics, and programming.
    Returns a Markdown-formatted explanation.
    """
    _check_rate_limit(_limiter_educate, request)
    from core.romeo_phd import RomeoExecutor  # lazy import — keeps bridge fast when unused

    executor = RomeoExecutor()
    result = await asyncio.to_thread(executor.execute, req.question)
    with _metrics_lock:
        _metrics["educate_total"] += 1
    return EducateResponse(**result.to_dict())


@app.post("/schedule", status_code=201)
async def schedule_analysis(req: ScheduleRequest):
    """Schedule a recurring analytical task (in-process + Moltis cron fallback)."""
    if not _CRONITER_AVAILABLE:
        raise HTTPException(status_code=503, detail="croniter not installed; pip install croniter")

    try:
        next_run = _croniter(req.cron_schedule, time.time()).get_next(float)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid cron expression: {req.cron_schedule!r}")

    job_id = str(uuid.uuid4())
    name   = req.name or f"job-{job_id[:8]}"
    with _job_store_lock:
        _job_store[job_id] = {
            "id":         job_id,
            "name":       name,
            "query":      req.query,
            "cron_expr":  req.cron_schedule,
            "next_run":   next_run,
            "last_run":   None,
            "run_count":  0,
            "created_at": time.time(),
        }

    # Best-effort: also register with Moltis GraphQL (non-fatal)
    try:
        bridge = get_bridge()
        await bridge.moltis.add_cron_job(schedule=req.cron_schedule, task=req.query, name=name)
    except Exception as exc:
        logger.debug(f"Moltis cron registration skipped: {exc}")

    return {"status": "scheduled", "job_id": job_id, "name": name,
            "cron_schedule": req.cron_schedule, "next_run": next_run, "query": req.query}


@app.get("/schedule")
async def list_scheduled_jobs():
    """List all in-process scheduled jobs."""
    with _job_store_lock:
        jobs = list(_job_store.values())
    return {"jobs": jobs, "count": len(jobs)}


@app.delete("/schedule/{job_id}", status_code=200)
async def delete_scheduled_job(job_id: str):
    """Remove an in-process scheduled job by ID."""
    with _job_store_lock:
        if job_id not in _job_store:
            raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
        removed = _job_store.pop(job_id)
    return {"status": "removed", "job_id": job_id, "name": removed["name"]}


@app.get("/results/{hash_hex}")
async def get_result(hash_hex: str):
    """Return the persisted result snapshot for a given SHA-256 hash."""
    snapshot = get_stored_result(hash_hex)
    if snapshot is None:
        raise HTTPException(status_code=404, detail=f"No result found for hash {hash_hex!r}")
    return snapshot


@app.post("/session/{session_id}/end")
async def end_session(session_id: str):
    """
    Explicitly close a session and consolidate episodic → semantic memory.
    Called by channel disconnect hooks or explicit user action.
    """
    bridge = get_bridge()
    record_id = await bridge.end_session(session_id)
    return {"session_id": session_id, "consolidated_record": record_id}


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
    main()
