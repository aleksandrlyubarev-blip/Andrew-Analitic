"""
bridge/client.py
================
Moltis connection configuration and async HTTP/GraphQL client.
"""

import hashlib
import logging
import os
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import httpx

logger = logging.getLogger("moltis_client")


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
# Client
# ============================================================

class MoltisClient:
    """
    Async HTTP/GraphQL client for the Moltis runtime.

    Handles:
    - Health checks
    - Sending messages (chat API)
    - Memory operations (store/recall)
    - Sandbox execution requests
    - Cron job management
    """

    def __init__(self, config: Optional[MoltisConfig] = None):
        self.config = config or MoltisConfig.from_env()
        self._client: Optional[httpx.AsyncClient] = None

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
        """Send a message to Moltis's chat API."""
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
        """Execute code in Moltis's Docker sandbox (replaces E2B — zero cloud cost)."""
        gql = """
        mutation ExecuteCode($code: String!, $language: String!) {
            toolCall(name: "shell", input: $code) {
                output
                error
                exitCode
            }
        }
        """
        wrapped = f'python3 -c """{code}"""' if language == "python" else code
        result = await self.graphql_query(gql, {"code": wrapped, "language": language})
        return result.get("data", {}).get("toolCall", {})

    async def add_cron_job(self, schedule: str, task: str, name: str = "") -> bool:
        """Schedule a recurring analytics task via Moltis's cron system."""
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
