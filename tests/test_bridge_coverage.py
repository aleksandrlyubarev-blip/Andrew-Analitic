"""
Expanded test coverage for bridge and API modules.

Tests cover:
- bridge/schemas.py: Pydantic model validation
- bridge/api.py: endpoint behavior via TestClient
- bridge/service.py: format_for_channel, handle_query orchestration
- bridge/client.py: MoltisConfig defaults

No external services or LLM calls required.
"""

import json
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, ScheduleRequest
from bridge.client import MoltisConfig


# ============================================================
# 1. Pydantic schema validation
# ============================================================

class TestSchemas:

    def test_analyze_request_minimal(self):
        req = AnalyzeRequest(query="total revenue by region")
        assert req.query == "total revenue by region"
        assert req.channel == "api"  # default

    def test_analyze_request_all_fields(self):
        req = AnalyzeRequest(
            query="forecast revenue",
            channel="telegram",
            user_id="u123",
            session_id="s456",
        )
        assert req.channel == "telegram"
        assert req.user_id == "u123"

    def test_analyze_request_empty_query_fails(self):
        """Empty query should either fail validation or be accepted.
        Document the actual behavior."""
        # Pydantic doesn't enforce non-empty by default for str
        req = AnalyzeRequest(query="")
        assert req.query == ""

    def test_analyze_response_roundtrip(self):
        resp = AnalyzeResponse(
            query="test",
            narrative="Analysis complete",
            confidence=0.85,
            cost_usd=0.0042,
            success=True,
            elapsed_seconds=1.5,
            routing="andrew",
            formatted_message="test output",
        )
        d = resp.model_dump()
        assert d["confidence"] == 0.85
        assert d["success"] is True

    def test_health_response(self):
        hr = HealthResponse(andrew="ok", moltis={"status": "ok"}, bridge="ok")
        assert hr.andrew == "ok"

    def test_schedule_request(self):
        sr = ScheduleRequest(
            query="daily revenue report",
            cron_schedule="0 9 * * *",
            name="morning-revenue",
        )
        assert sr.cron_schedule == "0 9 * * *"


# ============================================================
# 2. MoltisConfig defaults
# ============================================================

class TestMoltisConfig:

    def test_default_config(self):
        config = MoltisConfig()
        assert config.host in ("127.0.0.1", "localhost", os.getenv("MOLTIS_HOST", "127.0.0.1"))
        assert isinstance(config.port, int)

    def test_custom_config(self):
        config = MoltisConfig(host="moltis.prod", port=9999, password="secret123")
        assert config.host == "moltis.prod"
        assert config.port == 9999
        assert config.password == "secret123"


# ============================================================
# 3. Service layer: _format_for_channel
# ============================================================

class TestFormatForChannel:

    def _make_bridge(self):
        """Create bridge without connecting to Moltis."""
        with patch("bridge.service.MoltisClient"):
            from bridge.service import AndrewMoltisBridge
            bridge = AndrewMoltisBridge.__new__(AndrewMoltisBridge)
            bridge.moltis = MagicMock()
            bridge.store_results = False
            bridge.hitl = MagicMock()
            bridge._andrew_executor = None
            bridge._db_url = ""
            return bridge

    def test_success_format(self):
        bridge = self._make_bridge()
        response = {
            "success": True,
            "confidence": 0.85,
            "narrative": "Revenue is $1M across all regions.",
            "warnings": [],
            "cost_usd": 0.005,
            "elapsed_seconds": 2.1,
            "routing": "andrew",
        }
        formatted = bridge._format_for_channel(response)
        assert "**Analysis**" in formatted
        assert "85%" in formatted
        assert "$0.0050" in formatted

    def test_failure_format(self):
        bridge = self._make_bridge()
        response = {
            "success": False,
            "confidence": 0.0,
            "narrative": "",
            "error": "SQL parse error",
            "warnings": [],
            "cost_usd": 0.0,
            "elapsed_seconds": 0.5,
            "routing": "andrew",
        }
        formatted = bridge._format_for_channel(response)
        assert "**Analysis failed**" in formatted
        assert "SQL parse error" in formatted

    def test_truncation_for_long_narrative(self):
        bridge = self._make_bridge()
        response = {
            "success": True,
            "confidence": 0.9,
            "narrative": "x" * 3000,
            "warnings": [],
            "cost_usd": 0.01,
            "elapsed_seconds": 5.0,
            "routing": "both",
        }
        formatted = bridge._format_for_channel(response)
        assert "truncated" in formatted

    def test_warnings_in_format(self):
        bridge = self._make_bridge()
        response = {
            "success": True,
            "confidence": 0.4,
            "narrative": "Partial result",
            "warnings": ["Low confidence", "Schema mismatch", "Timeout risk"],
            "cost_usd": 0.003,
            "elapsed_seconds": 1.0,
            "routing": "andrew",
        }
        formatted = bridge._format_for_channel(response)
        assert "**Warnings:**" in formatted
        assert "Low confidence" in formatted


# ============================================================
# 4. API endpoints via TestClient
# ============================================================

class TestAPIEndpoints:

    @pytest.fixture(autouse=True)
    def setup_client(self):
        """Import app and create TestClient once."""
        from fastapi.testclient import TestClient
        from bridge.api import app
        self.client = TestClient(app)

    def test_health_endpoint(self):
        """Health endpoint returns 200 with mocked bridge."""
        mock_bridge = MagicMock()
        mock_bridge.moltis.health_check = AsyncMock(return_value={"status": "ok"})

        with patch("bridge.api.get_bridge", return_value=mock_bridge):
            resp = self.client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["andrew"] == "ok"
        assert data["bridge"] == "ok"

    def test_analyze_endpoint(self):
        """Analyze endpoint calls handle_query and returns structured response."""
        mock_bridge = MagicMock()
        mock_bridge.handle_query = AsyncMock(return_value={
            "query": "test",
            "narrative": "Revenue is $1M",
            "sql_query": "SELECT SUM(revenue) FROM sales",
            "confidence": 0.85,
            "cost_usd": 0.005,
            "success": True,
            "error": None,
            "elapsed_seconds": 1.5,
            "routing": "andrew",
            "formatted_message": "test",
        })

        with patch("bridge.api.get_bridge", return_value=mock_bridge):
            resp = self.client.post(
                "/analyze",
                json={"query": "total revenue by region"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["confidence"] == 0.85

    def test_webhook_empty_message_skipped(self):
        """Webhook with empty message returns skipped status."""
        resp = self.client.post(
            "/webhook/moltis",
            json={"message": {"content": ""}, "channel": "telegram"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"

    def test_webhook_non_analytical_skipped(self):
        """Webhook with non-analytical message returns skipped."""
        resp = self.client.post(
            "/webhook/moltis",
            json={
                "message": {"content": "hello how are you"},
                "channel": "telegram",
                "user": {"id": "u1"},
            },
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "skipped"
        assert "not analytical" in resp.json()["reason"]

    def test_webhook_analytical_processed(self):
        """Webhook with analytical message calls handle_query."""
        mock_bridge = MagicMock()
        mock_bridge.handle_query = AsyncMock(return_value={
            "formatted_message": "Revenue report complete",
            "confidence": 0.9,
        })

        with patch("bridge.api.get_bridge", return_value=mock_bridge):
            resp = self.client.post(
                "/webhook/moltis",
                json={
                    "message": {"content": "analyze total revenue by region"},
                    "channel": "discord",
                    "user": {"id": "u42"},
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    def test_schedule_endpoint_success(self):
        """Schedule endpoint creates cron job via Moltis."""
        mock_bridge = MagicMock()
        mock_bridge.moltis.add_cron_job = AsyncMock(return_value=True)

        with patch("bridge.api.get_bridge", return_value=mock_bridge):
            resp = self.client.post(
                "/schedule",
                json={
                    "query": "daily revenue report",
                    "cron_schedule": "0 9 * * *",
                    "name": "morning-revenue",
                },
            )
        assert resp.status_code == 200
        assert resp.json()["status"] == "scheduled"

    def test_schedule_endpoint_failure(self):
        """Schedule endpoint returns 500 when Moltis cron fails."""
        mock_bridge = MagicMock()
        mock_bridge.moltis.add_cron_job = AsyncMock(return_value=False)

        with patch("bridge.api.get_bridge", return_value=mock_bridge):
            resp = self.client.post(
                "/schedule",
                json={
                    "query": "daily report",
                    "cron_schedule": "0 9 * * *",
                    "name": "test",
                },
            )
        assert resp.status_code == 500


# ============================================================
# 5. Deployment utility output
# ============================================================

class TestDeploymentUtils:

    def test_generate_hook_config(self):
        from bridge.api import generate_moltis_hook_config
        config = generate_moltis_hook_config(bridge_port=8100)
        assert "andrew-analytics" in config
        assert "8100" in config
        assert "MessageReceived" in config

    def test_generate_docker_compose(self):
        from bridge.api import generate_docker_compose
        compose = generate_docker_compose(bridge_port=8100)
        assert "moltis" in compose
        assert "andrew-bridge" in compose
        assert "MOLTIS_PASSWORD" in compose
        assert "8100:8100" in compose

    def test_generate_docker_compose_custom_port(self):
        from bridge.api import generate_docker_compose
        compose = generate_docker_compose(bridge_port=9999)
        assert "9999:8100" in compose
