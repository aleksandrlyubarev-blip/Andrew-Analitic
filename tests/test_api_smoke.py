"""API smoke tests for the FastAPI bridge endpoints."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

from bridge import moltis_bridge
from core.romeo_phd import RomeoResult


class _StubMoltis:
    async def health_check(self):
        return {"status": "ok"}


class _StubBridge:
    def __init__(self):
        self.moltis = _StubMoltis()

    async def handle_query(self, query, context=None):
        return {
            "query": query,
            "narrative": "Smoke test narrative",
            "sql_query": "SELECT 1 AS ok",
            "query_results": [{"ok": 1}],
            "confidence": 0.91,
            "cost_usd": 0.0,
            "success": True,
            "error": None,
            "elapsed_seconds": 0.01,
            "routing": "standard",
            "formatted_message": "Smoke test narrative",
            "hitl_required": False,
            "hitl_reason": None,
            "session_id": "smoke-session",
            "session_length": 1,
            "memory_records_retrieved": 0,
            "data_profile": None,
        }


def test_health_endpoint_returns_bridge_and_moltis_status(monkeypatch):
    monkeypatch.setattr(moltis_bridge, "get_bridge", lambda: _StubBridge())

    with TestClient(moltis_bridge.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["andrew"] == "ok"
    assert body["bridge"] == "ok"
    assert body["moltis"]["status"] == "ok"


def test_metrics_endpoint_returns_expected_shape():
    with TestClient(moltis_bridge.app) as client:
        response = client.get("/metrics")

    assert response.status_code == 200
    body = response.json()
    assert "queries" in body
    assert "cost" in body
    assert "confidence" in body
    assert "routing_lanes" in body


def test_analyze_endpoint_returns_stubbed_result(monkeypatch):
    monkeypatch.setattr(moltis_bridge, "get_bridge", lambda: _StubBridge())

    with TestClient(moltis_bridge.app) as client:
        response = client.post("/analyze", json={"query": "Total revenue by region"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["query"] == "Total revenue by region"
    assert body["sql_query"] == "SELECT 1 AS ok"


def test_educate_endpoint_returns_stubbed_result(monkeypatch):
    def fake_execute(self, question: str):
        return RomeoResult(
            question=question,
            answer="**Smoke test** explanation",
            cost_usd=0.0,
            model="gpt-4o-mini",
            elapsed=0.01,
            success=True,
        )

    monkeypatch.setattr("core.romeo_phd.RomeoExecutor.execute", fake_execute)

    with TestClient(moltis_bridge.app) as client:
        response = client.post("/educate", json={"question": "What is gradient descent?"})

    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert body["question"] == "What is gradient descent?"
    assert "Smoke test" in body["answer"]
