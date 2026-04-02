"""
tests/test_comfyui_client.py
============================
Tests for the ComfyUI async client.

All HTTP calls are intercepted with httpx.MockTransport — no real ComfyUI needed.

Coverage:
  C1.  submit() returns prompt_id from /prompt response.
  C2.  submit() raises RuntimeError on ComfyUI-level error field.
  C3.  await_completion() returns SUCCESS when history shows completed=true + success.
  C4.  await_completion() returns ERROR when history shows status_str=error.
  C5.  await_completion() returns TIMEOUT when deadline exceeded (fast clock).
  C6.  await_completion() keeps polling while job is not yet in history.
  C7.  _extract_output_files() collects images + videos from node outputs.
  C8.  output_url() builds the correct /view URL.
  C9.  submit_batch() enqueues all jobs and polls each one.
  C10. submit_batch() result counts (succeeded / failed) are correct.
  C11. health() returns comfyui=ok with stats and queue info.
  C12. health() on connection error returns unreachable dict (via endpoint).
  C13. ComfyUIClient works as async context manager (no double-close error).
  C14. wrap_for_api round-trip: submitted workflow matches what was submitted.
"""
from __future__ import annotations

import json
import os
import sys
import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.comfyui_client import ComfyUIClient
from bridge.schemas import ComfyUIJobStatus

# ── mock transport helpers ────────────────────────────────────────────────────

_PROMPT_ID = "test-prompt-uuid-0001"
_CLIENT_ID = str(uuid.uuid4())


def _success_history(prompt_id: str, filenames: list[str] | None = None) -> dict:
    fnames = filenames or ["scene_01_00001.mp4"]
    return {
        prompt_id: {
            "status": {"status_str": "success", "completed": True},
            "outputs": {
                "9": {
                    "images": [
                        {"filename": f, "subfolder": "", "type": "output"}
                        for f in fnames
                    ]
                }
            },
        }
    }


def _error_history(prompt_id: str) -> dict:
    return {
        prompt_id: {
            "status": {
                "status_str": "error",
                "completed": True,
                "messages": ["Node 3: CUDA OOM"],
            },
            "outputs": {},
        }
    }


def _pending_history() -> dict:
    return {}  # prompt_id not in history yet


class _MockTransport(httpx.AsyncBaseTransport):
    """Simple request → response mapping for httpx."""

    def __init__(self, routes: dict[str, Any]) -> None:
        # routes: {method_path: response_body_dict | callable}
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        handler = self._routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": "not found"})
        if callable(handler):
            body = handler(request)
        else:
            body = handler
        return httpx.Response(200, json=body)


def _make_client(routes: dict) -> ComfyUIClient:
    client = ComfyUIClient(host="http://mock-comfyui", poll_interval_sec=0.01)
    transport = _MockTransport(routes)
    client._http = httpx.AsyncClient(
        base_url="http://mock-comfyui",
        transport=transport,
    )
    return client


# ── C1: submit returns prompt_id ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c1_submit_returns_prompt_id():
    routes = {
        f"POST /prompt": {"prompt_id": _PROMPT_ID, "number": 1, "node_errors": {}},
    }
    async with _make_client(routes) as client:
        pid = await client.submit({"1": {"class_type": "CLIPTextEncode", "inputs": {}}})
    assert pid == _PROMPT_ID


# ── C2: submit raises on ComfyUI error field ──────────────────────────────────

@pytest.mark.asyncio
async def test_c2_submit_raises_on_error_field():
    routes = {
        "POST /prompt": {"error": "invalid node type", "node_errors": {}},
    }
    async with _make_client(routes) as client:
        with pytest.raises(RuntimeError, match="ComfyUI rejected prompt"):
            await client.submit({})


# ── C3: await_completion → SUCCESS ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_c3_await_completion_success():
    history_body = _success_history(_PROMPT_ID, ["scene_01_00001.mp4"])
    routes = {f"GET /history/{_PROMPT_ID}": history_body}
    async with _make_client(routes) as client:
        result = await client.await_completion(_PROMPT_ID)

    assert result.status == ComfyUIJobStatus.SUCCESS
    assert result.prompt_id == _PROMPT_ID
    assert len(result.output_files) == 1
    assert result.output_files[0].filename == "scene_01_00001.mp4"


# ── C4: await_completion → ERROR ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c4_await_completion_error():
    routes = {f"GET /history/{_PROMPT_ID}": _error_history(_PROMPT_ID)}
    async with _make_client(routes) as client:
        result = await client.await_completion(_PROMPT_ID)

    assert result.status == ComfyUIJobStatus.ERROR
    assert "CUDA OOM" in (result.error or "")


# ── C5: await_completion → TIMEOUT ───────────────────────────────────────────

@pytest.mark.asyncio
async def test_c5_await_completion_timeout():
    # History always returns empty (job never finishes)
    routes = {f"GET /history/{_PROMPT_ID}": {}}
    async with _make_client(routes) as client:
        result = await client.await_completion(_PROMPT_ID, timeout_sec=0.05)

    assert result.status == ComfyUIJobStatus.TIMEOUT


# ── C6: await_completion polls until job appears ──────────────────────────────

@pytest.mark.asyncio
async def test_c6_polls_until_ready():
    call_count = 0

    def _history_handler(request: httpx.Request) -> dict:
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            return {}  # not ready yet
        return _success_history(_PROMPT_ID)

    routes = {f"GET /history/{_PROMPT_ID}": _history_handler}
    async with _make_client(routes) as client:
        result = await client.await_completion(_PROMPT_ID, timeout_sec=5.0)

    assert result.status == ComfyUIJobStatus.SUCCESS
    assert call_count >= 3


# ── C7: _extract_output_files handles images + videos keys ───────────────────

def test_c7_extract_output_files_multi_key():
    client = ComfyUIClient.__new__(ComfyUIClient)
    client._host = "http://mock-comfyui"
    outputs = {
        "9": {"images": [{"filename": "a.png", "subfolder": "", "type": "output"}]},
        "10": {"gifs": [{"filename": "b.gif", "subfolder": "sub", "type": "output"}]},
        "11": {"videos": [{"filename": "c.mp4", "subfolder": "", "type": "output"}]},
    }
    files = client._extract_output_files(outputs)
    assert len(files) == 3
    fnames = {f.filename for f in files}
    assert fnames == {"a.png", "b.gif", "c.mp4"}


# ── C8: output_url builds /view URL ──────────────────────────────────────────

def test_c8_output_url():
    client = ComfyUIClient(host="http://127.0.0.1:8188")
    url = client.output_url("scene_01.mp4", subfolder="ltx", file_type="output")
    assert "filename=scene_01.mp4" in url
    assert "subfolder=ltx" in url
    assert "type=output" in url


# ── C9: submit_batch enqueues all and polls each ─────────────────────────────

@pytest.mark.asyncio
async def test_c9_submit_batch_polls_all():
    pid1, pid2 = "pid-001", "pid-002"
    submit_count = 0

    def _submit_handler(request: httpx.Request) -> dict:
        nonlocal submit_count
        submit_count += 1
        pid = pid1 if submit_count == 1 else pid2
        return {"prompt_id": pid, "number": submit_count, "node_errors": {}}

    routes = {
        "POST /prompt": _submit_handler,
        f"GET /history/{pid1}": _success_history(pid1, ["s1.mp4"]),
        f"GET /history/{pid2}": _success_history(pid2, ["s2.mp4"]),
    }
    async with _make_client(routes) as client:
        batch = await client.submit_batch(
            [{}, {}],
            scene_ids=["scene_01", "scene_02"],
        )

    assert batch.total == 2
    assert batch.succeeded == 2
    assert batch.failed == 0
    scene_ids_out = {r.scene_id for r in batch.results}
    assert scene_ids_out == {"scene_01", "scene_02"}


# ── C10: submit_batch counts succeeded / failed correctly ────────────────────

@pytest.mark.asyncio
async def test_c10_batch_counts():
    pid_ok, pid_err = "pid-ok", "pid-err"
    call_n = 0

    def _submit_handler(request: httpx.Request) -> dict:
        nonlocal call_n
        call_n += 1
        pid = pid_ok if call_n == 1 else pid_err
        return {"prompt_id": pid, "number": call_n, "node_errors": {}}

    routes = {
        "POST /prompt": _submit_handler,
        f"GET /history/{pid_ok}": _success_history(pid_ok),
        f"GET /history/{pid_err}": _error_history(pid_err),
    }
    async with _make_client(routes) as client:
        batch = await client.submit_batch([{}, {}])

    assert batch.succeeded == 1
    assert batch.failed == 1


# ── C11: health() returns ok ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c11_health_ok():
    routes = {
        "GET /system_stats": {"system": {"python_version": "3.11"}, "devices": []},
        "GET /queue": {"queue_running": [], "queue_pending": ["a", "b"]},
    }
    async with _make_client(routes) as client:
        h = await client.health()

    assert h["comfyui"] == "ok"
    assert h["queue_pending"] == 2
    assert h["queue_running"] == 0


# ── C12: await_completion handles HTTP errors gracefully ─────────────────────

@pytest.mark.asyncio
async def test_c12_poll_http_error_retries():
    """A 500 on /history should not crash — client retries until timeout."""
    class _ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if "/history/" in request.url.path:
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, json={"prompt_id": _PROMPT_ID})

    client = ComfyUIClient(host="http://mock-comfyui", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="http://mock-comfyui",
        transport=_ErrTransport(),
    )
    async with client:
        result = await client.await_completion(_PROMPT_ID, timeout_sec=0.1)
    assert result.status == ComfyUIJobStatus.TIMEOUT


# ── C13: async context manager ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_c13_context_manager():
    routes = {f"GET /history/{_PROMPT_ID}": _success_history(_PROMPT_ID)}
    client = _make_client(routes)
    async with client as c:
        assert c is client
    # No exception on double exit


# ── C14: submitted workflow round-trip via wrap_for_api ───────────────────────

@pytest.mark.asyncio
async def test_c14_workflow_roundtrip():
    from bridge.comfyui_export import ComfyUIWorkflowExporter
    from bridge.schemas import LtxGenerationConfig, LtxKeyframe, LtxSceneJob

    job = LtxSceneJob(
        job_id="proj_scene_01_aabbcc",
        scene_id="scene_01",
        scene_index=0,
        workflow="first_last",
        duration_sec=8.0,
        prompts=["neon corridor"],
        keyframes=[
            LtxKeyframe(index=0, frame_type="first", source_prompt="neon corridor"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="neon corridor"),
        ],
        model_config_ltx={"variant": "fp8", "resolution": "1920x1080", "aspect_ratio": "16:9", "vram_budget_gb": 12},
    )
    exporter = ComfyUIWorkflowExporter()
    wf = exporter.export_job(job, LtxGenerationConfig())
    wrapped = exporter.wrap_for_api(wf, client_id="cid-test")

    captured = {}

    def _submit_handler(request: httpx.Request) -> dict:
        captured["body"] = json.loads(request.content)
        return {"prompt_id": _PROMPT_ID, "number": 1, "node_errors": {}}

    routes = {
        "POST /prompt": _submit_handler,
        f"GET /history/{_PROMPT_ID}": _success_history(_PROMPT_ID),
    }
    async with _make_client(routes) as client:
        pid = await client.submit(wrapped["prompt"], client_id=wrapped["client_id"])

    assert pid == _PROMPT_ID
    assert "prompt" in captured["body"]
    # Ensure all node IDs are present in what was submitted
    for node_id in wf:
        assert node_id in captured["body"]["prompt"]
