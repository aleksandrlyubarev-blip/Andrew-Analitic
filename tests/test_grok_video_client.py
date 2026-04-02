"""
tests/test_grok_video_client.py
================================
Tests for GrokVideoClient and Grok 4.2 routing in VideoDispatcher.

G-series: GrokVideoClient unit tests (mocked httpx transport).
R-series: VideoDispatcher routing for grok4.2 model tag.

Coverage:
  G1.  submit() returns generation id from POST /v1/video/generations.
  G2.  submit() raises RuntimeError when id absent from response.
  G3.  await_completion() returns SUCCESS on status=completed + output.url.
  G4.  await_completion() returns SUCCESS on status=succeeded (alias).
  G5.  await_completion() returns ERROR on status=failed.
  G6.  await_completion() returns TIMEOUT when deadline exceeded.
  G7.  Pending statuses (queued/processing) keep polling.
  G8.  _extract_outputs() handles nested output.url field.
  G9.  _extract_outputs() handles top-level video_url field.
  G10. _extract_outputs() handles outputs list field.
  G11. _build_payload() includes camera_motion when set.
  G12. _build_payload() includes start/end frame URLs.
  G13. _build_payload() omits None/empty optional fields.
  G14. HTTP 500 on poll is retried until timeout.
  G15. Context manager enter/exit works cleanly.

  R1.  [MODEL: grok4.2] routes to GrokVideoClient, not Higgsfield or ComfyUI.
  R2.  [MODEL: grok-4.2] alias also routes to Grok.
  R3.  Mixed scenario: grok4.2 + sora2 + ltx2.3 all route correctly.
  R4.  Grok error captured in batch result (failed count increments).
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.grok_video_client import GrokVideoClient
from bridge.schemas import (
    ComfyUIJobStatus,
    ComfyUIOutputFile,
    ComfyUISubmitResult,
    GrokVideoRequest,
    LtxGenerationConfig,
    LtxKeyframe,
    LtxSceneJob,
)
from bridge.video_dispatcher import VideoDispatcher

# ── shared ────────────────────────────────────────────────────────────────────

_GEN_ID = "xai-gen-uuid-0001"

DEFAULT_REQ = GrokVideoRequest(
    scene_id="scene_05",
    prompt="extreme slow-motion katana vs chain, sparks frozen, Pixar style",
    duration_sec=15,
)


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, routes: dict) -> None:
        self._routes = routes

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = f"{request.method} {request.url.path}"
        handler = self._routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": "not found"})
        body = handler(request) if callable(handler) else handler
        return httpx.Response(200, json=body)


def _make_client(routes: dict) -> GrokVideoClient:
    client = GrokVideoClient(api_key="test-xai-key", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-xai.ai",
        transport=_MockTransport(routes),
    )
    return client


def _make_scene_job(model: str, scene_id: str = "scene_05", duration: float = 15.0) -> LtxSceneJob:
    return LtxSceneJob(
        job_id=f"proj_{scene_id}_xyz",
        scene_id=scene_id,
        scene_index=4,
        workflow="multi_keyframe",
        duration_sec=duration,
        prompts=["extreme slow-motion, Pixar style"],
        preferred_model=model,
        camera_motion="45-degree orbit around point of contact, creeping slow",
        keyframes=[
            LtxKeyframe(index=0, frame_type="first", source_prompt="p"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="p"),
        ],
        model_config_ltx={"variant": "fp8", "resolution": "1920x1080", "aspect_ratio": "16:9", "vram_budget_gb": 12},
    )


def _ok_response(gen_id: str = _GEN_ID, status: str = "completed") -> dict:
    return {"id": gen_id, "status": status, "output": {"url": f"https://cdn.x.ai/{gen_id}.mp4"}}


# ═══════════════════════════════════════════════════════════════════════════════
# G-series: GrokVideoClient
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_g1_submit_returns_gen_id():
    routes = {"POST /v1/video/generations": {"id": _GEN_ID, "status": "queued"}}
    async with _make_client(routes) as client:
        gen_id = await client.submit(DEFAULT_REQ)
    assert gen_id == _GEN_ID


@pytest.mark.asyncio
async def test_g2_submit_raises_without_id():
    routes = {"POST /v1/video/generations": {"status": "queued"}}
    async with _make_client(routes) as client:
        with pytest.raises(RuntimeError, match="no generation id"):
            await client.submit(DEFAULT_REQ)


@pytest.mark.asyncio
async def test_g3_await_completion_success_completed():
    routes = {f"GET /v1/video/generations/{_GEN_ID}": _ok_response(status="completed")}
    async with _make_client(routes) as client:
        result = await client.await_completion(_GEN_ID, scene_id="scene_05")
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert result.scene_id == "scene_05"
    assert len(result.output_files) == 1
    assert f"{_GEN_ID}.mp4" in result.output_files[0].filename


@pytest.mark.asyncio
async def test_g4_await_completion_success_succeeded():
    routes = {f"GET /v1/video/generations/{_GEN_ID}": _ok_response(status="succeeded")}
    async with _make_client(routes) as client:
        result = await client.await_completion(_GEN_ID)
    assert result.status == ComfyUIJobStatus.SUCCESS


@pytest.mark.asyncio
async def test_g5_await_completion_error():
    routes = {f"GET /v1/video/generations/{_GEN_ID}": {
        "id": _GEN_ID, "status": "failed", "error": "NSFW content detected"
    }}
    async with _make_client(routes) as client:
        result = await client.await_completion(_GEN_ID)
    assert result.status == ComfyUIJobStatus.ERROR
    assert "NSFW" in (result.error or "")


@pytest.mark.asyncio
async def test_g6_await_completion_timeout():
    routes = {f"GET /v1/video/generations/{_GEN_ID}": {"id": _GEN_ID, "status": "processing"}}
    async with _make_client(routes) as client:
        result = await client.await_completion(_GEN_ID, timeout_sec=0.05)
    assert result.status == ComfyUIJobStatus.TIMEOUT


@pytest.mark.asyncio
async def test_g7_pending_statuses_keep_polling():
    call_n = 0

    def _handler(request: httpx.Request) -> dict:
        nonlocal call_n
        call_n += 1
        if call_n == 1:
            return {"id": _GEN_ID, "status": "queued"}
        if call_n == 2:
            return {"id": _GEN_ID, "status": "processing"}
        return _ok_response()

    routes = {f"GET /v1/video/generations/{_GEN_ID}": _handler}
    async with _make_client(routes) as client:
        result = await client.await_completion(_GEN_ID, timeout_sec=5.0)
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert call_n == 3


def test_g8_extract_outputs_nested_output():
    client = GrokVideoClient.__new__(GrokVideoClient)
    data = {"status": "completed", "output": {"url": "https://cdn.x.ai/scene05.mp4"}}
    files = client._extract_outputs(_GEN_ID, data)
    assert len(files) == 1
    assert files[0].filename == "scene05.mp4"
    assert files[0].url == "https://cdn.x.ai/scene05.mp4"
    assert files[0].node_id == "grok_video"


def test_g9_extract_outputs_top_level_video_url():
    client = GrokVideoClient.__new__(GrokVideoClient)
    data = {"status": "completed", "video_url": "https://cdn.x.ai/clip.mp4"}
    files = client._extract_outputs(_GEN_ID, data)
    assert len(files) == 1
    assert files[0].filename == "clip.mp4"


def test_g10_extract_outputs_list():
    client = GrokVideoClient.__new__(GrokVideoClient)
    data = {
        "status": "completed",
        "outputs": [
            {"url": "https://cdn.x.ai/a.mp4", "type": "video"},
            {"url": "https://cdn.x.ai/b.mp4", "type": "video"},
        ],
    }
    files = client._extract_outputs(_GEN_ID, data)
    assert len(files) == 2
    assert {f.filename for f in files} == {"a.mp4", "b.mp4"}


def test_g11_build_payload_camera_motion():
    client = GrokVideoClient(api_key="x")
    req = GrokVideoRequest(prompt="test", camera_motion="45-degree orbit")
    payload = client._build_payload(req)
    assert payload["camera_motion"] == "45-degree orbit"


def test_g12_build_payload_start_end_frames():
    client = GrokVideoClient(api_key="x")
    req = GrokVideoRequest(
        prompt="test",
        start_frame_url="https://imgs/first.png",
        end_frame_url="https://imgs/last.png",
    )
    payload = client._build_payload(req)
    assert payload["start_frame_url"] == "https://imgs/first.png"
    assert payload["end_frame_url"] == "https://imgs/last.png"


def test_g13_build_payload_omits_none_fields():
    client = GrokVideoClient(api_key="x")
    req = GrokVideoRequest(prompt="test")  # no optional fields
    payload = client._build_payload(req)
    assert "camera_motion" not in payload
    assert "start_frame_url" not in payload
    assert "end_frame_url" not in payload
    assert "seed" not in payload


@pytest.mark.asyncio
async def test_g14_http_500_retried_until_timeout():
    class _ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if request.method == "GET":
                return httpx.Response(500, text="Internal Server Error")
            return httpx.Response(200, json={"id": _GEN_ID})

    client = GrokVideoClient(api_key="x", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-xai.ai", transport=_ErrTransport()
    )
    async with client:
        result = await client.await_completion(_GEN_ID, timeout_sec=0.1)
    assert result.status == ComfyUIJobStatus.TIMEOUT


@pytest.mark.asyncio
async def test_g15_context_manager():
    routes = {f"GET /v1/video/generations/{_GEN_ID}": _ok_response()}
    client = _make_client(routes)
    async with client as c:
        assert c is client  # no exception


# ═══════════════════════════════════════════════════════════════════════════════
# R-series: dispatcher routing for grok4.2
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_grok_client(scene_id: str = "scene_05") -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="xai-pid",
        scene_id=scene_id,
        status=ComfyUIJobStatus.SUCCESS,
        output_files=[ComfyUIOutputFile(node_id="grok_video", filename="out.mp4", url="https://cdn.x.ai/out.mp4")],
    ))
    return mock


def _mock_hf_client() -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="hf-pid", scene_id="", status=ComfyUIJobStatus.SUCCESS, output_files=[],
    ))
    return mock


def _mock_cu_client() -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="cu-pid", scene_id="", status=ComfyUIJobStatus.SUCCESS, output_files=[],
    ))
    return mock


@pytest.mark.asyncio
async def test_r1_grok42_routes_to_grok():
    from unittest.mock import patch
    job = _make_scene_job("grok4.2", "scene_05")
    grok_mock = _mock_grok_client("scene_05")
    hf_mock = _mock_hf_client()
    cu_mock = _mock_cu_client()

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xai-key")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    grok_mock.submit_and_await.assert_called_once()
    hf_mock.submit_and_await.assert_not_called()
    cu_mock.submit_and_await.assert_not_called()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_r2_grok_dash_alias_routes_to_grok():
    from unittest.mock import patch
    job = _make_scene_job("grok-4.2", "scene_05")
    grok_mock = _mock_grok_client("scene_05")
    hf_mock = _mock_hf_client()
    cu_mock = _mock_cu_client()

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xai-key")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    grok_mock.submit_and_await.assert_called_once()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_r3_mixed_grok_higgsfield_comfyui():
    from unittest.mock import patch
    jobs = [
        _make_scene_job("grok4.2", "scene_05"),
        _make_scene_job("sora2",   "scene_01"),
        _make_scene_job("",        "scene_02"),
    ]
    grok_calls, hf_calls, cu_calls = [], [], []

    async def _grok_side(req, **_):
        grok_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="g", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS, output_files=[],
        )

    async def _hf_side(req, **_):
        hf_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="h", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS, output_files=[],
        )

    async def _cu_side(workflow, scene_id="", **_):
        cu_calls.append(scene_id)
        return ComfyUISubmitResult(
            prompt_id="c", scene_id=scene_id,
            status=ComfyUIJobStatus.SUCCESS, output_files=[],
        )

    grok_mock = AsyncMock()
    grok_mock.__aenter__ = AsyncMock(return_value=grok_mock)
    grok_mock.__aexit__ = AsyncMock(return_value=False)
    grok_mock.submit_and_await = AsyncMock(side_effect=_grok_side)

    hf_mock = AsyncMock()
    hf_mock.__aenter__ = AsyncMock(return_value=hf_mock)
    hf_mock.__aexit__ = AsyncMock(return_value=False)
    hf_mock.submit_and_await = AsyncMock(side_effect=_hf_side)

    cu_mock = AsyncMock()
    cu_mock.__aenter__ = AsyncMock(return_value=cu_mock)
    cu_mock.__aexit__ = AsyncMock(return_value=False)
    cu_mock.submit_and_await = AsyncMock(side_effect=_cu_side)

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xai-key", higgsfield_api_key="hf-key")
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert grok_calls == ["scene_05"]
    assert hf_calls == ["scene_01"]
    assert cu_calls == ["scene_02"]
    assert result.total == 3
    assert result.succeeded == 3


@pytest.mark.asyncio
async def test_r4_grok_error_in_batch():
    from unittest.mock import patch
    job = _make_scene_job("grok4.2", "scene_05")

    grok_mock = AsyncMock()
    grok_mock.__aenter__ = AsyncMock(return_value=grok_mock)
    grok_mock.__aexit__ = AsyncMock(return_value=False)
    grok_mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="x", scene_id="scene_05",
        status=ComfyUIJobStatus.ERROR, error="rate limit exceeded",
    ))
    hf_mock = _mock_hf_client()
    cu_mock = _mock_cu_client()

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xai-key")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert result.succeeded == 0
    assert "rate limit" in (result.results[0].error or "")
