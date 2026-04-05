"""
tests/test_veo_client.py
========================
Tests for VeoVideoClient (Google Veo 3.1 LRO API) and Veo routing in dispatcher.

V-series: VeoVideoClient unit tests (mocked httpx transport).
W-series: VideoDispatcher routing for veo3.1 — no Higgsfield dependency.

Coverage:
  V1.  submit() returns operation name from POST response.
  V2.  submit() raises RuntimeError when name absent.
  V3.  await_completion() returns SUCCESS when done=true + generatedSamples.
  V4.  await_completion() returns SUCCESS with flat videos[] fallback format.
  V5.  await_completion() returns ERROR when done=true + error field.
  V6.  await_completion() returns TIMEOUT when deadline exceeded.
  V7.  done=false keeps polling.
  V8.  _extract_outputs() handles generateVideoResponse.generatedSamples.
  V9.  _extract_outputs() handles flat response.videos[] format.
  V10. _build_payload() includes cameraControl when camera_motion set.
  V11. _build_payload() includes lastFrame when end_frame_url set.
  V12. _build_payload() includes referenceImages for multi-ref.
  V13. _build_payload() omits optional fields when not set.
  V14. HTTP 500 on poll is retried until timeout.
  V15. Context manager works cleanly.

  W1.  veo3.1 routes to VeoVideoClient, not GrokVideoClient or ComfyUI.
  W2.  veo-3.1 alias also routes to Veo.
  W3.  Mixed: veo3.1 + grok4.2 + ltx2.3 all route correctly.
  W4.  Veo error captured in batch result.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.veo_client import VeoVideoClient, VEO_OPS_PREFIX, VEO_ENDPOINT
from bridge.schemas import (
    ComfyUIJobStatus,
    ComfyUIOutputFile,
    ComfyUISubmitResult,
    LtxGenerationConfig,
    LtxKeyframe,
    LtxSceneJob,
    VeoVideoRequest,
)
from bridge.video_dispatcher import VideoDispatcher

# ── shared ────────────────────────────────────────────────────────────────────

_OP_NAME = "operations/veo-op-uuid-001"
_OP_ID = "veo-op-uuid-001"
_VIDEO_URL = "https://storage.googleapis.com/veo-output/scene01.mp4"

DEFAULT_REQ = VeoVideoRequest(
    scene_id="scene_01",
    prompt="extreme close-up of Pinoblanco's glowing eyes, Pixar style",
    duration_sec=5,
)

_SUCCESS_RESPONSE = {
    "name": _OP_NAME,
    "done": True,
    "response": {
        "generateVideoResponse": {
            "generatedSamples": [
                {"video": {"uri": _VIDEO_URL}}
            ]
        }
    },
}

_ERROR_RESPONSE = {
    "name": _OP_NAME,
    "done": True,
    "error": {"code": 429, "message": "Quota exceeded for Veo 3.1"},
}

_PENDING_RESPONSE = {"name": _OP_NAME, "done": False}


class _MockTransport(httpx.AsyncBaseTransport):
    def __init__(self, routes: dict) -> None:
        self._routes = routes

    async def handle_async_request(self, req: httpx.Request) -> httpx.Response:
        key = f"{req.method} {req.url.path}"
        handler = self._routes.get(key)
        if handler is None:
            return httpx.Response(404, json={"error": "not found"})
        body = handler(req) if callable(handler) else handler
        return httpx.Response(200, json=body)


def _make_client(routes: dict) -> VeoVideoClient:
    client = VeoVideoClient(api_key="test-google-key", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-gai.googleapis.com",
        transport=_MockTransport(routes),
    )
    return client


def _make_scene_job(model: str, scene_id: str = "scene_01") -> LtxSceneJob:
    return LtxSceneJob(
        job_id=f"proj_{scene_id}_abc",
        scene_id=scene_id,
        scene_index=0,
        workflow="first_last",
        duration_sec=5.0,
        prompts=["Pixar close-up, moonlit night"],
        preferred_model=model,
        camera_motion="slow push-in zoom",
        keyframes=[
            LtxKeyframe(index=0, frame_type="first", source_prompt="p"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="p"),
        ],
        model_config_ltx={"variant": "fp8", "resolution": "1920x1080", "aspect_ratio": "16:9", "vram_budget_gb": 12},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# V-series: VeoVideoClient
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_v1_submit_returns_op_name():
    routes = {f"POST {VEO_ENDPOINT}": {"name": _OP_NAME, "done": False}}
    async with _make_client(routes) as client:
        op = await client.submit(DEFAULT_REQ)
    assert op == _OP_NAME


@pytest.mark.asyncio
async def test_v2_submit_raises_without_name():
    routes = {f"POST {VEO_ENDPOINT}": {"done": False}}
    async with _make_client(routes) as client:
        with pytest.raises(RuntimeError, match="operation name"):
            await client.submit(DEFAULT_REQ)


@pytest.mark.asyncio
async def test_v3_await_completion_success_generated_samples():
    routes = {f"GET {VEO_OPS_PREFIX}{_OP_ID}": _SUCCESS_RESPONSE}
    async with _make_client(routes) as client:
        result = await client.await_completion(_OP_NAME, scene_id="scene_01")
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert result.scene_id == "scene_01"
    assert len(result.output_files) == 1
    assert result.output_files[0].url == _VIDEO_URL
    assert result.output_files[0].node_id == "veo3.1"


@pytest.mark.asyncio
async def test_v4_await_completion_flat_videos_format():
    flat_response = {
        "name": _OP_NAME,
        "done": True,
        "response": {
            "videos": [{"uri": "https://storage.googleapis.com/veo/flat.mp4"}]
        },
    }
    routes = {f"GET {VEO_OPS_PREFIX}{_OP_ID}": flat_response}
    async with _make_client(routes) as client:
        result = await client.await_completion(_OP_NAME)
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert "flat.mp4" in result.output_files[0].filename


@pytest.mark.asyncio
async def test_v5_await_completion_error():
    routes = {f"GET {VEO_OPS_PREFIX}{_OP_ID}": _ERROR_RESPONSE}
    async with _make_client(routes) as client:
        result = await client.await_completion(_OP_NAME)
    assert result.status == ComfyUIJobStatus.ERROR
    assert "Quota exceeded" in (result.error or "")


@pytest.mark.asyncio
async def test_v6_await_completion_timeout():
    routes = {f"GET {VEO_OPS_PREFIX}{_OP_ID}": _PENDING_RESPONSE}
    async with _make_client(routes) as client:
        result = await client.await_completion(_OP_NAME, timeout_sec=0.05)
    assert result.status == ComfyUIJobStatus.TIMEOUT


@pytest.mark.asyncio
async def test_v7_done_false_keeps_polling():
    call_n = 0

    def _handler(req: httpx.Request) -> dict:
        nonlocal call_n
        call_n += 1
        if call_n < 3:
            return _PENDING_RESPONSE
        return _SUCCESS_RESPONSE

    routes = {f"GET {VEO_OPS_PREFIX}{_OP_ID}": _handler}
    async with _make_client(routes) as client:
        result = await client.await_completion(_OP_NAME, timeout_sec=5.0)
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert call_n == 3


def test_v8_extract_generated_samples():
    client = VeoVideoClient.__new__(VeoVideoClient)
    files = client._extract_outputs(_OP_NAME, _SUCCESS_RESPONSE)
    assert len(files) == 1
    assert files[0].filename == "scene01.mp4"
    assert files[0].node_id == "veo3.1"


def test_v9_extract_flat_videos():
    client = VeoVideoClient.__new__(VeoVideoClient)
    data = {
        "done": True,
        "response": {"videos": [
            {"uri": "https://storage.googleapis.com/veo/a.mp4"},
            {"uri": "https://storage.googleapis.com/veo/b.mp4"},
        ]},
    }
    files = client._extract_outputs(_OP_NAME, data)
    assert len(files) == 2
    assert {f.filename for f in files} == {"a.mp4", "b.mp4"}


def test_v10_build_payload_camera_control():
    client = VeoVideoClient(api_key="x")
    req = VeoVideoRequest(prompt="test", camera_motion="45-degree orbit")
    payload = client._build_payload(req)
    assert payload["parameters"]["cameraControl"] == "45-degree orbit"


def test_v11_build_payload_last_frame():
    client = VeoVideoClient(api_key="x")
    req = VeoVideoRequest(prompt="test", end_frame_url="gs://bucket/last.png")
    payload = client._build_payload(req)
    assert payload["parameters"]["lastFrame"] == {"uri": "gs://bucket/last.png"}


def test_v12_build_payload_reference_images():
    client = VeoVideoClient(api_key="x")
    req = VeoVideoRequest(
        prompt="test",
        reference_image_urls=["https://imgs/ref1.png", "https://imgs/ref2.png"],
    )
    payload = client._build_payload(req)
    refs = payload["parameters"]["referenceImages"]
    assert len(refs) == 2
    assert refs[0]["referenceType"] == "SUBJECT"


def test_v13_build_payload_omits_empty_optionals():
    client = VeoVideoClient(api_key="x")
    req = VeoVideoRequest(prompt="test")
    payload = client._build_payload(req)
    assert "cameraControl" not in payload["parameters"]
    assert "lastFrame" not in payload["parameters"]
    assert "referenceImages" not in payload["parameters"]
    assert "seed" not in payload["parameters"]


@pytest.mark.asyncio
async def test_v14_http_500_retried_until_timeout():
    class _ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, req: httpx.Request) -> httpx.Response:
            if req.method == "GET":
                return httpx.Response(500, text="Server Error")
            return httpx.Response(200, json={"name": _OP_NAME})

    client = VeoVideoClient(api_key="x", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-gai.ai", transport=_ErrTransport()
    )
    async with client:
        result = await client.await_completion(_OP_NAME, timeout_sec=0.1)
    assert result.status == ComfyUIJobStatus.TIMEOUT


@pytest.mark.asyncio
async def test_v15_context_manager():
    routes = {f"GET {VEO_OPS_PREFIX}{_OP_ID}": _SUCCESS_RESPONSE}
    client = _make_client(routes)
    async with client as c:
        assert c is client


# ═══════════════════════════════════════════════════════════════════════════════
# W-series: dispatcher routing for veo3.1
# ═══════════════════════════════════════════════════════════════════════════════

def _mock_veo_client(scene_id: str = "scene_01") -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id=_OP_NAME, scene_id=scene_id,
        status=ComfyUIJobStatus.SUCCESS,
        output_files=[ComfyUIOutputFile(node_id="veo3.1", filename="out.mp4", url=_VIDEO_URL)],
    ))
    return mock


def _mock_grok_client(scene_id: str = "scene_05") -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="xai-pid", scene_id=scene_id,
        status=ComfyUIJobStatus.SUCCESS, output_files=[],
    ))
    return mock


def _mock_cu_client(scene_id: str = "scene_06") -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="cu-pid", scene_id=scene_id,
        status=ComfyUIJobStatus.SUCCESS, output_files=[],
    ))
    return mock


@pytest.mark.asyncio
async def test_w1_veo31_routes_to_veo():
    from unittest.mock import patch
    job = _make_scene_job("veo3.1", "scene_01")
    veo_mock = _mock_veo_client("scene_01")
    grok_mock = _mock_grok_client()
    cu_mock = _mock_cu_client()

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    veo_mock.submit_and_await.assert_called_once()
    grok_mock.submit_and_await.assert_not_called()
    cu_mock.submit_and_await.assert_not_called()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_w2_veo_dash_alias_routes_to_veo():
    from unittest.mock import patch
    job = _make_scene_job("veo-3.1", "scene_01")
    veo_mock = _mock_veo_client("scene_01")
    grok_mock = _mock_grok_client()
    cu_mock = _mock_cu_client()

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    veo_mock.submit_and_await.assert_called_once()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_w3_mixed_veo_grok_comfyui():
    from unittest.mock import patch
    jobs = [
        _make_scene_job("veo3.1",  "scene_01"),
        _make_scene_job("grok4.2", "scene_05"),
        _make_scene_job("",        "scene_06"),
    ]
    veo_calls, grok_calls, cu_calls = [], [], []

    async def _veo_side(req, **_):
        veo_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="v", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS, output_files=[],
        )

    async def _grok_side(req, **_):
        grok_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="g", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS, output_files=[],
        )

    async def _cu_side(workflow, scene_id="", **_):
        cu_calls.append(scene_id)
        return ComfyUISubmitResult(
            prompt_id="c", scene_id=scene_id,
            status=ComfyUIJobStatus.SUCCESS, output_files=[],
        )

    veo_mock = AsyncMock()
    veo_mock.__aenter__ = AsyncMock(return_value=veo_mock)
    veo_mock.__aexit__ = AsyncMock(return_value=False)
    veo_mock.submit_and_await = AsyncMock(side_effect=_veo_side)

    grok_mock = AsyncMock()
    grok_mock.__aenter__ = AsyncMock(return_value=grok_mock)
    grok_mock.__aexit__ = AsyncMock(return_value=False)
    grok_mock.submit_and_await = AsyncMock(side_effect=_grok_side)

    cu_mock = AsyncMock()
    cu_mock.__aenter__ = AsyncMock(return_value=cu_mock)
    cu_mock.__aexit__ = AsyncMock(return_value=False)
    cu_mock.submit_and_await = AsyncMock(side_effect=_cu_side)

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", xai_api_key="xkey")
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert veo_calls == ["scene_01"]
    assert grok_calls == ["scene_05"]
    assert cu_calls == ["scene_06"]
    assert result.total == 3
    assert result.succeeded == 3


@pytest.mark.asyncio
async def test_w4_veo_error_in_batch():
    from unittest.mock import patch
    job = _make_scene_job("veo3.1", "scene_01")

    veo_mock = AsyncMock()
    veo_mock.__aenter__ = AsyncMock(return_value=veo_mock)
    veo_mock.__aexit__ = AsyncMock(return_value=False)
    veo_mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="x", scene_id="scene_01",
        status=ComfyUIJobStatus.ERROR, error="safety filter triggered",
    ))
    grok_mock = _mock_grok_client()
    cu_mock = _mock_cu_client()

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=False)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert "safety filter" in (result.results[0].error or "")
