"""
tests/test_higgsfield_client.py + test_video_dispatcher.py (combined)
======================================================================
Tests for HiggsfieldClient (new platform.higgsfield.ai API) and VideoDispatcher.

H-series: HiggsfieldClient unit tests (mocked httpx transport).
  H1.  submit() returns request_id from POST /{model_slug} response.
  H2.  submit() raises RuntimeError when request_id absent.
  H3.  await_completion() returns SUCCESS on status=completed + video URL.
  H4.  await_completion() returns ERROR on status=failed.
  H5.  await_completion() returns TIMEOUT when deadline exceeded.
  H6.  Unknown status keeps polling until a terminal status arrives.
  H7.  _extract_outputs() handles "video" field.
  H8.  _extract_outputs() handles "output_url" field.
  H9.  _build_payload() maps image_url from start_frame_url.
  H10. _build_payload() maps end_image_url from end_frame_url.
  H11. _build_payload() includes camera_motion when set.
  H12. MODEL_MAP maps friendly tags → Higgsfield app slugs.
  H13. HTTP 500 on /requests/{id}/status is retried until timeout.

D-series: VideoDispatcher routing / integration tests (all backends mocked).
  D1.  veo3.1 scene → routed to Veo.
  D2.  ltx2.3 scene → routed to ComfyUI.
  D3.  empty preferred_model → routed to ComfyUI.
  D4.  Mixed batch: veo3.1+grok4.2+ltx2.3+empty routes correctly.
  D5.  dispatch_scenario() parses scenario and routes correctly.
  D6.  Veo error → captured in result (status=ERROR, batch counts update).
  D7.  dispatch_batch() result counts (succeeded / failed) are accurate.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.higgsfield_client import HiggsfieldClient, MODEL_MAP
from bridge.schemas import (
    ComfyUIJobStatus,
    HiggsfieldGenerationRequest,
    LtxGenerationConfig,
    LtxKeyframe,
    LtxSceneJob,
    VideoDispatchRequest,
)
from bridge.video_dispatcher import VideoDispatcher, dispatch_scenario

# ── shared fixtures ───────────────────────────────────────────────────────────

_REQ_ID  = "d7e6c0f3-6699-4f6c-bb45-2ad7fd9158ff"
_MODEL   = "dop"
_SLUG    = "higgsfield-ai/dop/standard"
_VIDEO_URL = "https://cdn.higgsfield.ai/out/clip.mp4"

DEFAULT_REQ = HiggsfieldGenerationRequest(
    scene_id="scene_01",
    model=_MODEL,
    prompt="cinematic close-up of Pinoblanco's glowing eyes",
    duration_sec=5,
    start_frame_url="https://imgs/frame.png",
)

# New platform API response shapes
def _submit_ok(req_id: str = _REQ_ID) -> dict:
    base = f"https://platform.higgsfield.ai/requests/{req_id}"
    return {
        "status": "queued",
        "request_id": req_id,
        "status_url": f"{base}/status",
        "cancel_url": f"{base}/cancel",
    }


def _status_success(req_id: str = _REQ_ID, url: str = _VIDEO_URL) -> dict:
    return {
        "status": "completed",
        "request_id": req_id,
        "video": {"url": url},
    }


def _status_failed(req_id: str = _REQ_ID) -> dict:
    return {"status": "failed", "request_id": req_id, "error": "CUDA OOM on cloud node"}


def _status_queued(req_id: str = _REQ_ID) -> dict:
    return {"status": "queued", "request_id": req_id}


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


def _make_hf_client(routes: dict) -> HiggsfieldClient:
    client = HiggsfieldClient(hf_key="test-id:test-secret", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-higgsfield.ai",
        transport=_MockTransport(routes),
    )
    return client


def _make_scene_job(
    model: str = "sora2",
    scene_id: str = "scene_01",
    duration: float = 5.0,
) -> LtxSceneJob:
    return LtxSceneJob(
        job_id=f"proj_{scene_id}_abc",
        scene_id=scene_id,
        scene_index=0,
        workflow="first_last",
        duration_sec=duration,
        prompts=["test prompt"],
        preferred_model=model,
        keyframes=[
            LtxKeyframe(index=0, frame_type="first", source_prompt="test"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="test"),
        ],
        model_config_ltx={"variant": "fp8", "resolution": "1920x1080", "aspect_ratio": "16:9", "vram_budget_gb": 12},
    )


# ═══════════════════════════════════════════════════════════════════════════════
# H-series: HiggsfieldClient
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio
async def test_h1_submit_returns_request_id():
    routes = {f"POST /{_SLUG}": _submit_ok()}
    async with _make_hf_client(routes) as client:
        req_id = await client.submit(DEFAULT_REQ)
    assert req_id == _REQ_ID


@pytest.mark.asyncio
async def test_h2_submit_raises_without_request_id():
    routes = {f"POST /{_SLUG}": {"status": "queued"}}  # missing request_id
    async with _make_hf_client(routes) as client:
        with pytest.raises(RuntimeError, match="no request_id"):
            await client.submit(DEFAULT_REQ)


@pytest.mark.asyncio
async def test_h3_await_completion_success():
    routes = {f"GET /requests/{_REQ_ID}/status": _status_success()}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_REQ_ID, scene_id="scene_01")
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert result.scene_id == "scene_01"
    assert len(result.output_files) == 1
    assert "clip.mp4" in result.output_files[0].filename


@pytest.mark.asyncio
async def test_h4_await_completion_error():
    routes = {f"GET /requests/{_REQ_ID}/status": _status_failed()}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_REQ_ID)
    assert result.status == ComfyUIJobStatus.ERROR
    assert "CUDA OOM" in (result.error or "")


@pytest.mark.asyncio
async def test_h5_await_completion_timeout():
    routes = {f"GET /requests/{_REQ_ID}/status": _status_queued()}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_REQ_ID, timeout_sec=0.05)
    assert result.status == ComfyUIJobStatus.TIMEOUT


@pytest.mark.asyncio
async def test_h6_unknown_status_keeps_polling():
    call_n = 0

    def _handler(request: httpx.Request) -> dict:
        nonlocal call_n
        call_n += 1
        if call_n < 3:
            return {"status": "in_progress", "request_id": _REQ_ID}
        return _status_success()

    routes = {f"GET /requests/{_REQ_ID}/status": _handler}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_REQ_ID, timeout_sec=5.0)
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert call_n >= 3


def test_h7_extract_outputs_video_field():
    client = HiggsfieldClient.__new__(HiggsfieldClient)
    data = {"status": "completed", "video": {"url": "https://cdn.higgsfield.ai/scene01.mp4"}}
    files = client._extract_outputs("req1", data)
    assert len(files) == 1
    assert files[0].filename == "scene01.mp4"
    assert files[0].url == "https://cdn.higgsfield.ai/scene01.mp4"


def test_h8_extract_outputs_fallback_output_url():
    client = HiggsfieldClient.__new__(HiggsfieldClient)
    data = {"status": "completed", "output_url": "https://cdn.higgsfield.ai/scene01.mp4"}
    files = client._extract_outputs("req1", data)
    assert len(files) == 1
    assert files[0].url == "https://cdn.higgsfield.ai/scene01.mp4"


def test_h9_build_payload_image_url():
    client = HiggsfieldClient(hf_key="k:s")
    req = HiggsfieldGenerationRequest(
        model="dop",
        prompt="test",
        start_frame_url="https://imgs/first.png",
    )
    payload = client._build_payload(req)
    assert payload["image_url"] == "https://imgs/first.png"


def test_h10_build_payload_end_image_url():
    client = HiggsfieldClient(hf_key="k:s")
    req = HiggsfieldGenerationRequest(
        model="dop",
        prompt="test",
        start_frame_url="https://imgs/first.png",
        end_frame_url="https://imgs/last.png",
    )
    payload = client._build_payload(req)
    assert payload["end_image_url"] == "https://imgs/last.png"


def test_h11_build_payload_camera_motion():
    client = HiggsfieldClient(hf_key="k:s")
    req = HiggsfieldGenerationRequest(
        model="dop", prompt="test", camera_motion="slow push-in"
    )
    payload = client._build_payload(req)
    assert payload["camera_motion"] == "slow push-in"


def test_h12_model_map():
    assert MODEL_MAP["dop"]          == "higgsfield-ai/dop/standard"
    assert MODEL_MAP["dop-lite"]     == "higgsfield-ai/dop/lite"
    assert MODEL_MAP["dop-turbo"]    == "higgsfield-ai/dop/turbo"
    assert MODEL_MAP["kling2.1"]     == "kling-video/v2.1/standard/image-to-video"
    assert MODEL_MAP["seedance"]     == "bytedance/seedance/v1/lite/image-to-video"


@pytest.mark.asyncio
async def test_h13_http_500_retried_until_timeout():
    class _ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if "/requests/" in request.url.path and request.method == "GET":
                return httpx.Response(500, text="Server Error")
            return httpx.Response(200, json=_submit_ok())

    client = HiggsfieldClient(hf_key="k:s", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-hf.ai", transport=_ErrTransport()
    )
    async with client:
        result = await client.await_completion(_REQ_ID, timeout_sec=0.1)
    assert result.status == ComfyUIJobStatus.TIMEOUT


# ═══════════════════════════════════════════════════════════════════════════════
# D-series: VideoDispatcher (Veo 3.1 / Grok 4.2 / ComfyUI routing)
# ═══════════════════════════════════════════════════════════════════════════════

# We patch all three inner clients so no real HTTP calls are made.

def _make_mock_veo_client(status=ComfyUIJobStatus.SUCCESS, scene_id="scene_01"):
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="veo-op-id",
        scene_id=scene_id,
        status=status,
        output_files=[ComfyUIOutputFile(node_id="veo3.1", filename="out.mp4", url="https://veo.googleapis.com/out.mp4")],
    ))
    return mock


def _make_mock_grok_client(status=ComfyUIJobStatus.SUCCESS, scene_id="scene_02"):
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="xai-pid",
        scene_id=scene_id,
        status=status,
        output_files=[ComfyUIOutputFile(node_id="grok_video", filename="out.mp4", url="https://cdn.x.ai/out.mp4")],
    ))
    return mock


def _make_mock_cu_client(status=ComfyUIJobStatus.SUCCESS, scene_id="scene_03"):
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="cu-pid",
        scene_id=scene_id,
        status=status,
        output_files=[ComfyUIOutputFile(node_id="9", filename="scene.mp4", url="http://127.0.0.1:8188/view?filename=scene.mp4")],
    ))
    return mock


@pytest.mark.asyncio
async def test_d1_veo31_routes_to_veo():
    job = _make_scene_job(model="veo3.1", scene_id="scene_01")
    veo_mock = _make_mock_veo_client(scene_id="scene_01")
    grok_mock = _make_mock_grok_client(scene_id="scene_01")
    cu_mock = _make_mock_cu_client(scene_id="scene_01")

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
async def test_d2_ltx23_routes_to_comfyui():
    job = _make_scene_job(model="ltx2.3", scene_id="scene_02")
    veo_mock = _make_mock_veo_client(scene_id="scene_02")
    cu_mock = _make_mock_cu_client(scene_id="scene_02")

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_mock_grok_client()),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher()
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    cu_mock.submit_and_await.assert_called_once()
    veo_mock.submit_and_await.assert_not_called()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_d3_empty_model_routes_to_comfyui():
    job = _make_scene_job(model="", scene_id="scene_03")
    veo_mock = _make_mock_veo_client(scene_id="scene_03")
    cu_mock = _make_mock_cu_client(scene_id="scene_03")

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_mock_grok_client()),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher()
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    cu_mock.submit_and_await.assert_called_once()
    veo_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_d4_mixed_batch_routes_correctly():
    jobs = [
        _make_scene_job(model="veo3.1", scene_id="scene_01"),
        _make_scene_job(model="grok4.2", scene_id="scene_02"),
        _make_scene_job(model="",        scene_id="scene_03"),
        _make_scene_job(model="ltx2.3",  scene_id="scene_04"),
    ]

    veo_calls: list[str] = []
    grok_calls: list[str] = []
    cu_calls: list[str] = []

    async def _veo_side_effect(req, **_):
        from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
        veo_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="veo", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=[ComfyUIOutputFile(node_id="veo3.1", filename="o.mp4", url="https://x/o.mp4")],
        )

    async def _grok_side_effect(req, **_):
        from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
        grok_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="grok", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=[ComfyUIOutputFile(node_id="grok_video", filename="o.mp4", url="https://x/o.mp4")],
        )

    async def _cu_side_effect(workflow, scene_id="", **_):
        from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
        cu_calls.append(scene_id)
        return ComfyUISubmitResult(
            prompt_id="cu", scene_id=scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=[ComfyUIOutputFile(node_id="9", filename="o.mp4", url="http://x/o.mp4")],
        )

    veo_mock = AsyncMock()
    veo_mock.__aenter__ = AsyncMock(return_value=veo_mock)
    veo_mock.__aexit__ = AsyncMock(return_value=False)
    veo_mock.submit_and_await = AsyncMock(side_effect=_veo_side_effect)

    grok_mock = AsyncMock()
    grok_mock.__aenter__ = AsyncMock(return_value=grok_mock)
    grok_mock.__aexit__ = AsyncMock(return_value=False)
    grok_mock.submit_and_await = AsyncMock(side_effect=_grok_side_effect)

    cu_mock = AsyncMock()
    cu_mock.__aenter__ = AsyncMock(return_value=cu_mock)
    cu_mock.__aexit__ = AsyncMock(return_value=False)
    cu_mock.submit_and_await = AsyncMock(side_effect=_cu_side_effect)

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", xai_api_key="xkey")
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert veo_calls == ["scene_01"]
    assert grok_calls == ["scene_02"]
    assert set(cu_calls) == {"scene_03", "scene_04"}
    assert result.total == 4
    assert result.succeeded == 4


@pytest.mark.asyncio
async def test_d5_dispatch_scenario_parses_and_routes():
    scenario = """\
## Scene 1 (0:00–0:05)
[VISUAL: close-up of dog eyes under moonlight]
[MODEL: veo3.1]

## Scene 2 (0:05–0:12)
[VISUAL: sword clash in slow motion]
[MODEL: ltx2.3]
"""
    req = VideoDispatchRequest(
        project_id="test_proj",
        scenario_text=scenario,
        google_api_key="gkey",
    )

    veo_mock = _make_mock_veo_client(scene_id="scene_01")
    cu_mock = _make_mock_cu_client(scene_id="scene_02")

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_mock_grok_client()),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        result = await dispatch_scenario(req)

    assert result.total == 2
    veo_mock.submit_and_await.assert_called_once()
    cu_mock.submit_and_await.assert_called_once()


@pytest.mark.asyncio
async def test_d6_veo_error_captured_in_result():
    job = _make_scene_job(model="veo3.1", scene_id="scene_err")
    from bridge.schemas import ComfyUISubmitResult

    veo_mock = AsyncMock()
    veo_mock.__aenter__ = AsyncMock(return_value=veo_mock)
    veo_mock.__aexit__ = AsyncMock(return_value=False)
    veo_mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="x", scene_id="scene_err",
        status=ComfyUIJobStatus.ERROR,
        error="quota exceeded",
    ))
    cu_mock = _make_mock_cu_client()

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_mock_grok_client()),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=False)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert result.succeeded == 0
    assert result.results[0].error == "quota exceeded"


@pytest.mark.asyncio
async def test_d7_batch_counts_accurate():
    jobs = [
        _make_scene_job(model="veo3.1",  scene_id="s1"),
        _make_scene_job(model="grok4.2", scene_id="s2"),
        _make_scene_job(model="",        scene_id="s3"),
    ]
    from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile

    # Veo: success; Grok: error; ComfyUI: success
    veo_mock = AsyncMock()
    veo_mock.__aenter__ = AsyncMock(return_value=veo_mock)
    veo_mock.__aexit__ = AsyncMock(return_value=False)
    veo_mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="v1", scene_id="s1", status=ComfyUIJobStatus.SUCCESS,
        output_files=[ComfyUIOutputFile(node_id="veo3.1", filename="o.mp4", url="u")],
    ))

    grok_mock = AsyncMock()
    grok_mock.__aenter__ = AsyncMock(return_value=grok_mock)
    grok_mock.__aexit__ = AsyncMock(return_value=False)
    grok_mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="g1", scene_id="s2", status=ComfyUIJobStatus.ERROR, error="fail",
    ))

    cu_mock = _make_mock_cu_client(scene_id="s3")

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", xai_api_key="xkey", fallback_to_comfyui=False)
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert result.total == 3
    assert result.succeeded == 2   # s1 + s3
    assert result.failed == 1      # s2
