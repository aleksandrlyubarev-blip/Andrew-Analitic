"""
tests/test_higgsfield_client.py + test_video_dispatcher.py (combined)
======================================================================
Tests for HiggsfieldClient and VideoDispatcher.

H-series: HiggsfieldClient unit tests (mocked httpx transport).
D-series: VideoDispatcher routing / integration tests (both backends mocked).

Coverage:
  H1.  submit() returns job_id from /v1/generation response.
  H2.  submit() raises RuntimeError when job_id absent.
  H3.  await_completion() returns SUCCESS on status=completed + output_url.
  H4.  await_completion() returns ERROR on status=failed.
  H5.  await_completion() returns TIMEOUT when deadline exceeded.
  H6.  Unknown status keeps polling until timeout.
  H7.  _extract_outputs() handles single output_url field.
  H8.  _extract_outputs() handles outputs list field.
  H9.  _build_payload() includes camera_motion when set.
  H10. _build_payload() includes start/end frame URLs for WAN 2.6.
  H11. _build_payload() includes reference_image_urls for Veo 3.1.
  H12. MODEL_MAP maps sora2 → sora-2, wan2.6 → wan-2.6, veo3.1 → veo-3.1.
  H13. HTTP 500 on /v1/generation/{id} is retried until timeout.

  D1.  sora2 scene → routed to Higgsfield.
  D2.  ltx2.3 scene → routed to ComfyUI.
  D3.  empty preferred_model → routed to ComfyUI.
  D4.  Mixed batch: some Higgsfield, some ComfyUI.
  D5.  dispatch_scenario() parses scenario and routes correctly.
  D6.  Higgsfield error → scene result has status=ERROR, batch counts update.
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

_JOB_ID = "hf-job-uuid-0001"

DEFAULT_REQ = HiggsfieldGenerationRequest(
    scene_id="scene_01",
    model="sora2",
    prompt="cinematic close-up of Pinoblanco's glowing eyes",
    duration_sec=5,
)


def _hf_success(job_id: str = _JOB_ID, url: str = "https://cdn.higgsfield.ai/out/clip.mp4") -> dict:
    return {"job_id": job_id, "status": "completed", "output_url": url}


def _hf_failed(job_id: str = _JOB_ID) -> dict:
    return {"job_id": job_id, "status": "failed", "error": "CUDA OOM on cloud node"}


def _hf_pending(job_id: str = _JOB_ID) -> dict:
    return {"job_id": job_id, "status": "processing"}


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
    client = HiggsfieldClient(api_key="test-key", poll_interval_sec=0.01)
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
async def test_h1_submit_returns_job_id():
    routes = {"POST /v1/generation": {"job_id": _JOB_ID, "status": "queued"}}
    async with _make_hf_client(routes) as client:
        job_id = await client.submit(DEFAULT_REQ)
    assert job_id == _JOB_ID


@pytest.mark.asyncio
async def test_h2_submit_raises_without_job_id():
    routes = {"POST /v1/generation": {"status": "queued"}}
    async with _make_hf_client(routes) as client:
        with pytest.raises(RuntimeError, match="no job_id"):
            await client.submit(DEFAULT_REQ)


@pytest.mark.asyncio
async def test_h3_await_completion_success():
    routes = {f"GET /v1/generation/{_JOB_ID}": _hf_success()}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_JOB_ID, scene_id="scene_01")
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert result.scene_id == "scene_01"
    assert len(result.output_files) == 1
    assert "clip.mp4" in result.output_files[0].filename


@pytest.mark.asyncio
async def test_h4_await_completion_error():
    routes = {f"GET /v1/generation/{_JOB_ID}": _hf_failed()}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_JOB_ID)
    assert result.status == ComfyUIJobStatus.ERROR
    assert "CUDA OOM" in (result.error or "")


@pytest.mark.asyncio
async def test_h5_await_completion_timeout():
    routes = {f"GET /v1/generation/{_JOB_ID}": _hf_pending()}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_JOB_ID, timeout_sec=0.05)
    assert result.status == ComfyUIJobStatus.TIMEOUT


@pytest.mark.asyncio
async def test_h6_unknown_status_keeps_polling():
    call_n = 0

    def _handler(request: httpx.Request) -> dict:
        nonlocal call_n
        call_n += 1
        if call_n < 3:
            return {"job_id": _JOB_ID, "status": "unknown_state"}
        return _hf_success()

    routes = {f"GET /v1/generation/{_JOB_ID}": _handler}
    async with _make_hf_client(routes) as client:
        result = await client.await_completion(_JOB_ID, timeout_sec=5.0)
    assert result.status == ComfyUIJobStatus.SUCCESS
    assert call_n >= 3


def test_h7_extract_outputs_single_url():
    client = HiggsfieldClient.__new__(HiggsfieldClient)
    data = {"status": "completed", "output_url": "https://cdn.higgsfield.ai/scene01.mp4"}
    files = client._extract_outputs("job1", data)
    assert len(files) == 1
    assert files[0].filename == "scene01.mp4"
    assert files[0].url == "https://cdn.higgsfield.ai/scene01.mp4"


def test_h8_extract_outputs_list():
    client = HiggsfieldClient.__new__(HiggsfieldClient)
    data = {
        "status": "completed",
        "outputs": [
            {"url": "https://cdn.higgsfield.ai/s1.mp4", "type": "video"},
            {"url": "https://cdn.higgsfield.ai/s2.mp4", "type": "video"},
        ],
    }
    files = client._extract_outputs("job1", data)
    assert len(files) == 2
    assert {f.filename for f in files} == {"s1.mp4", "s2.mp4"}


def test_h9_build_payload_camera_motion():
    client = HiggsfieldClient(api_key="x")
    req = HiggsfieldGenerationRequest(
        model="sora2", prompt="test", camera_motion="ultra-slow push-in zoom"
    )
    payload = client._build_payload(req)
    assert payload["camera_motion"] == "ultra-slow push-in zoom"


def test_h10_build_payload_start_end_frames():
    client = HiggsfieldClient(api_key="x")
    req = HiggsfieldGenerationRequest(
        model="wan2.6",
        prompt="test",
        start_frame_url="https://imgs/first.png",
        end_frame_url="https://imgs/last.png",
    )
    payload = client._build_payload(req)
    assert payload["start_frame_url"] == "https://imgs/first.png"
    assert payload["end_frame_url"] == "https://imgs/last.png"


def test_h11_build_payload_reference_images():
    client = HiggsfieldClient(api_key="x")
    req = HiggsfieldGenerationRequest(
        model="veo3.1",
        prompt="test",
        reference_image_urls=["https://imgs/ref1.png", "https://imgs/ref2.png"],
    )
    payload = client._build_payload(req)
    assert payload["reference_image_urls"] == ["https://imgs/ref1.png", "https://imgs/ref2.png"]


def test_h12_model_map():
    assert MODEL_MAP["sora2"] == "sora-2"
    assert MODEL_MAP["wan2.6"] == "wan-2.6"
    assert MODEL_MAP["veo3.1"] == "veo-3.1"


@pytest.mark.asyncio
async def test_h13_http_500_retried_until_timeout():
    class _ErrTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            if "/v1/generation/" in request.url.path and request.method == "GET":
                return httpx.Response(500, text="Server Error")
            return httpx.Response(200, json={"job_id": _JOB_ID})

    client = HiggsfieldClient(api_key="x", poll_interval_sec=0.01)
    client._http = httpx.AsyncClient(
        base_url="https://mock-hf.ai", transport=_ErrTransport()
    )
    async with client:
        result = await client.await_completion(_JOB_ID, timeout_sec=0.1)
    assert result.status == ComfyUIJobStatus.TIMEOUT


# ═══════════════════════════════════════════════════════════════════════════════
# D-series: VideoDispatcher
# ═══════════════════════════════════════════════════════════════════════════════

# We patch both inner clients so no real HTTP calls are made.

def _make_mock_hf_client(status=ComfyUIJobStatus.SUCCESS, scene_id="scene_01"):
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
    mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="hf-pid",
        scene_id=scene_id,
        status=status,
        output_files=[ComfyUIOutputFile(node_id="hf", filename="out.mp4", url="https://cdn/out.mp4")],
    ))
    return mock


def _make_mock_cu_client(status=ComfyUIJobStatus.SUCCESS, scene_id="scene_02"):
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
async def test_d1_sora2_routes_to_higgsfield():
    job = _make_scene_job(model="sora2", scene_id="scene_01")
    hf_mock = _make_mock_hf_client(scene_id="scene_01")
    cu_mock = _make_mock_cu_client(scene_id="scene_01")

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(higgsfield_api_key="key")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    hf_mock.submit_and_await.assert_called_once()
    cu_mock.submit_and_await.assert_not_called()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_d2_ltx23_routes_to_comfyui():
    job = _make_scene_job(model="ltx2.3", scene_id="scene_02")
    hf_mock = _make_mock_hf_client(scene_id="scene_02")
    cu_mock = _make_mock_cu_client(scene_id="scene_02")

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher()
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    cu_mock.submit_and_await.assert_called_once()
    hf_mock.submit_and_await.assert_not_called()
    assert result.succeeded == 1


@pytest.mark.asyncio
async def test_d3_empty_model_routes_to_comfyui():
    job = _make_scene_job(model="", scene_id="scene_03")
    hf_mock = _make_mock_hf_client(scene_id="scene_03")
    cu_mock = _make_mock_cu_client(scene_id="scene_03")

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher()
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    cu_mock.submit_and_await.assert_called_once()
    hf_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_d4_mixed_batch_routes_correctly():
    jobs = [
        _make_scene_job(model="wan2.6", scene_id="scene_01"),
        _make_scene_job(model="",       scene_id="scene_02"),
        _make_scene_job(model="veo3.1", scene_id="scene_03"),
        _make_scene_job(model="ltx2.3", scene_id="scene_04"),
    ]

    hf_calls: list[str] = []
    cu_calls: list[str] = []

    async def _hf_side_effect(req, **_):
        from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
        hf_calls.append(req.scene_id)
        return ComfyUISubmitResult(
            prompt_id="hf", scene_id=req.scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=[ComfyUIOutputFile(node_id="hf", filename="o.mp4", url="https://x/o.mp4")],
        )

    async def _cu_side_effect(workflow, scene_id="", **_):
        from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile
        cu_calls.append(scene_id)
        return ComfyUISubmitResult(
            prompt_id="cu", scene_id=scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=[ComfyUIOutputFile(node_id="9", filename="o.mp4", url="http://x/o.mp4")],
        )

    hf_mock = AsyncMock()
    hf_mock.__aenter__ = AsyncMock(return_value=hf_mock)
    hf_mock.__aexit__ = AsyncMock(return_value=False)
    hf_mock.submit_and_await = AsyncMock(side_effect=_hf_side_effect)

    cu_mock = AsyncMock()
    cu_mock.__aenter__ = AsyncMock(return_value=cu_mock)
    cu_mock.__aexit__ = AsyncMock(return_value=False)
    cu_mock.submit_and_await = AsyncMock(side_effect=_cu_side_effect)

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(higgsfield_api_key="key")
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert set(hf_calls) == {"scene_01", "scene_03"}
    assert set(cu_calls) == {"scene_02", "scene_04"}
    assert result.total == 4
    assert result.succeeded == 4


@pytest.mark.asyncio
async def test_d5_dispatch_scenario_parses_and_routes():
    scenario = """\
## Scene 1 (0:00–0:05)
[VISUAL: close-up of dog eyes under moonlight]
[MODEL: sora2]

## Scene 2 (0:05–0:12)
[VISUAL: sword clash in slow motion]
[MODEL: ltx2.3]
"""
    req = VideoDispatchRequest(
        project_id="test_proj",
        scenario_text=scenario,
        higgsfield_api_key="key",
    )

    hf_mock = _make_mock_hf_client(scene_id="scene_01")
    cu_mock = _make_mock_cu_client(scene_id="scene_02")

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        result = await dispatch_scenario(req)

    assert result.total == 2
    hf_mock.submit_and_await.assert_called_once()
    cu_mock.submit_and_await.assert_called_once()


@pytest.mark.asyncio
async def test_d6_higgsfield_error_captured_in_result():
    job = _make_scene_job(model="sora2", scene_id="scene_err")
    from bridge.schemas import ComfyUISubmitResult

    hf_mock = AsyncMock()
    hf_mock.__aenter__ = AsyncMock(return_value=hf_mock)
    hf_mock.__aexit__ = AsyncMock(return_value=False)
    hf_mock.submit_and_await = AsyncMock(return_value=ComfyUISubmitResult(
        prompt_id="x", scene_id="scene_err",
        status=ComfyUIJobStatus.ERROR,
        error="quota exceeded",
    ))
    cu_mock = _make_mock_cu_client()

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(higgsfield_api_key="key")
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert result.succeeded == 0
    assert result.results[0].error == "quota exceeded"


@pytest.mark.asyncio
async def test_d7_batch_counts_accurate():
    jobs = [
        _make_scene_job(model="sora2",  scene_id="s1"),
        _make_scene_job(model="wan2.6", scene_id="s2"),
        _make_scene_job(model="",       scene_id="s3"),
    ]
    from bridge.schemas import ComfyUISubmitResult, ComfyUIOutputFile

    # HF: first success, second error
    hf_side = [
        ComfyUISubmitResult(prompt_id="h1", scene_id="s1", status=ComfyUIJobStatus.SUCCESS,
                            output_files=[ComfyUIOutputFile(node_id="hf", filename="o.mp4", url="u")]),
        ComfyUISubmitResult(prompt_id="h2", scene_id="s2", status=ComfyUIJobStatus.ERROR, error="fail"),
    ]
    hf_mock = AsyncMock()
    hf_mock.__aenter__ = AsyncMock(return_value=hf_mock)
    hf_mock.__aexit__ = AsyncMock(return_value=False)
    hf_mock.submit_and_await = AsyncMock(side_effect=hf_side)

    cu_mock = _make_mock_cu_client(scene_id="s3")

    with (
        patch("bridge.video_dispatcher.HiggsfieldClient", return_value=hf_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(higgsfield_api_key="key")
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert result.total == 3
    assert result.succeeded == 2   # s1 + s3
    assert result.failed == 1      # s2
