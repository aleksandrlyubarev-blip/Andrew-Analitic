"""
tests/test_fallback_routing.py
==============================
Tests for VideoDispatcher fallback routing.

When a cloud backend (Veo 3.1 or Grok 4.2) returns ERROR or TIMEOUT,
the dispatcher should transparently retry the scene on local ComfyUI
if ``fallback_to_comfyui=True`` (the default).

F-series: fallback behaviour
  F1.  Veo ERROR → fallback to ComfyUI → SUCCESS.
  F2.  Veo TIMEOUT → fallback to ComfyUI → SUCCESS.
  F3.  Grok ERROR → fallback to ComfyUI → SUCCESS.
  F4.  Grok TIMEOUT → fallback to ComfyUI → SUCCESS.
  F5.  Veo SUCCESS → ComfyUI NOT called (no unnecessary fallback).
  F6.  Grok SUCCESS → ComfyUI NOT called.
  F7.  fallback_to_comfyui=False: Veo ERROR stays ERROR, ComfyUI not called.
  F8.  fallback_to_comfyui=False: Grok ERROR stays ERROR, ComfyUI not called.
  F9.  ComfyUI fallback ERROR propagated correctly (failed count = 1).
  F10. Mixed batch: scene1 veo-OK, scene2 veo-ERROR→CU-OK, scene3 grok-ERROR→CU-OK.
  F11. dispatch_scenario respects fallback_to_comfyui=False field from request.
  F12. dispatch_scenario uses fallback_to_comfyui=True by default.
"""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.schemas import (
    ComfyUIJobStatus,
    ComfyUIOutputFile,
    ComfyUISubmitResult,
    LtxGenerationConfig,
    LtxKeyframe,
    LtxSceneJob,
    VideoDispatchRequest,
)
from bridge.video_dispatcher import VideoDispatcher, dispatch_scenario

# ── helpers ───────────────────────────────────────────────────────────────────


def _make_job(model: str, scene_id: str = "scene_01") -> LtxSceneJob:
    return LtxSceneJob(
        job_id=f"proj_{scene_id}_abc",
        scene_id=scene_id,
        scene_index=0,
        workflow="first_last",
        duration_sec=5.0,
        prompts=["test prompt"],
        preferred_model=model,
        keyframes=[
            LtxKeyframe(index=0, frame_type="first", source_prompt="test"),
            LtxKeyframe(index=-1, frame_type="last", source_prompt="test"),
        ],
        model_config_ltx={
            "variant": "fp8", "resolution": "1920x1080",
            "aspect_ratio": "16:9", "vram_budget_gb": 12,
        },
    )


def _ok_result(scene_id: str = "scene_01", backend: str = "cu") -> ComfyUISubmitResult:
    return ComfyUISubmitResult(
        prompt_id=f"{backend}-pid",
        scene_id=scene_id,
        status=ComfyUIJobStatus.SUCCESS,
        output_files=[
            ComfyUIOutputFile(node_id=backend, filename="out.mp4", url=f"https://x/{backend}.mp4")
        ],
    )


def _err_result(scene_id: str = "scene_01", error: str = "quota exceeded") -> ComfyUISubmitResult:
    return ComfyUISubmitResult(
        prompt_id="",
        scene_id=scene_id,
        status=ComfyUIJobStatus.ERROR,
        error=error,
    )


def _timeout_result(scene_id: str = "scene_01") -> ComfyUISubmitResult:
    return ComfyUISubmitResult(
        prompt_id="",
        scene_id=scene_id,
        status=ComfyUIJobStatus.TIMEOUT,
        error="timeout",
    )


def _make_client(result: ComfyUISubmitResult) -> AsyncMock:
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(return_value=result)
    return mock


def _make_client_seq(*results: ComfyUISubmitResult) -> AsyncMock:
    """Client that returns results in sequence (first call → results[0], etc.)."""
    mock = AsyncMock()
    mock.__aenter__ = AsyncMock(return_value=mock)
    mock.__aexit__ = AsyncMock(return_value=False)
    mock.submit_and_await = AsyncMock(side_effect=list(results))
    return mock


# ── F-series ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_f1_veo_error_falls_back_to_comfyui():
    job = _make_job("veo3.1", "scene_01")
    veo_mock = _make_client(_err_result("scene_01"))
    grok_mock = _make_client(_ok_result("scene_01", "grok"))
    cu_mock = _make_client(_ok_result("scene_01", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_called_once()
    grok_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_f2_veo_timeout_falls_back_to_comfyui():
    job = _make_job("veo3.1", "scene_01")
    veo_mock = _make_client(_timeout_result("scene_01"))
    cu_mock = _make_client(_ok_result("scene_01", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_called_once()


@pytest.mark.asyncio
async def test_f3_grok_error_falls_back_to_comfyui():
    job = _make_job("grok4.2", "scene_05")
    grok_mock = _make_client(_err_result("scene_05", "rate limit"))
    cu_mock = _make_client(_ok_result("scene_05", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_called_once()


@pytest.mark.asyncio
async def test_f4_grok_timeout_falls_back_to_comfyui():
    job = _make_job("grok4.2", "scene_05")
    grok_mock = _make_client(_timeout_result("scene_05"))
    cu_mock = _make_client(_ok_result("scene_05", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_called_once()


@pytest.mark.asyncio
async def test_f5_veo_success_does_not_call_comfyui():
    job = _make_job("veo3.1", "scene_01")
    veo_mock = _make_client(_ok_result("scene_01", "veo"))
    cu_mock = _make_client(_ok_result("scene_01", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_f6_grok_success_does_not_call_comfyui():
    job = _make_job("grok4.2", "scene_05")
    grok_mock = _make_client(_ok_result("scene_05", "grok"))
    cu_mock = _make_client(_ok_result("scene_05", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_f7_no_fallback_veo_error_stays_error():
    job = _make_job("veo3.1", "scene_01")
    veo_mock = _make_client(_err_result("scene_01", "quota exceeded"))
    cu_mock = _make_client(_ok_result("scene_01", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=False)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert result.succeeded == 0
    assert result.results[0].error == "quota exceeded"
    cu_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_f8_no_fallback_grok_error_stays_error():
    job = _make_job("grok4.2", "scene_05")
    grok_mock = _make_client(_err_result("scene_05", "rate limit"))
    cu_mock = _make_client(_ok_result("scene_05", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(xai_api_key="xkey", fallback_to_comfyui=False)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert "rate limit" in (result.results[0].error or "")
    cu_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_f9_comfyui_fallback_also_errors_propagated():
    """Veo fails → fallback to ComfyUI → ComfyUI also fails → final ERROR."""
    job = _make_job("veo3.1", "scene_01")
    veo_mock = _make_client(_err_result("scene_01", "veo quota"))
    cu_mock = _make_client(_err_result("scene_01", "comfyui oom"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        dispatcher = VideoDispatcher(google_api_key="gkey", fallback_to_comfyui=True)
        result = await dispatcher.dispatch_batch([job], LtxGenerationConfig())

    assert result.failed == 1
    assert result.succeeded == 0
    assert result.results[0].error == "comfyui oom"


@pytest.mark.asyncio
async def test_f10_mixed_batch_with_fallbacks():
    """3 scenes: veo-OK, veo-ERROR→CU-OK, grok-ERROR→CU-OK → all succeed."""
    jobs = [
        _make_job("veo3.1",  "s1"),
        _make_job("veo3.1",  "s2"),
        _make_job("grok4.2", "s3"),
    ]

    veo_results = {
        "s1": _ok_result("s1", "veo"),
        "s2": _err_result("s2", "veo fail"),
    }
    grok_results = {
        "s3": _err_result("s3", "grok fail"),
    }
    cu_results = {
        "s2": _ok_result("s2", "cu"),
        "s3": _ok_result("s3", "cu"),
    }

    async def _veo_side(req, **_):
        return veo_results[req.scene_id]

    async def _grok_side(req, **_):
        return grok_results[req.scene_id]

    async def _cu_side(workflow, scene_id="", **_):
        return cu_results[scene_id]

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
        dispatcher = VideoDispatcher(
            google_api_key="gkey", xai_api_key="xkey", fallback_to_comfyui=True
        )
        result = await dispatcher.dispatch_batch(jobs, LtxGenerationConfig())

    assert result.total == 3
    assert result.succeeded == 3
    assert result.failed == 0
    assert cu_mock.submit_and_await.call_count == 2  # s2 and s3 fell back


@pytest.mark.asyncio
async def test_f11_dispatch_scenario_respects_no_fallback():
    scenario = """\
## Scene 1 (0:00–0:05)
[VISUAL: test scene]
[MODEL: veo3.1]
"""
    req = VideoDispatchRequest(
        project_id="test",
        scenario_text=scenario,
        google_api_key="gkey",
        fallback_to_comfyui=False,
    )

    veo_mock = _make_client(_err_result("scene_01", "quota"))
    cu_mock = _make_client(_ok_result("scene_01", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=veo_mock),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        result = await dispatch_scenario(req)

    assert result.failed == 1
    cu_mock.submit_and_await.assert_not_called()


@pytest.mark.asyncio
async def test_f12_dispatch_scenario_fallback_default_true():
    scenario = """\
## Scene 1 (0:00–0:05)
[VISUAL: test scene]
[MODEL: grok4.2]
"""
    req = VideoDispatchRequest(
        project_id="test",
        scenario_text=scenario,
        xai_api_key="xkey",
        # fallback_to_comfyui not set → defaults to True
    )

    grok_mock = _make_client(_err_result("scene_01", "rate limit"))
    cu_mock = _make_client(_ok_result("scene_01", "cu"))

    with (
        patch("bridge.video_dispatcher.VeoVideoClient", return_value=_make_client(_ok_result())),
        patch("bridge.video_dispatcher.GrokVideoClient", return_value=grok_mock),
        patch("bridge.video_dispatcher.ComfyUIClient", return_value=cu_mock),
    ):
        result = await dispatch_scenario(req)

    assert result.succeeded == 1
    cu_mock.submit_and_await.assert_called_once()
