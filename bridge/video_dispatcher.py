"""
bridge/video_dispatcher.py
==========================
Routes LtxSceneJob objects to the correct generation backend based on
the ``preferred_model`` field set by the [MODEL: ...] scenario tag.

Routing table
-------------
preferred_model        backend
─────────────────────  ─────────────────────────────────────────────
sora2 / sora-2         Higgsfield API  → HiggsfieldClient
wan2.6 / wan-2.6       Higgsfield API  → HiggsfieldClient
veo3.1 / veo-3.1       Higgsfield API  → HiggsfieldClient
ltx2.3 / ltx / ""      Local ComfyUI   → ComfyUIClient

All results are normalised to ComfyUISubmitResult so downstream
consumers (PinoCut pickup, SceneOps QA) see a uniform interface.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from bridge.comfyui_client import ComfyUIClient
from bridge.comfyui_export import ComfyUIWorkflowExporter
from bridge.higgsfield_client import HiggsfieldClient
from bridge.schemas import (
    ComfyUIBatchResult,
    ComfyUIJobStatus,
    ComfyUISubmitResult,
    HiggsfieldGenerationRequest,
    LtxGenerationConfig,
    LtxSceneJob,
    VideoDispatchRequest,
)
from bridge.ltx_video import LtxVideoPipeline
from bridge.schemas import LtxVideoJobRequest

logger = logging.getLogger("video_dispatcher")

# Models that go to Higgsfield
_HIGGSFIELD_MODELS = frozenset({"sora2", "sora-2", "wan2.6", "wan-2.6", "veo3.1", "veo-3.1"})

# Default guidance scales per model (Higgsfield tuning)
_MODEL_CFG: dict[str, float] = {
    "sora2": 7.5,
    "wan2.6": 6.0,
    "veo3.1": 9.0,
}


class VideoDispatcher:
    """
    Dispatch a batch of LtxSceneJobs to Higgsfield or ComfyUI in parallel.

    Parameters
    ----------
    higgsfield_api_key:
        Higgsfield API key (falls back to env HIGGSFIELD_API_KEY).
    comfyui_host:
        Base URL of the local ComfyUI instance.
    higgsfield_timeout_sec / comfyui_timeout_sec:
        Per-job timeouts for each backend.
    max_concurrent:
        Max parallel in-flight jobs across both backends.
    """

    def __init__(
        self,
        higgsfield_api_key: str | None = None,
        comfyui_host: str = "http://127.0.0.1:8188",
        higgsfield_timeout_sec: float = 900.0,
        comfyui_timeout_sec: float = 600.0,
        max_concurrent: int = 4,
    ) -> None:
        self._hf_key = higgsfield_api_key
        self._comfyui_host = comfyui_host
        self._hf_timeout = higgsfield_timeout_sec
        self._cu_timeout = comfyui_timeout_sec
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._exporter = ComfyUIWorkflowExporter()

    # ── public API ────────────────────────────────────────────────────────────

    async def dispatch_batch(
        self,
        jobs: list[LtxSceneJob],
        config: LtxGenerationConfig,
    ) -> ComfyUIBatchResult:
        """
        Dispatch all jobs concurrently; route each one by preferred_model.

        Returns a ComfyUIBatchResult with per-scene results.
        """
        async with (
            HiggsfieldClient(
                api_key=self._hf_key,
                timeout_sec=self._hf_timeout,
            ) as hf_client,
            ComfyUIClient(
                host=self._comfyui_host,
                timeout_sec=self._cu_timeout,
            ) as cu_client,
        ):
            tasks = [
                self._dispatch_one(job, config, hf_client, cu_client)
                for job in jobs
            ]
            results: list[ComfyUISubmitResult] = await asyncio.gather(*tasks)

        succeeded = sum(1 for r in results if r.status == ComfyUIJobStatus.SUCCESS)
        failed = len(results) - succeeded
        return ComfyUIBatchResult(
            total=len(results),
            succeeded=succeeded,
            failed=failed,
            results=results,
        )

    # ── private ───────────────────────────────────────────────────────────────

    async def _dispatch_one(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig,
        hf_client: HiggsfieldClient,
        cu_client: ComfyUIClient,
    ) -> ComfyUISubmitResult:
        async with self._semaphore:
            model = job.preferred_model.lower().replace(" ", "")
            if model in _HIGGSFIELD_MODELS:
                return await self._run_higgsfield(job, config, hf_client)
            return await self._run_comfyui(job, config, cu_client)

    async def _run_higgsfield(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig,
        client: HiggsfieldClient,
    ) -> ComfyUISubmitResult:
        prompt = job.prompts[0] if job.prompts else ""
        req = HiggsfieldGenerationRequest(
            scene_id=job.scene_id,
            model=job.preferred_model,
            prompt=prompt,
            duration_sec=max(1, int(job.duration_sec)),
            fps=24,
            resolution=config.resolution,
            aspect_ratio=config.aspect_ratio,
            guidance_scale=_MODEL_CFG.get(job.preferred_model, 7.5),
            camera_motion=job.camera_motion,
            # Keyframe images if pre-rendered stills were provided
            start_frame_url=self._keyframe_url(job, frame_type="first"),
            end_frame_url=self._keyframe_url(job, frame_type="last"),
            reference_image_urls=self._reference_urls(job),
        )
        logger.info(
            "→ Higgsfield  scene=%s  model=%s  duration=%ds",
            job.scene_id, job.preferred_model, req.duration_sec,
        )
        try:
            return await client.submit_and_await(req, timeout_sec=self._hf_timeout)
        except Exception as exc:
            logger.error("Higgsfield error for %s: %s", job.scene_id, exc)
            return ComfyUISubmitResult(
                prompt_id="",
                scene_id=job.scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=str(exc),
            )

    async def _run_comfyui(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig,
        client: ComfyUIClient,
    ) -> ComfyUISubmitResult:
        workflow = self._exporter.export_job(job, config)
        logger.info("→ ComfyUI  scene=%s  workflow=%s", job.scene_id, job.workflow)
        try:
            return await client.submit_and_await(
                workflow,
                scene_id=job.scene_id,
                timeout_sec=self._cu_timeout,
            )
        except Exception as exc:
            logger.error("ComfyUI error for %s: %s", job.scene_id, exc)
            return ComfyUISubmitResult(
                prompt_id="",
                scene_id=job.scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=str(exc),
            )

    def _keyframe_url(self, job: LtxSceneJob, frame_type: str) -> str | None:
        for kf in job.keyframes:
            if kf.frame_type == frame_type and kf.image_path:
                return kf.image_path  # caller should be an HTTPS URL for cloud use
        return None

    def _reference_urls(self, job: LtxSceneJob) -> list[str]:
        return [
            kf.image_path
            for kf in job.keyframes
            if kf.image_path and kf.frame_type == "middle"
        ]


# ── convenience facade ────────────────────────────────────────────────────────


async def dispatch_scenario(req: VideoDispatchRequest) -> ComfyUIBatchResult:
    """
    Full pipeline: scenario text → parse → dispatch → results.

    Entry point for POST /video/dispatch.
    """
    pipeline = LtxVideoPipeline()
    job_response = pipeline.run(
        LtxVideoJobRequest(
            project_id=req.project_id,
            scenario_text=req.scenario_text,
            config=req.config,
        )
    )

    if not job_response.scene_jobs:
        return ComfyUIBatchResult(total=0, succeeded=0, failed=0, results=[])

    dispatcher = VideoDispatcher(
        higgsfield_api_key=req.higgsfield_api_key,
        comfyui_host=req.comfyui_host,
        higgsfield_timeout_sec=req.higgsfield_timeout_sec,
        comfyui_timeout_sec=req.comfyui_timeout_sec,
    )
    return await dispatcher.dispatch_batch(job_response.scene_jobs, req.config)
