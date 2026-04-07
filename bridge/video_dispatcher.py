"""
bridge/video_dispatcher.py
==========================
Routes LtxSceneJob objects to the correct generation backend based on
the ``preferred_model`` field set by the [MODEL: ...] scenario tag.

Routing table
-------------
preferred_model                    backend
─────────────────────────────────  ─────────────────────────────────────────────
veo3.1 / veo-3.1                   Google Veo 3.1  → VeoVideoClient  (AI Studio)
grok4.2 / grok-4.2                 xAI Grok 4.2    → GrokVideoClient
dop / dop-lite / dop-turbo         Higgsfield DOP  → HiggsfieldClient
seedance / seedance-pro            Higgsfield Seedance → HiggsfieldClient
seedance1.5-pro / seedance1.5      Higgsfield Seedance 1.5 → HiggsfieldClient
kling2.1 / kling2.1-pro            Higgsfield Kling → HiggsfieldClient
wan2.6 / wan2.6-t2v                Higgsfield Wan  → HiggsfieldClient
higgsfield-ai/… / bytedance/… /…   Full Higgsfield slug → HiggsfieldClient
ltx2.3 / ltx / ""                  Local ComfyUI   → ComfyUIClient

All results are normalised to ComfyUISubmitResult so downstream
consumers (PinoCut pickup, SceneOps QA) see a uniform interface.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Callable

from bridge.comfyui_client import ComfyUIClient
from bridge.comfyui_export import ComfyUIWorkflowExporter
from bridge.grok_video_client import GrokVideoClient
from bridge.higgsfield_client import HiggsfieldClient, MODEL_MAP as _HF_MODEL_MAP
from bridge.veo_client import VeoVideoClient
from bridge.schemas import (
    ComfyUIBatchResult,
    ComfyUIJobStatus,
    ComfyUISubmitResult,
    GrokVideoRequest,
    HiggsfieldGenerationRequest,
    LtxGenerationConfig,
    LtxSceneJob,
    VeoVideoRequest,
    VideoDispatchRequest,
)
from bridge.ltx_video import LtxVideoPipeline
from bridge.schemas import LtxVideoJobRequest

logger = logging.getLogger("video_dispatcher")

# Models routed to Google Veo 3.1 (AI Studio direct)
_VEO_MODELS = frozenset({"veo3.1", "veo-3.1", "veo3", "veo-3"})

# Models routed to xAI Grok 4.2
_GROK_MODELS = frozenset({"grok4.2", "grok-4.2", "grok4", "grok-4"})

# Models routed to Higgsfield platform API — friendly aliases from MODEL_MAP
_HIGGSFIELD_MODELS = frozenset(_HF_MODEL_MAP.keys())

# Full-slug prefixes that must be sent to Higgsfield (e.g. "higgsfield-ai/dop/turbo")
_HIGGSFIELD_SLUG_PREFIXES = (
    "higgsfield-ai/",
    "kling-video/",
    "bytedance/",
    "wan/",
    "alibaba-cloud/",
)


def _is_higgsfield(model: str) -> bool:
    return model in _HIGGSFIELD_MODELS or any(
        model.startswith(p) for p in _HIGGSFIELD_SLUG_PREFIXES
    )



class VideoDispatcher:
    """
    Dispatch a batch of LtxSceneJobs to the appropriate backend in parallel.

    Routing:
        veo3.1 / veo-3.1        → Google Veo 3.1   (AI Studio LRO)
        grok4.2 / grok-4.2      → xAI Grok 4.2
        dop / seedance1.5-pro / → Higgsfield platform API
         wan2.6 / kling2.1 / …
        anything else            → local ComfyUI

    Parameters
    ----------
    google_api_key:
        Google AI Studio key for Veo 3.1.
    xai_api_key:
        xAI key for Grok 4.2.
    hf_key:
        Higgsfield credential ``"{api_key_id}:{api_key_secret}"``.
        Falls back to env HF_KEY / HF_API_KEY + HF_API_SECRET.
    comfyui_host:
        Base URL of the local ComfyUI instance.
    veo_timeout_sec / grok_timeout_sec / higgsfield_timeout_sec / comfyui_timeout_sec:
        Per-job timeouts for each backend.
    fallback_to_comfyui:
        If True (default), a cloud backend failure automatically retries
        the scene on local ComfyUI before marking it as ERROR.
    max_concurrent:
        Max parallel in-flight jobs across all backends.
    """

    def __init__(
        self,
        google_api_key: str | None = None,
        xai_api_key: str | None = None,
        hf_key: str | None = None,
        comfyui_host: str = "http://127.0.0.1:8188",
        veo_timeout_sec: float = 900.0,
        grok_timeout_sec: float = 900.0,
        higgsfield_timeout_sec: float = 900.0,
        comfyui_timeout_sec: float = 600.0,
        fallback_to_comfyui: bool = True,
        max_concurrent: int = 4,
    ) -> None:
        # Env-var fallbacks so callers don't need to pass keys explicitly
        self._google_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        self._xai_key = xai_api_key or os.getenv("XAI_API_KEY")
        self._hf_key = hf_key  # HiggsfieldClient handles its own env fallback
        self._comfyui_host = comfyui_host
        self._veo_timeout = veo_timeout_sec
        self._grok_timeout = grok_timeout_sec
        self._hf_timeout = higgsfield_timeout_sec
        self._cu_timeout = comfyui_timeout_sec
        self._fallback = fallback_to_comfyui
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._exporter = ComfyUIWorkflowExporter()

    # ── public API ────────────────────────────────────────────────────────────

    async def dispatch_batch(
        self,
        jobs: list[LtxSceneJob],
        config: LtxGenerationConfig,
        on_scene_done: Callable[[ComfyUISubmitResult], None] | None = None,
    ) -> ComfyUIBatchResult:
        """
        Dispatch all jobs concurrently; route each one by preferred_model.

        Parameters
        ----------
        on_scene_done:
            Optional callback invoked after each scene finishes (any status).
            Called from within the asyncio event loop; keep it non-blocking.

        Returns a ComfyUIBatchResult with per-scene results.
        """
        async with (
            VeoVideoClient(
                api_key=self._google_key,
                timeout_sec=self._veo_timeout,
            ) as veo_client,
            GrokVideoClient(
                api_key=self._xai_key,
                timeout_sec=self._grok_timeout,
            ) as grok_client,
            HiggsfieldClient(
                hf_key=self._hf_key,
                timeout_sec=self._hf_timeout,
            ) as hf_client,
            ComfyUIClient(
                host=self._comfyui_host,
                timeout_sec=self._cu_timeout,
            ) as cu_client,
        ):
            tasks = [
                self._dispatch_one(job, config, veo_client, grok_client, hf_client,
                                   cu_client, on_scene_done=on_scene_done)
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
        veo_client: VeoVideoClient,
        grok_client: GrokVideoClient,
        hf_client: HiggsfieldClient,
        cu_client: ComfyUIClient,
        on_scene_done: Callable[[ComfyUISubmitResult], None] | None = None,
    ) -> ComfyUISubmitResult:
        async with self._semaphore:
            model = job.preferred_model.strip().lower()
            if model in _VEO_MODELS:
                result = await self._run_veo(job, config, veo_client)
                if result.status != ComfyUIJobStatus.SUCCESS and self._fallback:
                    logger.warning(
                        "Veo failed for %s (%s) — falling back to ComfyUI",
                        job.scene_id, result.error,
                    )
                    result = await self._run_comfyui(job, config, cu_client)
            elif model in _GROK_MODELS:
                result = await self._run_grok(job, config, grok_client)
                if result.status != ComfyUIJobStatus.SUCCESS and self._fallback:
                    logger.warning(
                        "Grok failed for %s (%s) — falling back to ComfyUI",
                        job.scene_id, result.error,
                    )
                    result = await self._run_comfyui(job, config, cu_client)
            elif _is_higgsfield(model):
                result = await self._run_higgsfield(job, config, hf_client)
                if result.status != ComfyUIJobStatus.SUCCESS and self._fallback:
                    logger.warning(
                        "Higgsfield failed for %s (%s) — falling back to ComfyUI",
                        job.scene_id, result.error,
                    )
                    result = await self._run_comfyui(job, config, cu_client)
            else:
                result = await self._run_comfyui(job, config, cu_client)

            if on_scene_done is not None:
                on_scene_done(result)
            return result

    async def _run_veo(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig,
        client: VeoVideoClient,
    ) -> ComfyUISubmitResult:
        prompt = job.prompts[0] if job.prompts else ""
        req = VeoVideoRequest(
            scene_id=job.scene_id,
            prompt=prompt,
            duration_sec=max(1, int(job.duration_sec)),
            fps=24,
            resolution=config.resolution,
            aspect_ratio=config.aspect_ratio,
            camera_motion=job.camera_motion,
            start_frame_url=self._keyframe_url(job, frame_type="first"),
            end_frame_url=self._keyframe_url(job, frame_type="last"),
            reference_image_urls=self._reference_urls(job),
        )
        logger.info(
            "→ Veo 3.1  scene=%s  duration=%ds",
            job.scene_id, req.duration_sec,
        )
        try:
            return await client.submit_and_await(req, timeout_sec=self._veo_timeout)
        except Exception as exc:
            logger.error("Veo error for %s: %s", job.scene_id, exc)
            return ComfyUISubmitResult(
                prompt_id="",
                scene_id=job.scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=str(exc),
            )

    async def _run_grok(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig,
        client: GrokVideoClient,
    ) -> ComfyUISubmitResult:
        prompt = job.prompts[0] if job.prompts else ""
        req = GrokVideoRequest(
            scene_id=job.scene_id,
            prompt=prompt,
            duration_sec=max(1, int(job.duration_sec)),
            fps=24,
            resolution=config.resolution,
            aspect_ratio=config.aspect_ratio,
            camera_motion=job.camera_motion,
            start_frame_url=self._keyframe_url(job, frame_type="first"),
            end_frame_url=self._keyframe_url(job, frame_type="last"),
        )
        logger.info(
            "→ Grok 4.2  scene=%s  duration=%ds",
            job.scene_id, req.duration_sec,
        )
        try:
            return await client.submit_and_await(req, timeout_sec=self._grok_timeout)
        except Exception as exc:
            logger.error("Grok error for %s: %s", job.scene_id, exc)
            return ComfyUISubmitResult(
                prompt_id="",
                scene_id=job.scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=str(exc),
            )

    async def _run_higgsfield(
        self,
        job: LtxSceneJob,
        config: LtxGenerationConfig,
        client: HiggsfieldClient,
    ) -> ComfyUISubmitResult:
        prompt = job.prompts[0] if job.prompts else ""
        req = HiggsfieldGenerationRequest(
            scene_id=job.scene_id,
            model=job.preferred_model.strip().lower(),
            prompt=prompt,
            duration_sec=max(1, int(job.duration_sec)),
            fps=24,
            resolution=config.resolution,
            aspect_ratio=config.aspect_ratio,
            camera_motion=job.camera_motion,
            start_frame_url=self._keyframe_url(job, frame_type="first"),
            end_frame_url=self._keyframe_url(job, frame_type="last"),
        )
        logger.info(
            "→ Higgsfield  scene=%s  model=%s  duration=%ds",
            job.scene_id, req.model, req.duration_sec,
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


async def dispatch_scenario(
    req: VideoDispatchRequest,
    on_scene_done: Callable[[ComfyUISubmitResult], None] | None = None,
) -> ComfyUIBatchResult:
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
        google_api_key=req.google_api_key,
        xai_api_key=req.xai_api_key,
        hf_key=req.hf_key,
        comfyui_host=req.comfyui_host,
        veo_timeout_sec=req.veo_timeout_sec,
        grok_timeout_sec=req.grok_timeout_sec,
        higgsfield_timeout_sec=req.higgsfield_timeout_sec,
        comfyui_timeout_sec=req.comfyui_timeout_sec,
        fallback_to_comfyui=req.fallback_to_comfyui,
    )
    return await dispatcher.dispatch_batch(
        job_response.scene_jobs, req.config, on_scene_done=on_scene_done
    )
