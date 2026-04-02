"""
bridge/higgsfield_client.py
===========================
Async httpx client for the Higgsfield AI video generation API.

Supports models available on the $39/month Higgsfield plan:
    sora2     → "sora-2"       (Cinema quality, best prompt adherence)
    wan2.6    → "wan-2.6"      (Multi-character, Start+End frame mode)
    veo3.1    → "veo-3.1"      (High-speed action, Multi-Image Reference)

Lifecycle:
    1. submit()           — POST /v1/generation → job_id
    2. await_completion() — polls GET /v1/generation/{job_id} until done
    3. Result carries output_url (signed S3/CDN) + metadata

API base: https://api.higgsfield.ai
Auth:     Authorization: Bearer {HIGGSFIELD_API_KEY}

Update MODEL_MAP if Higgsfield renames model slugs in a future release.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import httpx

from bridge.schemas import (
    ComfyUIJobStatus,
    ComfyUIOutputFile,
    ComfyUISubmitResult,
    HiggsfieldGenerationRequest,
)

logger = logging.getLogger("higgsfield_client")

HIGGSFIELD_API_BASE = "https://api.higgsfield.ai"

# Maps our preferred_model slugs → Higgsfield API model identifiers
MODEL_MAP: dict[str, str] = {
    "sora2":  "sora-2",
    "wan2.6": "wan-2.6",
    "veo3.1": "veo-3.1",
    # aliases
    "sora-2": "sora-2",
    "wan-2.6": "wan-2.6",
    "veo-3.1": "veo-3.1",
}

DEFAULT_TIMEOUT_SEC = 900.0    # 15 min — cloud jobs can queue
DEFAULT_POLL_INTERVAL_SEC = 6.0
DEFAULT_FPS = 24
DEFAULT_CFG = 12.0             # guidance_scale for slow-mo / detail-heavy scenes


class HiggsfieldClient:
    """
    Async Higgsfield API client.

    Parameters
    ----------
    api_key:
        Higgsfield API key. Falls back to env var HIGGSFIELD_API_KEY.
    base_url:
        Override API base (useful for testing with a mock server).
    timeout_sec:
        Per-job wall-clock timeout for polling.
    poll_interval_sec:
        Seconds between status polls.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = HIGGSFIELD_API_BASE,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        self._api_key = api_key or os.getenv("HIGGSFIELD_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._poll_interval = poll_interval_sec
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers=self._auth_headers(),
            timeout=httpx.Timeout(30.0),
        )

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(self, req: HiggsfieldGenerationRequest) -> str:
        """
        Submit one generation job to Higgsfield.

        Returns
        -------
        str
            Higgsfield ``job_id``.
        """
        payload = self._build_payload(req)
        resp = await self._http.post("/v1/generation", json=payload)
        resp.raise_for_status()
        data = resp.json()
        job_id: str = data.get("job_id") or data.get("id", "")
        if not job_id:
            raise RuntimeError(f"Higgsfield returned no job_id: {data}")
        logger.info("Higgsfield job submitted: %s  model=%s", job_id, req.model)
        return job_id

    async def await_completion(
        self,
        job_id: str,
        *,
        scene_id: str = "",
        timeout_sec: float | None = None,
        poll_interval_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """
        Poll GET /v1/generation/{job_id} until the job finishes or times out.

        Returns a ComfyUISubmitResult so the dispatcher can treat Higgsfield
        and ComfyUI results uniformly.
        """
        deadline = asyncio.get_event_loop().time() + (timeout_sec or self._timeout_sec)
        interval = poll_interval_sec or self._poll_interval

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return ComfyUISubmitResult(
                    prompt_id=job_id,
                    scene_id=scene_id,
                    status=ComfyUIJobStatus.TIMEOUT,
                    error=f"Higgsfield job {job_id} timed out after {timeout_sec or self._timeout_sec:.0f}s",
                )

            result = await self._poll_once(job_id, scene_id)
            if result is not None:
                return result

            await asyncio.sleep(min(interval, remaining))

    async def submit_and_await(
        self,
        req: HiggsfieldGenerationRequest,
        *,
        timeout_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Submit and wait. Convenience wrapper."""
        job_id = await self.submit(req)
        return await self.await_completion(
            job_id,
            scene_id=req.scene_id,
            timeout_sec=timeout_sec,
        )

    async def health(self) -> dict[str, Any]:
        """Ping the Higgsfield API and return account / quota info."""
        try:
            resp = await self._http.get("/v1/account")
            resp.raise_for_status()
            return {"higgsfield": "ok", "account": resp.json()}
        except Exception as exc:
            return {"higgsfield": "unreachable", "error": str(exc)}

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "HiggsfieldClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── private ───────────────────────────────────────────────────────────────

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def _build_payload(self, req: HiggsfieldGenerationRequest) -> dict[str, Any]:
        model_slug = MODEL_MAP.get(req.model, req.model)
        payload: dict[str, Any] = {
            "model": model_slug,
            "prompt": req.prompt,
            "duration": req.duration_sec,
            "fps": req.fps,
            "resolution": req.resolution,
            "aspect_ratio": req.aspect_ratio,
            "guidance_scale": req.guidance_scale,
        }
        if req.negative_prompt:
            payload["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.start_frame_url:
            payload["start_frame_url"] = req.start_frame_url
        if req.end_frame_url:
            payload["end_frame_url"] = req.end_frame_url
        if req.reference_image_urls:
            payload["reference_image_urls"] = req.reference_image_urls
        if req.camera_motion:
            payload["camera_motion"] = req.camera_motion
        return payload

    async def _poll_once(
        self, job_id: str, scene_id: str
    ) -> ComfyUISubmitResult | None:
        try:
            resp = await self._http.get(f"/v1/generation/{job_id}")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Higgsfield poll HTTP %s: %s", exc.response.status_code, exc)
            return None
        except httpx.RequestError as exc:
            logger.warning("Higgsfield poll connection error: %s", exc)
            return None

        status_str: str = data.get("status", "")

        if status_str in ("queued", "processing", "pending", "running"):
            return None

        if status_str == "failed":
            error = data.get("error") or data.get("message") or "Higgsfield generation failed"
            logger.error("Higgsfield job %s failed: %s", job_id, error)
            return ComfyUISubmitResult(
                prompt_id=job_id,
                scene_id=scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=error,
            )

        if status_str == "completed":
            output_files = self._extract_outputs(job_id, data)
            logger.info(
                "Higgsfield job %s completed — %d output(s)", job_id, len(output_files)
            )
            return ComfyUISubmitResult(
                prompt_id=job_id,
                scene_id=scene_id,
                status=ComfyUIJobStatus.SUCCESS,
                output_files=output_files,
            )

        # Unknown status — keep polling
        logger.debug("Higgsfield job %s unknown status=%s", job_id, status_str)
        return None

    def _extract_outputs(
        self, job_id: str, data: dict[str, Any]
    ) -> list[ComfyUIOutputFile]:
        """
        Extract output video URLs from the Higgsfield response.

        Higgsfield returns either:
          {"output_url": "https://..."}
          {"outputs": [{"url": "...", "type": "video"}]}
        """
        files: list[ComfyUIOutputFile] = []

        # Single output_url
        url = data.get("output_url") or data.get("video_url") or ""
        if url:
            filename = url.split("/")[-1].split("?")[0] or f"{job_id}.mp4"
            files.append(
                ComfyUIOutputFile(
                    node_id="higgsfield",
                    filename=filename,
                    file_type="output",
                    url=url,
                )
            )
            return files

        # List of outputs
        for item in data.get("outputs", []):
            item_url = item.get("url", "")
            if not item_url:
                continue
            filename = item_url.split("/")[-1].split("?")[0] or f"{job_id}.mp4"
            files.append(
                ComfyUIOutputFile(
                    node_id="higgsfield",
                    filename=filename,
                    file_type=item.get("type", "output"),
                    url=item_url,
                )
            )
        return files
