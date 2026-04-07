"""
bridge/higgsfield_client.py
===========================
Async httpx client for the Higgsfield AI platform API (v2).

New API (platform.higgsfield.ai):
    Auth:   Authorization: Key {api_key_id}:{api_key_secret}
    Submit: POST /{app_model_slug}  →  {request_id, status_url, cancel_url}
    Poll:   GET  /requests/{request_id}/status  →  {status, video, ...}

App model slugs (DOP — image-to-video):
    higgsfield-ai/dop/lite
    higgsfield-ai/dop/standard
    higgsfield-ai/dop/turbo
    higgsfield-ai/dop/lite/first-last-frame
    higgsfield-ai/dop/standard/first-last-frame
    higgsfield-ai/dop/turbo/first-last-frame

Other models available on the platform:
    kling-video/v2.1/pro/image-to-video
    kling-video/v2.1/standard/image-to-video
    bytedance/seedance/v1/lite/image-to-video
    bytedance/seedance/v1/pro/image-to-video

Request body (all DOP models):
    image_url    str   — source image URL (required)
    prompt       str   — motion description
    duration     int   — seconds (3-10 typical)
    end_image_url str  — optional last frame (first-last-frame variants only)

Status values: queued | in_progress | completed | failed | nsfw | canceled

Env vars:
    HF_KEY         — combined "api_key_id:api_key_secret" (takes priority)
    HF_API_KEY     — API Key ID (UUID)
    HF_API_SECRET  — API Key Secret (hex string)
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

HIGGSFIELD_API_BASE = "https://platform.higgsfield.ai"

# Maps our model tags → Higgsfield app model slugs
# DOP tiers: lite (fastest/cheapest), standard, turbo (best quality)
MODEL_MAP: dict[str, str] = {
    # DOP — Higgsfield's flagship i2v model
    "dop":              "higgsfield-ai/dop/standard",
    "dop-lite":         "higgsfield-ai/dop/lite",
    "dop-turbo":        "higgsfield-ai/dop/turbo",
    # Kling
    "kling2.1":         "kling-video/v2.1/standard/image-to-video",
    "kling2.1-pro":     "kling-video/v2.1/pro/image-to-video",
    # Seedance v1 (API-only tier)
    "seedance":         "bytedance/seedance/v1/lite/image-to-video",
    "seedance-pro":     "bytedance/seedance/v1/pro/image-to-video",
    # Seedance v1.5 (Unlimited plan)
    "seedance1.5-pro":  "bytedance/seedance/v1.5/pro",
    "seedance1.5":      "bytedance/seedance/v1.5/lite",
    # Wan 2.6 (Unlimited plan)
    "wan2.6":           "wan/2.6/image-to-video",
    "wan2.6-t2v":       "wan/2.6/t2v",
}

# Terminal status values
_DONE_STATUSES = {"completed", "failed", "nsfw", "canceled"}
_ERROR_STATUSES = {"failed", "nsfw"}

DEFAULT_TIMEOUT_SEC    = 900.0
DEFAULT_POLL_INTERVAL  = 6.0


def _resolve_key() -> str:
    """Return the combined API credential string for the Authorization header."""
    hf_key = os.getenv("HF_KEY", "")
    if hf_key:
        return hf_key
    api_key    = os.getenv("HF_API_KEY", "")
    api_secret = os.getenv("HF_API_SECRET", "")
    if api_key and api_secret:
        return f"{api_key}:{api_secret}"
    return ""


class HiggsfieldClient:
    """
    Async client for the Higgsfield platform API (platform.higgsfield.ai).

    Parameters
    ----------
    api_key_id:
        API Key ID (UUID from cloud.higgsfield.ai → API Keys).
        Falls back to env HF_API_KEY.
    api_key_secret:
        API Key Secret.
        Falls back to env HF_API_SECRET.
        If ``hf_key`` is provided it overrides both.
    hf_key:
        Pre-combined credential string ``"{api_key_id}:{api_key_secret}"``.
        Falls back to env HF_KEY.
    base_url:
        Override API base (for testing with a mock server).
    timeout_sec:
        Per-job wall-clock timeout for polling.
    poll_interval_sec:
        Seconds between status polls.
    """

    def __init__(
        self,
        api_key_id: str | None = None,
        api_key_secret: str | None = None,
        *,
        hf_key: str | None = None,
        base_url: str = HIGGSFIELD_API_BASE,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL,
    ) -> None:
        # Build combined credential
        if hf_key:
            credential = hf_key
        elif api_key_id and api_key_secret:
            credential = f"{api_key_id}:{api_key_secret}"
        else:
            credential = _resolve_key()

        self._credential = credential
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._poll_interval = poll_interval_sec
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Key {credential}",
                "Content-Type": "application/json",
                "User-Agent": "higgsfield-client-py/1.0",
            },
            timeout=httpx.Timeout(30.0),
        )

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(self, req: HiggsfieldGenerationRequest) -> str:
        """
        Submit one generation job to Higgsfield.

        Returns
        -------
        str
            Higgsfield ``request_id``.
        """
        app_slug = self._resolve_slug(req.model)
        payload  = self._build_payload(req)

        resp = await self._http.post(f"/{app_slug}", json=payload)
        resp.raise_for_status()
        data = resp.json()

        request_id: str = data.get("request_id", "")
        if not request_id:
            raise RuntimeError(f"Higgsfield returned no request_id: {data}")

        logger.info("Higgsfield job submitted: %s  slug=%s", request_id, app_slug)
        return request_id

    async def await_completion(
        self,
        request_id: str,
        *,
        scene_id: str = "",
        timeout_sec: float | None = None,
        poll_interval_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """
        Poll GET /requests/{request_id}/status until done or timeout.

        Returns a ``ComfyUISubmitResult`` so the dispatcher treats Higgsfield
        and ComfyUI results uniformly.
        """
        deadline = asyncio.get_event_loop().time() + (timeout_sec or self._timeout_sec)
        interval = poll_interval_sec or self._poll_interval

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return ComfyUISubmitResult(
                    prompt_id=request_id,
                    scene_id=scene_id,
                    status=ComfyUIJobStatus.TIMEOUT,
                    error=f"Higgsfield job {request_id} timed out",
                )

            result = await self._poll_once(request_id, scene_id)
            if result is not None:
                return result

            await asyncio.sleep(min(interval, remaining))

    async def submit_and_await(
        self,
        req: HiggsfieldGenerationRequest,
        *,
        timeout_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Submit and wait. Convenience wrapper used by VideoDispatcher."""
        request_id = await self.submit(req)
        return await self.await_completion(
            request_id,
            scene_id=req.scene_id,
            timeout_sec=timeout_sec,
        )

    async def health(self) -> dict[str, Any]:
        """Ping the platform and return reachability status."""
        try:
            resp = await self._http.get("/health")
            return {"higgsfield": "ok", "http_status": resp.status_code}
        except Exception as exc:
            return {"higgsfield": "unreachable", "error": str(exc)}

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "HiggsfieldClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── private ───────────────────────────────────────────────────────────────

    def _resolve_slug(self, model: str) -> str:
        """
        Convert a friendly model tag to a Higgsfield app model slug.

        If the model string already contains "/" it's used as-is (allows
        callers to pass full slugs like "higgsfield-ai/dop/turbo").
        """
        if "/" in model:
            return model
        return MODEL_MAP.get(model, "higgsfield-ai/dop/standard")

    def _build_payload(self, req: HiggsfieldGenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "prompt": req.prompt,
            "duration": max(1, int(req.duration_sec)),
        }
        # image_url is required by all DOP models
        if req.start_frame_url:
            payload["image_url"] = req.start_frame_url
        # end_image_url supported by first-last-frame variants
        if req.end_frame_url:
            payload["end_image_url"] = req.end_frame_url
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.camera_motion:
            payload["camera_motion"] = req.camera_motion
        return payload

    async def _poll_once(
        self, request_id: str, scene_id: str
    ) -> ComfyUISubmitResult | None:
        try:
            resp = await self._http.get(f"/requests/{request_id}/status")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Higgsfield poll HTTP %s: %s", exc.response.status_code, exc)
            return None
        except httpx.RequestError as exc:
            logger.warning("Higgsfield poll connection error: %s", exc)
            return None

        status_str: str = data.get("status", "")

        if status_str not in _DONE_STATUSES:
            logger.debug("Higgsfield %s status=%s", request_id, status_str)
            return None

        if status_str in _ERROR_STATUSES:
            error = data.get("error") or data.get("message") or f"Higgsfield status={status_str}"
            logger.error("Higgsfield job %s → %s: %s", request_id, status_str, error)
            return ComfyUISubmitResult(
                prompt_id=request_id,
                scene_id=scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=error,
            )

        # completed or canceled
        output_files = self._extract_outputs(request_id, data)
        logger.info("Higgsfield job %s completed — %d output(s)", request_id, len(output_files))
        return ComfyUISubmitResult(
            prompt_id=request_id,
            scene_id=scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=output_files,
        )

    def _extract_outputs(
        self, request_id: str, data: dict[str, Any]
    ) -> list[ComfyUIOutputFile]:
        """
        Extract video URLs from a completed Higgsfield response.

        Platform returns:  {"status": "completed", "video": "https://..."}
        Fallback patterns: "output_url", "video_url", "outputs":[{"url":...}]
        """
        files: list[ComfyUIOutputFile] = []

        for field in ("video", "output_url", "video_url"):
            raw = data.get(field)
            # "video" is {"url": "..."}, others are plain strings
            url = raw.get("url", "") if isinstance(raw, dict) else (raw or "")
            if url:
                filename = url.split("/")[-1].split("?")[0] or f"{request_id}.mp4"
                files.append(ComfyUIOutputFile(
                    node_id="higgsfield",
                    filename=filename,
                    file_type="output",
                    url=url,
                ))
                return files

        for item in data.get("outputs", []):
            url = item.get("url", "")
            if url:
                filename = url.split("/")[-1].split("?")[0] or f"{request_id}.mp4"
                files.append(ComfyUIOutputFile(
                    node_id="higgsfield",
                    filename=filename,
                    file_type=item.get("type", "output"),
                    url=url,
                ))
        return files
