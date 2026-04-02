"""
bridge/grok_video_client.py
===========================
Async httpx client for xAI Grok 4.2 video generation API.

xAI API is OpenAI-compatible; video generation follows the same
submit → poll pattern as Higgsfield.

API base  : https://api.x.ai/v1
Auth      : Authorization: Bearer {XAI_API_KEY}
Model slug: grok-4.2-video  (update XAI_VIDEO_MODEL if xAI renames it)

Endpoints used:
    POST /v1/video/generations          submit generation job
    GET  /v1/video/generations/{id}     poll status + output URL

Status values returned by xAI:
    queued | processing | completed | failed

Update XAI_VIDEO_MODEL and _STATUS_* constants if xAI changes the
API surface in a future release.
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
    GrokVideoRequest,
)

logger = logging.getLogger("grok_video_client")

XAI_API_BASE = "https://api.x.ai/v1"
XAI_VIDEO_MODEL = "grok-4.2-video"

_TERMINAL_OK = {"completed", "succeeded", "success"}
_TERMINAL_ERR = {"failed", "error", "cancelled"}
_PENDING = {"queued", "processing", "pending", "running"}

DEFAULT_TIMEOUT_SEC = 900.0
DEFAULT_POLL_INTERVAL_SEC = 5.0


class GrokVideoClient:
    """
    Async client for xAI Grok 4.2 video generation.

    Returns ComfyUISubmitResult so the VideoDispatcher treats all
    three backends (ComfyUI, Higgsfield, Grok) uniformly.

    Parameters
    ----------
    api_key:
        xAI API key. Falls back to env var XAI_API_KEY.
    base_url:
        Override API base (useful for mock testing).
    timeout_sec / poll_interval_sec:
        Polling config.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = XAI_API_BASE,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        self._api_key = api_key or os.getenv("XAI_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._poll_interval = poll_interval_sec
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {self._api_key}"},
            timeout=httpx.Timeout(30.0),
        )

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(self, req: GrokVideoRequest) -> str:
        """
        POST /v1/video/generations → generation id.
        """
        payload = self._build_payload(req)
        resp = await self._http.post("/v1/video/generations", json=payload)
        resp.raise_for_status()
        data = resp.json()
        gen_id: str = data.get("id") or data.get("generation_id") or ""
        if not gen_id:
            raise RuntimeError(f"xAI returned no generation id: {data}")
        logger.info("Grok 4.2 video submitted: id=%s  scene=%s", gen_id, req.scene_id)
        return gen_id

    async def await_completion(
        self,
        gen_id: str,
        *,
        scene_id: str = "",
        timeout_sec: float | None = None,
        poll_interval_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Poll GET /v1/video/generations/{id} until done or timeout."""
        deadline = asyncio.get_event_loop().time() + (timeout_sec or self._timeout_sec)
        interval = poll_interval_sec or self._poll_interval

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return ComfyUISubmitResult(
                    prompt_id=gen_id,
                    scene_id=scene_id,
                    status=ComfyUIJobStatus.TIMEOUT,
                    error=f"Grok 4.2 job {gen_id} timed out after {timeout_sec or self._timeout_sec:.0f}s",
                )

            result = await self._poll_once(gen_id, scene_id)
            if result is not None:
                return result

            await asyncio.sleep(min(interval, remaining))

    async def submit_and_await(
        self,
        req: GrokVideoRequest,
        *,
        timeout_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Submit and wait. Convenience wrapper."""
        gen_id = await self.submit(req)
        return await self.await_completion(
            gen_id,
            scene_id=req.scene_id,
            timeout_sec=timeout_sec,
        )

    async def health(self) -> dict[str, Any]:
        """Ping xAI API and return model availability info."""
        try:
            resp = await self._http.get("/v1/models")
            resp.raise_for_status()
            models = [m.get("id") for m in resp.json().get("data", [])]
            available = XAI_VIDEO_MODEL in models
            return {
                "grok_video": "ok" if available else "model_not_listed",
                "xai_api": "reachable",
                "video_model": XAI_VIDEO_MODEL,
                "model_listed": available,
            }
        except Exception as exc:
            return {"grok_video": "unreachable", "error": str(exc)}

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "GrokVideoClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── private ───────────────────────────────────────────────────────────────

    def _build_payload(self, req: GrokVideoRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": XAI_VIDEO_MODEL,
            "prompt": req.prompt,
            "duration": req.duration_sec,
            "fps": req.fps,
            "resolution": req.resolution,
            "aspect_ratio": req.aspect_ratio,
        }
        if req.negative_prompt:
            payload["negative_prompt"] = req.negative_prompt
        if req.seed is not None:
            payload["seed"] = req.seed
        if req.camera_motion:
            payload["camera_motion"] = req.camera_motion
        if req.start_frame_url:
            payload["start_frame_url"] = req.start_frame_url
        if req.end_frame_url:
            payload["end_frame_url"] = req.end_frame_url
        return payload

    async def _poll_once(
        self, gen_id: str, scene_id: str
    ) -> ComfyUISubmitResult | None:
        try:
            resp = await self._http.get(f"/v1/video/generations/{gen_id}")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Grok poll HTTP %s: %s", exc.response.status_code, exc)
            return None
        except httpx.RequestError as exc:
            logger.warning("Grok poll connection error: %s", exc)
            return None

        status_str: str = (data.get("status") or "").lower()

        if status_str in _PENDING or not status_str:
            return None

        if status_str in _TERMINAL_ERR:
            error = data.get("error") or data.get("message") or "Grok 4.2 generation failed"
            logger.error("Grok job %s failed: %s", gen_id, error)
            return ComfyUISubmitResult(
                prompt_id=gen_id,
                scene_id=scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=error,
            )

        if status_str in _TERMINAL_OK:
            output_files = self._extract_outputs(gen_id, data)
            logger.info("Grok job %s completed — %d output(s)", gen_id, len(output_files))
            return ComfyUISubmitResult(
                prompt_id=gen_id,
                scene_id=scene_id,
                status=ComfyUIJobStatus.SUCCESS,
                output_files=output_files,
            )

        logger.debug("Grok job %s unknown status=%s", gen_id, status_str)
        return None

    def _extract_outputs(
        self, gen_id: str, data: dict[str, Any]
    ) -> list[ComfyUIOutputFile]:
        """
        Parse output video URL(s) from xAI response.

        xAI may return:
            {"output": {"url": "https://..."}}
            {"video_url": "https://..."}
            {"outputs": [{"url": "..."}]}
        """
        files: list[ComfyUIOutputFile] = []

        # Nested output object
        output = data.get("output") or {}
        url = output.get("url") if isinstance(output, dict) else ""
        url = url or data.get("video_url") or data.get("output_url") or ""
        if url:
            filename = url.split("/")[-1].split("?")[0] or f"{gen_id}.mp4"
            files.append(
                ComfyUIOutputFile(
                    node_id="grok_video",
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
            filename = item_url.split("/")[-1].split("?")[0] or f"{gen_id}.mp4"
            files.append(
                ComfyUIOutputFile(
                    node_id="grok_video",
                    filename=filename,
                    file_type=item.get("type", "output"),
                    url=item_url,
                )
            )
        return files
