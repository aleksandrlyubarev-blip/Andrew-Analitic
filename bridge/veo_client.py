"""
bridge/veo_client.py
====================
Async httpx client for Google Veo 3.1 video generation via Google AI Studio.

Auth:     x-goog-api-key: {GOOGLE_API_KEY}
Base URL: https://generativelanguage.googleapis.com

Generation uses long-running operations (LRO):
  1. POST /v1beta/models/veo-3.1-generate:predictLongRunning
     → {"name": "operations/abc123", ...}

  2. GET /v1beta/operations/{operation_name}
     → {"done": true, "response": {"videos": [{"uri": "..."}]}}
     until done == true

Update VEO_MODEL and VEO_API_BASE if Google changes the endpoint in a future
release (the LRO polling pattern will remain the same).

Vertex AI alternative:
  Set base_url="https://{region}-aiplatform.googleapis.com/v1"
  and use OAuth2 instead of API key.
  Endpoint: /projects/{project}/locations/{region}/publishers/google/models/veo-3.1:predictLongRunning
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
    VeoVideoRequest,
)

logger = logging.getLogger("veo_client")

VEO_API_BASE = "https://generativelanguage.googleapis.com"
VEO_MODEL = "veo-3.1-generate"         # update when Google bumps model slug
VEO_ENDPOINT = f"/v1beta/models/{VEO_MODEL}:predictLongRunning"
VEO_OPS_PREFIX = "/v1beta/operations/"

DEFAULT_TIMEOUT_SEC = 900.0
DEFAULT_POLL_INTERVAL_SEC = 8.0        # Veo jobs typically take 2-5 min cloud-side

_DONE_STATUSES = {"done"}              # operation.done == True → finished


class VeoVideoClient:
    """
    Async Google Veo 3.1 client using the AI Studio long-running operations API.

    Parameters
    ----------
    api_key:
        Google AI Studio API key. Falls back to env var GOOGLE_API_KEY.
    base_url:
        Override API base (Vertex AI or mock server).
    timeout_sec / poll_interval_sec:
        Polling config.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = VEO_API_BASE,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
    ) -> None:
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY", "")
        self._base_url = base_url.rstrip("/")
        self._timeout_sec = timeout_sec
        self._poll_interval = poll_interval_sec
        self._http = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"x-goog-api-key": self._api_key},
            timeout=httpx.Timeout(30.0),
        )

    # ── public API ────────────────────────────────────────────────────────────

    async def submit(self, req: VeoVideoRequest) -> str:
        """
        POST to /v1beta/models/veo-3.1-generate:predictLongRunning.

        Returns the LRO operation name (e.g. "operations/abc123").
        """
        payload = self._build_payload(req)
        resp = await self._http.post(VEO_ENDPOINT, json=payload)
        resp.raise_for_status()
        data = resp.json()
        op_name: str = data.get("name", "")
        if not op_name:
            raise RuntimeError(f"Veo returned no operation name: {data}")
        logger.info("Veo 3.1 operation started: %s  scene=%s", op_name, req.scene_id)
        return op_name

    async def await_completion(
        self,
        op_name: str,
        *,
        scene_id: str = "",
        timeout_sec: float | None = None,
        poll_interval_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Poll GET /v1beta/operations/{op_name} until done or timeout."""
        deadline = asyncio.get_event_loop().time() + (timeout_sec or self._timeout_sec)
        interval = poll_interval_sec or self._poll_interval

        # Strip full path prefix if caller passed the raw operation name
        op_id = op_name.split("/")[-1] if "/" in op_name else op_name

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return ComfyUISubmitResult(
                    prompt_id=op_name,
                    scene_id=scene_id,
                    status=ComfyUIJobStatus.TIMEOUT,
                    error=f"Veo 3.1 operation {op_name} timed out after {timeout_sec or self._timeout_sec:.0f}s",
                )

            result = await self._poll_once(op_name, op_id, scene_id)
            if result is not None:
                return result

            await asyncio.sleep(min(interval, remaining))

    async def submit_and_await(
        self,
        req: VeoVideoRequest,
        *,
        timeout_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Submit and wait. Convenience wrapper."""
        op_name = await self.submit(req)
        return await self.await_completion(
            op_name,
            scene_id=req.scene_id,
            timeout_sec=timeout_sec,
        )

    async def health(self) -> dict[str, Any]:
        """List available Veo models via AI Studio."""
        try:
            resp = await self._http.get("/v1beta/models")
            resp.raise_for_status()
            models = resp.json().get("models", [])
            veo_models = [m["name"] for m in models if "veo" in m.get("name", "").lower()]
            return {
                "veo": "ok",
                "google_ai_studio": "reachable",
                "veo_models_found": veo_models,
            }
        except Exception as exc:
            return {"veo": "unreachable", "error": str(exc)}

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "VeoVideoClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── private ───────────────────────────────────────────────────────────────

    def _build_payload(self, req: VeoVideoRequest) -> dict[str, Any]:
        """
        Build the Veo /predictLongRunning request body.

        Google Veo API format:
        {
          "instances": [{"prompt": "..."}],
          "parameters": {
            "aspectRatio": "16:9",
            "durationSeconds": 8,
            "fps": 24,
            "resolution": "1920x1080",
            "negativePrompt": "...",
            "seed": 42,
            "referenceImages": [{"image": {"uri": "..."}, "referenceType": "SUBJECT"}]
          }
        }
        """
        parameters: dict[str, Any] = {
            "aspectRatio": req.aspect_ratio,
            "durationSeconds": req.duration_sec,
            "fps": req.fps,
            "resolution": req.resolution,
        }
        if req.negative_prompt:
            parameters["negativePrompt"] = req.negative_prompt
        if req.seed is not None:
            parameters["seed"] = req.seed
        if req.camera_motion:
            parameters["cameraControl"] = req.camera_motion

        # First/last frame conditioning
        instances: list[dict[str, Any]] = [{"prompt": req.prompt}]
        if req.start_frame_url:
            instances[0]["image"] = {"uri": req.start_frame_url}
        if req.end_frame_url:
            parameters["lastFrame"] = {"uri": req.end_frame_url}

        # Reference images (subject / style)
        if req.reference_image_urls:
            parameters["referenceImages"] = [
                {"image": {"uri": uri}, "referenceType": "SUBJECT"}
                for uri in req.reference_image_urls
            ]

        return {"instances": instances, "parameters": parameters}

    async def _poll_once(
        self, op_name: str, op_id: str, scene_id: str
    ) -> ComfyUISubmitResult | None:
        """Poll GET /v1beta/operations/{op_id}. Returns None if still pending."""
        try:
            resp = await self._http.get(f"{VEO_OPS_PREFIX}{op_id}")
            resp.raise_for_status()
            data: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("Veo poll HTTP %s: %s", exc.response.status_code, exc)
            return None
        except httpx.RequestError as exc:
            logger.warning("Veo poll connection error: %s", exc)
            return None

        if not data.get("done", False):
            return None

        # done == True: check for error
        if "error" in data:
            err = data["error"]
            error_msg = err.get("message") or str(err)
            logger.error("Veo operation %s failed: %s", op_name, error_msg)
            return ComfyUISubmitResult(
                prompt_id=op_name,
                scene_id=scene_id,
                status=ComfyUIJobStatus.ERROR,
                error=error_msg,
            )

        # done == True, no error → extract videos
        output_files = self._extract_outputs(op_name, data)
        logger.info("Veo operation %s completed — %d video(s)", op_name, len(output_files))
        return ComfyUISubmitResult(
            prompt_id=op_name,
            scene_id=scene_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=output_files,
        )

    def _extract_outputs(
        self, op_name: str, data: dict[str, Any]
    ) -> list[ComfyUIOutputFile]:
        """
        Extract video URIs from Veo LRO response.

        Google Veo response format:
          {"done": true, "response": {
              "generateVideoResponse": {
                  "generatedSamples": [{"video": {"uri": "https://..."}}]
              }
          }}
        Also handles flat: {"done": true, "response": {"videos": [{"uri": "..."}]}}
        """
        files: list[ComfyUIOutputFile] = []
        response = data.get("response", {})

        # Primary format
        gvr = response.get("generateVideoResponse", {})
        for sample in gvr.get("generatedSamples", []):
            uri = sample.get("video", {}).get("uri", "")
            if uri:
                filename = uri.split("/")[-1].split("?")[0] or f"{op_name}.mp4"
                files.append(ComfyUIOutputFile(
                    node_id="veo3.1",
                    filename=filename,
                    file_type="output",
                    url=uri,
                ))

        # Flat format fallback
        if not files:
            for video in response.get("videos", []):
                uri = video.get("uri", "")
                if uri:
                    filename = uri.split("/")[-1].split("?")[0] or f"{op_name}.mp4"
                    files.append(ComfyUIOutputFile(
                        node_id="veo3.1",
                        filename=filename,
                        file_type="output",
                        url=uri,
                    ))

        return files
