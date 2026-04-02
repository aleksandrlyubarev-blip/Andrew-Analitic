"""
bridge/comfyui_client.py
========================
Async httpx client for the ComfyUI prompt API.

Lifecycle for a single scene clip:
  1. submit()          — POST /prompt → prompt_id
  2. await_completion() — polls GET /history/{prompt_id} until done or timeout
  3. Result carries output file paths ready for download / Pino Cut pickup.

For a full scenario queue use submit_batch(): all prompts are enqueued first,
then polled concurrently (bounded by semaphore to avoid overwhelming ComfyUI).

ComfyUI API reference (local instance):
  POST  /prompt                    submit a workflow
  GET   /history/{prompt_id}       poll job status + outputs
  GET   /queue                     queue depth
  GET   /view?filename=&type=output download output file
  GET   /system_stats              health / VRAM info
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

import httpx

from bridge.schemas import (
    ComfyUIBatchResult,
    ComfyUIJobStatus,
    ComfyUIOutputFile,
    ComfyUISubmitResult,
)

logger = logging.getLogger("comfyui_client")

# ── defaults ──────────────────────────────────────────────────────────────────

DEFAULT_HOST = "http://127.0.0.1:8188"
DEFAULT_TIMEOUT_SEC = 600       # 10 min per scene — generous for 1080p
DEFAULT_POLL_INTERVAL_SEC = 4.0
DEFAULT_MAX_CONCURRENT_POLLS = 4

# ComfyUI history status strings
_STATUS_SUCCESS = "success"
_STATUS_ERROR = "error"


class ComfyUIClient:
    """
    Async client for ComfyUI's prompt-submission and history-polling API.

    Parameters
    ----------
    host:
        Base URL of the running ComfyUI instance (default: http://127.0.0.1:8188).
    timeout_sec:
        Per-job wall-clock timeout for polling (default: 600 s).
    poll_interval_sec:
        Seconds between /history polls (default: 4 s).
    max_concurrent_polls:
        Max parallel history polls in submit_batch() (default: 4).
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        poll_interval_sec: float = DEFAULT_POLL_INTERVAL_SEC,
        max_concurrent_polls: int = DEFAULT_MAX_CONCURRENT_POLLS,
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout_sec = timeout_sec
        self._poll_interval_sec = poll_interval_sec
        self._semaphore = asyncio.Semaphore(max_concurrent_polls)
        self._http = httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(30.0),
        )

    # ── public API ────────────────────────────────────────────────────────────

    async def health(self) -> dict[str, Any]:
        """Return ComfyUI system stats (VRAM, queue depth, version)."""
        resp = await self._http.get("/system_stats")
        resp.raise_for_status()
        stats = resp.json()
        queue_resp = await self._http.get("/queue")
        queue_resp.raise_for_status()
        queue = queue_resp.json()
        return {
            "comfyui": "ok",
            "system_stats": stats,
            "queue_running": len(queue.get("queue_running", [])),
            "queue_pending": len(queue.get("queue_pending", [])),
        }

    async def submit(
        self,
        workflow: dict[str, Any],
        client_id: str | None = None,
    ) -> str:
        """
        Submit one workflow to ComfyUI.

        Parameters
        ----------
        workflow:
            Raw ComfyUI node graph dict (output of ComfyUIWorkflowExporter.export_job()).
        client_id:
            Optional client identifier (auto-generated if omitted).

        Returns
        -------
        str
            The ``prompt_id`` assigned by ComfyUI.
        """
        cid = client_id or str(uuid.uuid4())
        payload = {"prompt": workflow, "client_id": cid}
        resp = await self._http.post("/prompt", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "error" in data:
            raise RuntimeError(f"ComfyUI rejected prompt: {data['error']}")
        prompt_id: str = data["prompt_id"]
        logger.info("Submitted prompt_id=%s client_id=%s", prompt_id, cid)
        return prompt_id

    async def await_completion(
        self,
        prompt_id: str,
        *,
        timeout_sec: float | None = None,
        poll_interval_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """
        Poll /history/{prompt_id} until the job completes or timeout fires.

        Returns a ComfyUISubmitResult with status and output file paths.
        """
        deadline = asyncio.get_event_loop().time() + (timeout_sec or self._timeout_sec)
        interval = poll_interval_sec or self._poll_interval_sec

        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return ComfyUISubmitResult(
                    prompt_id=prompt_id,
                    status=ComfyUIJobStatus.TIMEOUT,
                    error=f"Timed out after {timeout_sec or self._timeout_sec:.0f}s",
                )

            result = await self._poll_once(prompt_id)
            if result is not None:
                return result

            await asyncio.sleep(min(interval, remaining))

    async def submit_and_await(
        self,
        workflow: dict[str, Any],
        *,
        scene_id: str = "",
        client_id: str | None = None,
        timeout_sec: float | None = None,
    ) -> ComfyUISubmitResult:
        """Submit a workflow and wait for it to finish. Convenience wrapper."""
        prompt_id = await self.submit(workflow, client_id=client_id)
        result = await self.await_completion(prompt_id, timeout_sec=timeout_sec)
        result.scene_id = scene_id
        return result

    async def submit_batch(
        self,
        workflows: list[dict[str, Any]],
        *,
        scene_ids: list[str] | None = None,
        timeout_sec: float | None = None,
    ) -> ComfyUIBatchResult:
        """
        Submit all workflows then poll them concurrently (bounded concurrency).

        Parameters
        ----------
        workflows:
            List of ComfyUI node graph dicts — one per scene.
        scene_ids:
            Optional parallel list of scene IDs for result labelling.
        timeout_sec:
            Per-job timeout; defaults to self._timeout_sec.

        Returns
        -------
        ComfyUIBatchResult
        """
        sids = scene_ids or [f"scene_{i + 1:02d}" for i in range(len(workflows))]

        # Enqueue all first (ComfyUI queues them server-side anyway)
        prompt_ids: list[str] = []
        for wf in workflows:
            pid = await self.submit(wf)
            prompt_ids.append(pid)

        # Poll concurrently
        async def _bounded_await(pid: str, sid: str) -> ComfyUISubmitResult:
            async with self._semaphore:
                result = await self.await_completion(pid, timeout_sec=timeout_sec)
                result.scene_id = sid
                return result

        results = await asyncio.gather(
            *(_bounded_await(pid, sid) for pid, sid in zip(prompt_ids, sids)),
            return_exceptions=False,
        )

        succeeded = [r for r in results if r.status == ComfyUIJobStatus.SUCCESS]
        failed = [r for r in results if r.status != ComfyUIJobStatus.SUCCESS]

        return ComfyUIBatchResult(
            total=len(results),
            succeeded=len(succeeded),
            failed=len(failed),
            results=list(results),
        )

    def output_url(self, filename: str, subfolder: str = "", file_type: str = "output") -> str:
        """Build the /view URL for downloading a generated output file."""
        params = f"filename={filename}&subfolder={subfolder}&type={file_type}"
        return f"{self._host}/view?{params}"

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "ComfyUIClient":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── private ───────────────────────────────────────────────────────────────

    async def _poll_once(self, prompt_id: str) -> ComfyUISubmitResult | None:
        """
        Fetch /history/{prompt_id}.

        Returns None if the job isn't finished yet.
        Returns a ComfyUISubmitResult once it is.
        """
        try:
            resp = await self._http.get(f"/history/{prompt_id}")
            resp.raise_for_status()
            history: dict[str, Any] = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning("History poll HTTP error: %s", exc)
            return None
        except httpx.RequestError as exc:
            logger.warning("History poll connection error: %s", exc)
            return None

        if prompt_id not in history:
            return None  # not ready yet

        entry = history[prompt_id]
        job_status = entry.get("status", {})
        status_str: str = job_status.get("status_str", "")
        completed: bool = job_status.get("completed", False)

        if not completed:
            return None

        if status_str == _STATUS_ERROR:
            messages = job_status.get("messages", [])
            error_text = "; ".join(str(m) for m in messages) if messages else "ComfyUI reported error"
            logger.error("Job %s failed: %s", prompt_id, error_text)
            return ComfyUISubmitResult(
                prompt_id=prompt_id,
                status=ComfyUIJobStatus.ERROR,
                error=error_text,
            )

        # Parse output files from all nodes
        outputs: dict[str, Any] = entry.get("outputs", {})
        output_files = self._extract_output_files(outputs)

        logger.info(
            "Job %s completed — %d output file(s)",
            prompt_id,
            len(output_files),
        )
        return ComfyUISubmitResult(
            prompt_id=prompt_id,
            status=ComfyUIJobStatus.SUCCESS,
            output_files=output_files,
        )

    def _extract_output_files(
        self, outputs: dict[str, Any]
    ) -> list[ComfyUIOutputFile]:
        """
        Walk the outputs dict and collect all image/video file references.

        ComfyUI outputs look like:
          {"10": {"images": [{"filename": "ltx_scene_01_00001.mp4",
                              "subfolder": "", "type": "output"}]}}
        """
        files: list[ComfyUIOutputFile] = []
        for node_id, node_output in outputs.items():
            for key in ("images", "gifs", "videos"):
                for item in node_output.get(key, []):
                    fname = item.get("filename", "")
                    subfolder = item.get("subfolder", "")
                    ftype = item.get("type", "output")
                    if fname:
                        files.append(
                            ComfyUIOutputFile(
                                node_id=node_id,
                                filename=fname,
                                subfolder=subfolder,
                                file_type=ftype,
                                url=self.output_url(fname, subfolder, ftype),
                            )
                        )
        return files
