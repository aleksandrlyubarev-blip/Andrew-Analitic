"""
bridge/job_store.py
===================
In-memory async job store for long-running video dispatch jobs.

A job transitions through: queued → running → done | error

Thread/task safety: all mutations go through asyncio.Lock so the store is
safe to share across concurrent FastAPI request handlers.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from bridge.schemas import ComfyUIBatchResult


class VideoJobStatus(str, Enum):
    QUEUED  = "queued"
    RUNNING = "running"
    DONE    = "done"
    ERROR   = "error"


class VideoJobRecord(BaseModel):
    job_id: str
    status: VideoJobStatus = VideoJobStatus.QUEUED
    scenes_total: int = 0
    scenes_done: int = 0
    result: Optional[ComfyUIBatchResult] = None
    error: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @property
    def progress(self) -> str:
        if self.scenes_total == 0:
            return "0/0"
        return f"{self.scenes_done}/{self.scenes_total}"


class VideoJobStore:
    """
    Singleton in-process job store.

    Usage::

        store = VideoJobStore.get()
        job_id = store.create(scenes_total=6)
        store.start(job_id)
        store.tick(job_id)          # scenes_done += 1
        store.complete(job_id, result)
    """

    _instance: Optional["VideoJobStore"] = None

    def __init__(self) -> None:
        self._jobs: dict[str, VideoJobRecord] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get(cls) -> "VideoJobStore":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ── mutations ─────────────────────────────────────────────────────────────

    async def create(self, scenes_total: int, job_id: str | None = None) -> str:
        job_id = job_id or str(uuid.uuid4())
        async with self._lock:
            self._jobs[job_id] = VideoJobRecord(
                job_id=job_id,
                scenes_total=scenes_total,
            )
        return job_id

    async def start(self, job_id: str) -> None:
        async with self._lock:
            rec = self._jobs[job_id]
            rec.status = VideoJobStatus.RUNNING
            rec.updated_at = time.time()

    async def tick(self, job_id: str) -> None:
        """Increment scenes_done by 1 (called after each scene finishes)."""
        async with self._lock:
            rec = self._jobs[job_id]
            rec.scenes_done = min(rec.scenes_done + 1, rec.scenes_total)
            rec.updated_at = time.time()

    async def complete(self, job_id: str, result: ComfyUIBatchResult) -> None:
        async with self._lock:
            rec = self._jobs[job_id]
            rec.status = VideoJobStatus.DONE
            rec.scenes_done = rec.scenes_total
            rec.result = result
            rec.updated_at = time.time()

    async def fail(self, job_id: str, error: str) -> None:
        async with self._lock:
            rec = self._jobs[job_id]
            rec.status = VideoJobStatus.ERROR
            rec.error = error
            rec.updated_at = time.time()

    # ── queries ───────────────────────────────────────────────────────────────

    def get_job(self, job_id: str) -> Optional[VideoJobRecord]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[VideoJobRecord]:
        return sorted(self._jobs.values(), key=lambda r: r.created_at, reverse=True)
