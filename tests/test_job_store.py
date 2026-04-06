"""
tests/test_job_store.py
=======================
Tests for VideoJobStore (async in-memory job tracking).

J-series:
  J1.  create() returns a UUID-like job_id.
  J2.  Freshly created job has status=queued, scenes_done=0.
  J3.  start() transitions status to running.
  J4.  tick() increments scenes_done.
  J5.  tick() never exceeds scenes_total.
  J6.  complete() sets status=done, result, and scenes_done=total.
  J7.  fail() sets status=error and error message.
  J8.  get_job() returns None for unknown job_id.
  J9.  list_jobs() returns most recent first.
  J10. Two create() calls produce distinct job_ids.
  J11. progress property returns "X/Y" string.
  J12. VideoJobStore.get() returns the same singleton.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.job_store import VideoJobStatus, VideoJobStore
from bridge.schemas import ComfyUIBatchResult, ComfyUIJobStatus


def _fresh_store() -> VideoJobStore:
    """Return a brand-new store (not the singleton)."""
    return VideoJobStore()


@pytest.mark.asyncio
async def test_j1_create_returns_job_id():
    store = _fresh_store()
    job_id = await store.create(scenes_total=3)
    assert isinstance(job_id, str) and len(job_id) > 0


@pytest.mark.asyncio
async def test_j2_fresh_job_is_queued():
    store = _fresh_store()
    job_id = await store.create(scenes_total=6)
    rec = store.get_job(job_id)
    assert rec is not None
    assert rec.status == VideoJobStatus.QUEUED
    assert rec.scenes_done == 0
    assert rec.scenes_total == 6


@pytest.mark.asyncio
async def test_j3_start_sets_running():
    store = _fresh_store()
    job_id = await store.create(scenes_total=2)
    await store.start(job_id)
    assert store.get_job(job_id).status == VideoJobStatus.RUNNING


@pytest.mark.asyncio
async def test_j4_tick_increments_done():
    store = _fresh_store()
    job_id = await store.create(scenes_total=3)
    await store.start(job_id)
    await store.tick(job_id)
    await store.tick(job_id)
    assert store.get_job(job_id).scenes_done == 2


@pytest.mark.asyncio
async def test_j5_tick_never_exceeds_total():
    store = _fresh_store()
    job_id = await store.create(scenes_total=2)
    for _ in range(5):
        await store.tick(job_id)
    assert store.get_job(job_id).scenes_done == 2


@pytest.mark.asyncio
async def test_j6_complete_sets_done():
    store = _fresh_store()
    job_id = await store.create(scenes_total=2)
    result = ComfyUIBatchResult(total=2, succeeded=2, failed=0, results=[])
    await store.complete(job_id, result)
    rec = store.get_job(job_id)
    assert rec.status == VideoJobStatus.DONE
    assert rec.scenes_done == 2
    assert rec.result is not None
    assert rec.result.succeeded == 2


@pytest.mark.asyncio
async def test_j7_fail_sets_error():
    store = _fresh_store()
    job_id = await store.create(scenes_total=3)
    await store.fail(job_id, "network timeout")
    rec = store.get_job(job_id)
    assert rec.status == VideoJobStatus.ERROR
    assert "timeout" in rec.error


@pytest.mark.asyncio
async def test_j8_get_unknown_job_returns_none():
    store = _fresh_store()
    assert store.get_job("does-not-exist") is None


@pytest.mark.asyncio
async def test_j9_list_jobs_most_recent_first():
    store = _fresh_store()
    id1 = await store.create(scenes_total=1)
    await asyncio.sleep(0.01)
    id2 = await store.create(scenes_total=1)
    jobs = store.list_jobs()
    ids = [j.job_id for j in jobs]
    assert ids.index(id2) < ids.index(id1)  # id2 (newer) comes first


@pytest.mark.asyncio
async def test_j10_distinct_job_ids():
    store = _fresh_store()
    id1 = await store.create(scenes_total=1)
    id2 = await store.create(scenes_total=1)
    assert id1 != id2


@pytest.mark.asyncio
async def test_j11_progress_string():
    store = _fresh_store()
    job_id = await store.create(scenes_total=6)
    await store.tick(job_id)
    await store.tick(job_id)
    await store.tick(job_id)
    rec = store.get_job(job_id)
    assert rec.progress == "3/6"


def test_j12_singleton_returns_same_instance():
    # Reset singleton for test isolation
    VideoJobStore._instance = None
    a = VideoJobStore.get()
    b = VideoJobStore.get()
    assert a is b
    VideoJobStore._instance = None  # clean up
