"""
bridge/scene_ops.py
===================
Frontend-facing SceneOps contract for RomeoFlexVision.

This module adapts PinoCut scene bundles, Andrew QA output, and Bassito job
state into the single snapshot shape consumed by the frontend.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from bridge.pinocut_review import PinoCutSceneReviewer
from bridge.schemas import (
    SceneOpsAggregateRequest,
    SceneOpsAndrewReview,
    SceneOpsBassitoJob,
    SceneOpsClipScore,
    SceneOpsSceneBundle,
    SceneOpsSnapshotResponse,
)


class PinoCutSceneOpsAggregator:
    """Build the RomeoFlexVision SceneOps snapshot from Andrew + PinoCut data."""

    def __init__(self, reviewer: PinoCutSceneReviewer | None = None):
        self.reviewer = reviewer or PinoCutSceneReviewer()

    def build_snapshot(
        self,
        request: SceneOpsAggregateRequest,
        *,
        review: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        review = review or self.reviewer.review(request)
        bassito_jobs = self._collect_bassito_jobs(request)
        queue_state = self._derive_queue_state(review, bassito_jobs)

        snapshot = SceneOpsSnapshotResponse(
            scene=SceneOpsSceneBundle(
                sceneId=request.scene_id,
                sceneGoal=request.scene_goal,
                editingTemplate=request.editing_template,
                targetDurationSec=request.target_duration_sec,
                actualDurationSec=request.actual_duration_sec,
                usedClips=list(request.used_clips),
                rejectedClips=list(request.rejected_clips),
                queueState=queue_state,
            ),
            clipScores={
                clip_id: SceneOpsClipScore(
                    visualQuality=score.visual_quality,
                    continuityFit=score.continuity_fit,
                    promptMatch=score.prompt_match,
                    motionStability=score.motion_stability,
                    timelineUsefulness=score.timeline_usefulness,
                    recommendedAction=score.recommended_action,
                )
                for clip_id, score in request.clip_scores.items()
            },
            andrew=SceneOpsAndrewReview(
                confidence=review.get("confidence", 0.0),
                summary=review.get("summary", ""),
                warnings=list(review.get("warnings", [])),
                recommendedActions=list(review.get("recommended_actions", [])),
                qualityBreakdown=dict(review.get("quality_breakdown", {})),
                hitlDecision=review.get("hitl_decision", "skipped"),
            ),
            bassitoJobs=bassito_jobs,
            updatedAt=request.updated_at or datetime.now(timezone.utc).isoformat(),
            source="api",
        )
        return snapshot.model_dump()

    def _collect_bassito_jobs(
        self,
        request: SceneOpsAggregateRequest,
    ) -> list[SceneOpsBassitoJob]:
        if request.bassito_jobs:
            return [
                SceneOpsBassitoJob(
                    jobId=job.job_id or self._fallback_job_id(request.scene_id, job.job_type, index),
                    jobType=job.job_type,
                    status=job.status,
                    sourceClipId=job.source_clip_id,
                    artifactPath=job.artifact_path or job.request_path,
                )
                for index, job in enumerate(request.bassito_jobs, start=1)
            ]

        derived_jobs: list[SceneOpsBassitoJob] = []
        for index, job in enumerate(request.bridge_jobs, start=1):
            derived_jobs.append(
                self._normalize_job_payload(
                    scene_id=request.scene_id,
                    payload=job,
                    default_job_type="bridge_shot",
                    index=index,
                )
            )

        for index, job in enumerate(request.regeneration_jobs, start=1):
            derived_jobs.append(
                self._normalize_job_payload(
                    scene_id=request.scene_id,
                    payload=job,
                    default_job_type="restyle",
                    index=index,
                )
            )

        return derived_jobs

    def _normalize_job_payload(
        self,
        *,
        scene_id: str,
        payload: dict[str, Any],
        default_job_type: str,
        index: int,
    ) -> SceneOpsBassitoJob:
        job_type = str(payload.get("job_type") or default_job_type)
        job_id = str(payload.get("job_id") or payload.get("id") or self._fallback_job_id(scene_id, job_type, index))
        return SceneOpsBassitoJob(
            jobId=job_id,
            jobType=job_type,
            status=str(payload.get("status") or "queued"),
            sourceClipId=payload.get("source_clip_id"),
            artifactPath=payload.get("artifact_path") or payload.get("result_path") or payload.get("request_path"),
        )

    def _derive_queue_state(
        self,
        review: dict[str, Any],
        bassito_jobs: list[SceneOpsBassitoJob],
    ) -> str:
        active_statuses = {"queued", "running", "accepted"}
        completed_statuses = {"completed", "completed_stub", "success"}

        if any(job.status in active_statuses for job in bassito_jobs):
            return "waiting_bassito"

        hitl_decision = review.get("hitl_decision", "skipped")
        if hitl_decision in {"modify", "reject"}:
            return "reviewing"

        if bassito_jobs and all(job.status in completed_statuses for job in bassito_jobs):
            return "completed"

        return "ready" if review.get("success", False) else "reviewing"

    def _fallback_job_id(self, scene_id: str, job_type: str, index: int) -> str:
        return f"{scene_id}_{job_type}_{index:02d}"


def build_demo_scene_ops_request() -> SceneOpsAggregateRequest:
    """Deterministic sample payload for frontend development and contract docs."""
    return SceneOpsAggregateRequest(
        project_id="rfv_pinnocat_001",
        scene_id="scene_03",
        scene_goal="arrival at abandoned spaceport",
        style_profile="cinematic dark sci-fi",
        editing_mode="hybrid",
        editing_template="cinematic_montage",
        target_duration_sec=35.0,
        actual_duration_sec=32.8,
        used_clips=["c04", "c02", "c07", "c01", "c05"],
        rejected_clips=["c03", "c06", "c08"],
        clip_scores={
            "c04": {
                "visual_quality": 4,
                "continuity_fit": 5,
                "prompt_match": 4,
                "motion_stability": 4,
                "timeline_usefulness": 5,
                "recommended_action": "keep",
            },
            "c02": {
                "visual_quality": 4,
                "continuity_fit": 4,
                "prompt_match": 4,
                "motion_stability": 2,
                "timeline_usefulness": 4,
                "risk_flags": ["excessive_camera_shake"],
                "recommended_action": "request_restyle",
            },
            "c07": {
                "visual_quality": 5,
                "continuity_fit": 4,
                "prompt_match": 5,
                "motion_stability": 4,
                "timeline_usefulness": 4,
                "risk_flags": ["unstable_lighting"],
                "recommended_action": "trim_for_pacing",
            },
        },
        bridge_jobs=[
            {
                "job_id": "pinocut_98d2af04c1",
                "job_type": "bridge_shot",
                "status": "queued",
                "request_path": "output/pinocut_jobs/queued/pinocut_98d2af04c1.request.json",
            }
        ],
        regeneration_jobs=[
            {
                "job_id": "pinocut_b7a4e2f019",
                "job_type": "restyle",
                "status": "completed_stub",
                "source_clip_id": "c02",
                "artifact_path": "output/pinocut_jobs/pinocut_b7a4e2f019/restyle.artifact.json",
            },
            {
                "job_id": "pinocut_c31fb66780",
                "job_type": "extend",
                "status": "queued",
                "source_clip_id": "c05",
                "request_path": "output/pinocut_jobs/queued/pinocut_c31fb66780.request.json",
            },
        ],
        timeline={
            "tracks": {
                "video": [
                    {"segment_id": "seg_01"},
                    {"segment_id": "seg_02"},
                    {"segment_id": "seg_03"},
                    {"segment_id": "seg_04"},
                    {"segment_id": "seg_05"},
                ]
            }
        },
        updated_at="2026-03-22T09:40:00+02:00",
    )
