"""
bridge/pinocut_review.py
========================
Deterministic scene reviewer for PinoCut / Pinnocat scene bundles.

Andrew does not try to edit video here. Instead, it acts as a quality-control
and HITL-friendly review layer over a scene bundle exported by PinoCut.
"""

from __future__ import annotations

from statistics import mean
from typing import Any

from bridge.schemas import SceneClipScore, SceneReviewRequest


class PinoCutSceneReviewer:
    """Summarize a scene bundle into structured QA outputs."""

    RUBRIC_FIELDS = (
        "visual_quality",
        "continuity_fit",
        "prompt_match",
        "motion_stability",
        "timeline_usefulness",
    )

    def review(self, request: SceneReviewRequest) -> dict[str, Any]:
        warnings: list[str] = []
        recommended_actions: list[str] = []

        if not request.used_clips:
            warnings.append("No approved clips were supplied for scene review.")
            recommended_actions.append("Rebuild the scene selection before exporting.")
            return {
                "project_id": request.project_id,
                "scene_id": request.scene_id,
                "success": False,
                "confidence": 0.0,
                "summary": (
                    f"Scene {request.scene_id} cannot be reviewed because the bundle has "
                    "no approved clips."
                ),
                "warnings": warnings,
                "recommended_actions": recommended_actions,
                "quality_breakdown": {},
            }

        approved_scores = {
            clip_id: score
            for clip_id, score in request.clip_scores.items()
            if clip_id in request.used_clips
        }
        quality_breakdown = self._quality_breakdown(approved_scores)
        warnings.extend(self._build_warnings(request, approved_scores, quality_breakdown))
        recommended_actions.extend(
            self._build_recommended_actions(request, approved_scores)
        )
        confidence = self._compute_confidence(
            request,
            approved_scores,
            quality_breakdown,
        )

        if not approved_scores:
            warnings.append("Andrew review is missing per-clip scorecards for approved clips.")
            recommended_actions.append("Export clip_scores with the scene bundle.")

        summary = self._build_summary(
            request,
            quality_breakdown,
            warnings,
            confidence,
        )

        return {
            "project_id": request.project_id,
            "scene_id": request.scene_id,
            "success": confidence >= 0.35,
            "confidence": confidence,
            "summary": summary,
            "warnings": warnings,
            "recommended_actions": recommended_actions,
            "quality_breakdown": quality_breakdown,
        }

    def _quality_breakdown(
        self,
        scores: dict[str, SceneClipScore],
    ) -> dict[str, float]:
        if not scores:
            return {}
        return {
            field: round(mean(getattr(score, field) for score in scores.values()), 2)
            for field in self.RUBRIC_FIELDS
        }

    def _build_warnings(
        self,
        request: SceneReviewRequest,
        scores: dict[str, SceneClipScore],
        quality_breakdown: dict[str, float],
    ) -> list[str]:
        warnings: list[str] = []
        if not scores:
            return warnings

        if quality_breakdown.get("motion_stability", 5.0) < 3.5:
            warnings.append("Average motion stability is below the preferred rough-cut threshold.")
        if quality_breakdown.get("visual_quality", 5.0) < 3.5:
            warnings.append("Average visual quality is soft for a clean scene export.")
        if quality_breakdown.get("continuity_fit", 5.0) < 3.5:
            warnings.append("Continuity fit suggests the clip set may feel uneven.")

        duration_delta = abs(request.actual_duration_sec - request.target_duration_sec)
        if request.target_duration_sec > 0 and duration_delta / request.target_duration_sec > 0.15:
            warnings.append(
                f"Scene duration drifts by {duration_delta:.1f}s from the target."
            )

        unique_flags = sorted(
            {
                flag
                for score in scores.values()
                for flag in score.risk_flags
            }
        )
        if unique_flags:
            warnings.append("Risk flags present: " + ", ".join(unique_flags) + ".")

        if request.bridge_jobs:
            warnings.append(
                f"{len(request.bridge_jobs)} bridge shot request(s) are still queued."
            )
        if request.regeneration_jobs:
            warnings.append(
                f"{len(request.regeneration_jobs)} regeneration job(s) are queued."
            )
        if request.rejected_clips:
            warnings.append(
                f"{len(request.rejected_clips)} clip(s) were rejected before assembly."
            )

        return warnings

    def _build_recommended_actions(
        self,
        request: SceneReviewRequest,
        scores: dict[str, SceneClipScore],
    ) -> list[str]:
        actions: list[str] = []
        for clip_id, score in scores.items():
            if score.recommended_action != "keep":
                actions.append(f"{clip_id}: {score.recommended_action}")

        if request.bridge_jobs:
            actions.append("Review bridge-shot output before final export.")
        if request.regeneration_jobs:
            actions.append("Run queued Bassito regeneration jobs and rebuild the rough cut.")

        duration_delta = abs(request.actual_duration_sec - request.target_duration_sec)
        if request.target_duration_sec > 0 and duration_delta / request.target_duration_sec > 0.15:
            actions.append("Rebalance trims so scene duration stays closer to target.")

        deduped: list[str] = []
        for action in actions:
            if action not in deduped:
                deduped.append(action)
        return deduped

    def _compute_confidence(
        self,
        request: SceneReviewRequest,
        scores: dict[str, SceneClipScore],
        quality_breakdown: dict[str, float],
    ) -> float:
        if not scores:
            return 0.0

        base = mean(quality_breakdown.values()) / 5.0 if quality_breakdown else 0.0
        penalty = 0.0

        duration_delta = abs(request.actual_duration_sec - request.target_duration_sec)
        if request.target_duration_sec > 0:
            drift_ratio = duration_delta / request.target_duration_sec
            if drift_ratio > 0.3:
                penalty += 0.2
            elif drift_ratio > 0.15:
                penalty += 0.1

        penalty += min(0.16, len(request.bridge_jobs) * 0.08)
        penalty += min(0.2, len(request.regeneration_jobs) * 0.05)

        if request.used_clips:
            reject_ratio = len(request.rejected_clips) / max(1, len(request.used_clips))
            penalty += min(0.15, reject_ratio * 0.1)

        if not quality_breakdown:
            penalty += 0.2
        if quality_breakdown.get("motion_stability", 5.0) < 3.0:
            penalty += 0.08
        if quality_breakdown.get("visual_quality", 5.0) < 3.0:
            penalty += 0.1

        return round(max(0.0, min(0.99, base - penalty)), 2)

    def _build_summary(
        self,
        request: SceneReviewRequest,
        quality_breakdown: dict[str, float],
        warnings: list[str],
        confidence: float,
    ) -> str:
        strongest = self._best_dimension(quality_breakdown)
        weakest = self._worst_dimension(quality_breakdown)
        segment_count = self._segment_count(request.timeline)

        summary = (
            f"Scene {request.scene_id} uses {len(request.used_clips)} approved clips"
            f" across {segment_count} timeline segment(s), landing at "
            f"{request.actual_duration_sec:.1f}s against a {request.target_duration_sec:.1f}s target. "
            f"Andrew confidence is {confidence:.0%}."
        )
        if strongest:
            summary += f" Strongest area: {strongest.replace('_', ' ')}."
        if weakest:
            summary += f" Weakest area: {weakest.replace('_', ' ')}."
        if warnings:
            summary += f" Main concern: {warnings[0]}"
        return summary

    def _best_dimension(self, quality_breakdown: dict[str, float]) -> str | None:
        if not quality_breakdown:
            return None
        return max(quality_breakdown, key=quality_breakdown.get)

    def _worst_dimension(self, quality_breakdown: dict[str, float]) -> str | None:
        if not quality_breakdown:
            return None
        return min(quality_breakdown, key=quality_breakdown.get)

    def _segment_count(self, timeline: dict[str, Any] | None) -> int:
        if not timeline:
            return 0
        tracks = timeline.get("tracks", {})
        video = tracks.get("video", [])
        return len(video) if isinstance(video, list) else 0
