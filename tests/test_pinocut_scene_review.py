"""
Tests for the Andrew <-> PinoCut scene review bridge.
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.hitl import HitlOutcome
from bridge.pinocut_review import PinoCutSceneReviewer
from bridge.schemas import SceneReviewRequest
from bridge.service import AndrewMoltisBridge


def _scene_payload(**overrides):
    payload = {
        "project_id": "rfv_pinnocat_001",
        "scene_id": "scene_03",
        "scene_goal": "arrival at abandoned spaceport",
        "style_profile": "cinematic dark sci-fi",
        "editing_mode": "hybrid",
        "editing_template": "cinematic_montage",
        "target_duration_sec": 35.0,
        "actual_duration_sec": 31.0,
        "used_clips": ["c04", "c02", "c07"],
        "rejected_clips": ["c03"],
        "clip_scores": {
            "c04": {
                "visual_quality": 4,
                "continuity_fit": 5,
                "prompt_match": 4,
                "motion_stability": 3,
                "timeline_usefulness": 5,
                "risk_flags": [],
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
        "bridge_jobs": [{"job_type": "bridge_shot", "status": "queued"}],
        "regeneration_jobs": [{"job_type": "restyle", "status": "queued"}],
        "timeline": {
            "tracks": {
                "video": [
                    {"segment_id": "seg_01"},
                    {"segment_id": "seg_02"},
                    {"segment_id": "seg_03"},
                ]
            }
        },
    }
    payload.update(overrides)
    return payload


def test_scene_reviewer_returns_structured_qa_summary():
    reviewer = PinoCutSceneReviewer()
    request = SceneReviewRequest(**_scene_payload())

    result = reviewer.review(request)

    assert result["success"] is True
    assert result["quality_breakdown"]["visual_quality"] == pytest.approx(4.33, rel=0.01)
    assert "Strongest area" in result["summary"]
    assert any("bridge shot" in warning.lower() for warning in result["warnings"])
    assert any("Bassito regeneration" in action for action in result["recommended_actions"])


def test_scene_reviewer_fails_when_no_approved_clips_are_present():
    reviewer = PinoCutSceneReviewer()
    request = SceneReviewRequest(**_scene_payload(used_clips=[], clip_scores={}))

    result = reviewer.review(request)

    assert result["success"] is False
    assert result["confidence"] == 0.0
    assert "no approved clips" in result["summary"].lower()


@pytest.mark.asyncio
async def test_bridge_service_handles_scene_review_and_hitl():
    bridge = AndrewMoltisBridge(store_results_in_memory=False)
    bridge.hitl.check = AsyncMock(
        return_value=HitlOutcome(
            triggered=True,
            decision="approve",
            output="Human-reviewed scene summary.",
            warnings=["Reviewer note: looks good after trim."],
            review_id="review-123",
        )
    )

    result = await bridge.handle_scene_review(_scene_payload())

    assert result["scene_id"] == "scene_03"
    assert result["summary"] == "Human-reviewed scene summary."
    assert result["hitl_decision"] == "approve"
    assert result["hitl_review_id"] == "review-123"
    assert result["formatted_message"].startswith("**Scene Review**")
