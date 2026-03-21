"""
Tests for the frontend-facing SceneOps aggregator contract.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.scene_ops import PinoCutSceneOpsAggregator, build_demo_scene_ops_request
from bridge.schemas import SceneOpsAggregateRequest
from bridge.service import AndrewMoltisBridge


def _scene_ops_payload(**overrides):
    payload = {
        "project_id": "rfv_pinnocat_001",
        "scene_id": "scene_03",
        "scene_goal": "arrival at abandoned spaceport",
        "style_profile": "cinematic dark sci-fi",
        "editing_mode": "hybrid",
        "editing_template": "cinematic_montage",
        "target_duration_sec": 35.0,
        "actual_duration_sec": 32.8,
        "used_clips": ["c04", "c02", "c07", "c01", "c05"],
        "rejected_clips": ["c03", "c06", "c08"],
        "clip_scores": {
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
        "bridge_jobs": [
            {
                "job_id": "pinocut_98d2af04c1",
                "job_type": "bridge_shot",
                "status": "queued",
                "request_path": "output/pinocut_jobs/queued/pinocut_98d2af04c1.request.json",
            }
        ],
        "regeneration_jobs": [
            {
                "job_id": "pinocut_b7a4e2f019",
                "job_type": "restyle",
                "status": "completed_stub",
                "source_clip_id": "c02",
                "artifact_path": "output/pinocut_jobs/pinocut_b7a4e2f019/restyle.artifact.json",
            }
        ],
        "timeline": {
            "tracks": {
                "video": [
                    {"segment_id": "seg_01"},
                    {"segment_id": "seg_02"},
                    {"segment_id": "seg_03"},
                ]
            }
        },
        "updated_at": "2026-03-22T09:40:00+02:00",
    }
    payload.update(overrides)
    return payload


def test_scene_ops_aggregator_returns_frontend_shape():
    aggregator = PinoCutSceneOpsAggregator()
    request = SceneOpsAggregateRequest(**_scene_ops_payload())

    snapshot = aggregator.build_snapshot(request)

    assert snapshot["scene"]["sceneId"] == "scene_03"
    assert snapshot["scene"]["queueState"] == "waiting_bassito"
    assert snapshot["clipScores"]["c04"]["visualQuality"] == 4
    assert snapshot["andrew"]["qualityBreakdown"]["visual_quality"] == pytest.approx(4.33, rel=0.01)
    assert snapshot["bassitoJobs"][0]["jobType"] == "bridge_shot"
    assert snapshot["updatedAt"] == "2026-03-22T09:40:00+02:00"
    assert snapshot["source"] == "api"


@pytest.mark.asyncio
async def test_bridge_service_handles_scene_ops_snapshot():
    bridge = AndrewMoltisBridge(store_results_in_memory=False)

    snapshot = await bridge.handle_scene_ops(_scene_ops_payload())

    assert snapshot["scene"]["sceneId"] == "scene_03"
    assert snapshot["scene"]["queueState"] == "waiting_bassito"
    assert snapshot["andrew"]["hitlDecision"] == "skipped"
    assert snapshot["bassitoJobs"][0]["status"] == "queued"
    assert snapshot["source"] == "api"


def test_demo_scene_ops_request_matches_frontend_contract():
    request = build_demo_scene_ops_request()

    assert request.scene_id == "scene_03"
    assert request.used_clips == ["c04", "c02", "c07", "c01", "c05"]
    assert request.bridge_jobs[0]["job_type"] == "bridge_shot"
