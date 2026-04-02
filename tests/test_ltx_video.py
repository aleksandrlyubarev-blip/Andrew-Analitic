"""
tests/test_ltx_video.py
=======================
Tests for the LTX 2.3 video generation pipeline.

Coverage:
  P1.  Scenario with explicit scene headers + all three tags → correct fields.
  P2.  Cyrillic scene headers (Сцена 1).
  P3.  Scenes longer than threshold → multi_keyframe workflow.
  P4.  Scenes at or below threshold → first_last workflow.
  P5.  No timing in header → duration falls back to max_clip_duration_sec cap.
  P6.  Missing VISUAL tag → first non-header line used as visual prompt.
  P7.  9:16 aspect ratio propagates into model_config_ltx.
  P8.  VRAM warning fires when estimate exceeds budget.
  P9.  Empty scenario → zero jobs + warning.
  P10. LtxGenerateTool.run() returns ToolResult with output and summary text.
  P11. double_upscale sets generation_resolution in model_config_ltx.
  P12. Multi-keyframe scene has correct keyframe count.
  P13. Job prompts include style when STYLE tag is present.
  P14. project_id is prefixed in job_id.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.ltx_video import LtxJobBuilder, LtxScenarioParser, LtxVideoPipeline
from bridge.schemas import LtxGenerationConfig, LtxVideoJobRequest
from core.tools.base import ToolUseContext
from core.tools.ltx_generate import LtxGenerateTool

# ── shared fixtures ───────────────────────────────────────────────────────────

SCENARIO_EN = """\
## Scene 1 — Neon Corridor (0:00–0:08)
[VISUAL: man walking through neon-lit corridor, slow motion]
[STYLE: cinematic neon-tech]
[AUDIO: ambient hum + footsteps echo]

## Scene 2 — Hologram Reveal (0:08–0:22)
[VISUAL: close-up of holographic display activating, sparks of light]
[STYLE: cinematic neon-tech]
[AUDIO: UI activation sound + voiceover: "The system is online"]
"""

SCENARIO_RU = """\
Сцена 1 (0:00-0:10)
[VISUAL: человек идёт по неоновому коридору]
[STYLE: sci-fi cinematic]
[AUDIO: фоновый шум]

Сцена 2 (0:10-0:25)
[VISUAL: голографический дисплей активируется]
[STYLE: sci-fi cinematic]
[AUDIO: звук интерфейса]
"""

DEFAULT_CONFIG = LtxGenerationConfig()
SHORTS_CONFIG = LtxGenerationConfig(aspect_ratio="9:16", resolution="1080x1920")


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse(text: str, config: LtxGenerationConfig | None = None) -> list:
    return LtxScenarioParser().parse(text, config or DEFAULT_CONFIG)


def _pipeline(text: str, project_id: str = "test_proj", config: LtxGenerationConfig | None = None):
    req = LtxVideoJobRequest(
        project_id=project_id,
        scenario_text=text,
        config=config or DEFAULT_CONFIG,
    )
    return LtxVideoPipeline().run(req)


# ── P1: happy path (EN scenario) ──────────────────────────────────────────────

def test_p1_en_scenario_basic_fields():
    scenes = _parse(SCENARIO_EN)
    assert len(scenes) == 2

    s1 = scenes[0]
    assert s1.scene_id == "scene_01"
    assert "neon" in s1.visual_prompt.lower()
    assert s1.style == "cinematic neon-tech"
    assert "footsteps" in s1.audio_description
    assert s1.duration_sec == pytest.approx(8.0, abs=0.1)

    s2 = scenes[1]
    assert "holographic" in s2.visual_prompt.lower()
    assert s2.duration_sec == pytest.approx(14.0, abs=0.1)


# ── P2: Cyrillic headers ──────────────────────────────────────────────────────

def test_p2_cyrillic_headers():
    scenes = _parse(SCENARIO_RU)
    assert len(scenes) == 2
    assert "неоновому" in scenes[0].visual_prompt or "человек" in scenes[0].visual_prompt
    assert scenes[1].duration_sec == pytest.approx(15.0, abs=0.1)  # capped at max


# ── P3: long scene → multi_keyframe ──────────────────────────────────────────

def test_p3_long_scene_multi_keyframe():
    scenario = """\
## Scene 1 (0:00–0:14)
[VISUAL: epic battle sequence with multiple characters and explosions]
[STYLE: dark fantasy]
"""
    scenes = _parse(scenario)
    assert scenes[0].workflow == "multi_keyframe"
    assert len(scenes[0].keyframes) >= 3


# ── P4: short scene → first_last ─────────────────────────────────────────────

def test_p4_short_scene_first_last():
    scenario = """\
## Scene 1 (0:00–0:08)
[VISUAL: close-up of an eye opening]
[STYLE: cinematic]
"""
    scenes = _parse(scenario)
    assert scenes[0].workflow == "first_last"
    kf_types = [kf.frame_type for kf in scenes[0].keyframes]
    assert kf_types == ["first", "last"]


# ── P5: no timing → fallback duration ────────────────────────────────────────

def test_p5_no_timing_fallback_duration():
    scenario = """\
## Scene 1
[VISUAL: cityscape at night]
"""
    config = LtxGenerationConfig(max_clip_duration_sec=12.0)
    scenes = _parse(scenario, config)
    assert scenes[0].duration_sec <= 12.0
    assert scenes[0].duration_sec > 0


# ── P6: missing VISUAL tag → first prose line used ───────────────────────────

def test_p6_no_visual_tag_fallback():
    scenario = """\
## Scene 1 (0:00–0:06)
A mysterious figure emerges from the fog.
"""
    scenes = _parse(scenario)
    assert "mysterious" in scenes[0].visual_prompt.lower()


# ── P7: 9:16 config propagates ───────────────────────────────────────────────

def test_p7_shorts_aspect_ratio():
    resp = _pipeline(SCENARIO_EN, config=SHORTS_CONFIG)
    for job in resp.scene_jobs:
        assert job.model_config_ltx["aspect_ratio"] == "9:16"
        assert job.model_config_ltx["resolution"] == "1080x1920"


# ── P8: VRAM warning ─────────────────────────────────────────────────────────

def test_p8_vram_warning():
    # bf16 without double_upscale at 1080p pushes ~22 GB → exceeds 12 GB budget
    heavy_config = LtxGenerationConfig(
        model_variant="bf16",
        double_upscale=False,
        vram_budget_gb=12,
    )
    resp = _pipeline(SCENARIO_EN, config=heavy_config)
    assert any("VRAM" in w for w in resp.warnings)


# ── P9: empty scenario ────────────────────────────────────────────────────────

def test_p9_empty_scenario():
    resp = _pipeline("   \n\n  ")
    assert resp.total_scenes == 0
    assert not resp.comfyui_queue_ready
    assert resp.warnings


# ── P10: tool run() returns ToolResult ───────────────────────────────────────

@pytest.mark.asyncio
async def test_p10_tool_run():
    tool = LtxGenerateTool()
    ctx = ToolUseContext()
    req = LtxVideoJobRequest(project_id="tool_test", scenario_text=SCENARIO_EN)
    result = await tool.run(req, ctx)

    assert result.success
    assert result.output is not None
    assert result.output.total_scenes == 2
    assert result.output_to_model is not None
    assert "tool_test" in result.output_to_model


# ── P11: double_upscale sets generation_resolution ───────────────────────────

def test_p11_double_upscale_generation_res():
    config = LtxGenerationConfig(double_upscale=True)
    resp = _pipeline(SCENARIO_EN, config=config)
    for job in resp.scene_jobs:
        assert "generation_resolution" in job.model_config_ltx
        assert job.model_config_ltx["upscale_pass"] == "double"


# ── P12: multi_keyframe keyframe count ───────────────────────────────────────

def test_p12_multi_keyframe_count():
    scenario = """\
## Scene 1 (0:00–0:20)
[VISUAL: sweeping landscape pan from mountains to ocean]
[STYLE: epic cinematic]
"""
    config = LtxGenerationConfig(multi_keyframe_threshold_sec=10.0)
    scenes = _parse(scenario, config)
    assert scenes[0].workflow == "multi_keyframe"
    # expect first + at least 1 middle + last = at least 3
    assert len(scenes[0].keyframes) >= 3
    frame_types = [kf.frame_type for kf in scenes[0].keyframes]
    assert frame_types[0] == "first"
    assert frame_types[-1] == "last"
    assert "middle" in frame_types


# ── P13: style included in prompt ────────────────────────────────────────────

def test_p13_style_in_prompt():
    scenario = """\
## Scene 1 (0:00–0:08)
[VISUAL: robot assembling itself]
[STYLE: hard sci-fi industrial]
"""
    resp = _pipeline(scenario)
    assert len(resp.scene_jobs) == 1
    job = resp.scene_jobs[0]
    for prompt in job.prompts:
        assert "hard sci-fi industrial" in prompt


# ── P14: project_id in job_id ────────────────────────────────────────────────

def test_p14_project_id_in_job_id():
    resp = _pipeline(SCENARIO_EN, project_id="bassito_proj_007")
    for job in resp.scene_jobs:
        assert job.job_id.startswith("bassito_proj_007_")
