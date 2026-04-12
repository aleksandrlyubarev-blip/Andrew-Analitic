"""
tests/test_ace_step.py
======================
Tests for the ACE-Step 1.5 XL music generation pipeline.

Coverage:
  M1.  Scenario with explicit segment headers + all four tags → correct fields.
  M2.  Cyrillic segment headers (Сцена 1).
  M3.  [AUDIO: ...] tag (LTX alias) is promoted to music_prompt.
  M4.  [LYRICS: ...] tag sets lyrics and flips job_type to ace_step_vocals.
  M5.  source_audio on segment flips job_type to ace_step_cover.
  M6.  Style tags are split on comma and stored as a list.
  M7.  xl_turbo variant forces inference_steps=8 in model_config_ace.
  M8.  xl_sft variant keeps the user-supplied inference_steps.
  M9.  VRAM warning fires when estimate exceeds budget.
  M10. Empty scenario → zero jobs + warning.
  M11. No timing in header → duration defaults to 30 s (capped at max).
  M12. Missing MUSIC/AUDIO tag → first prose line used as music_prompt.
  M13. project_id is prefixed in job_id.
  M14. Full prompt includes style tags appended to music_prompt.
  M15. use_offload=True → estimated_vram_gb ≤ 4.0.
  M16. use_quantisation=True, no offload → estimated_vram_gb ≤ 5.0.
  M17. AceStepGenerateTool.run() returns ToolResult with output and summary.
  M18. Segment timing parses correctly from 'M:SS–M:SS' range.
  M19. Multi-segment scenario produces one job per segment in correct order.
  M20. Language config propagates into model_config_ace.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bridge.ace_step import AceStepJobBuilder, AceStepScenarioParser, AceStepMusicPipeline
from bridge.schemas import AceStepGenerationConfig, AceStepMusicRequest, AceStepSegment
from core.tools.ace_step_generate import AceStepGenerateTool
from core.tools.base import ToolUseContext

# ── shared fixtures ───────────────────────────────────────────────────────────

SCENARIO_EN = """\
## Scene 1 — Opening Credits (0:00–0:30)
[MUSIC: sweeping orchestral intro with brass and strings]
[STYLE: cinematic, epic, orchestral]

## Scene 2 — Chase Sequence (0:30–1:15)
[MUSIC: high-energy electronic chase theme]
[STYLE: electronic, intense, percussive]
[LYRICS: Run, don't look back, the night is alive]
"""

SCENARIO_RU = """\
Сцена 1 (0:00-0:20)
[MUSIC: атмосферная электронная музыка]
[STYLE: ambient, sci-fi]

Сцена 2 (0:20-0:50)
[MUSIC: нарастающий оркестровый саундтрек]
[STYLE: cinematic, orchestral]
"""

SCENARIO_LTX_AUDIO = """\
## Scene 1 (0:00–0:15)
[VISUAL: man walking through neon corridor]
[STYLE: cinematic neon-tech]
[AUDIO: dark ambient synth pads with distant city hum]
"""

DEFAULT_CONFIG = AceStepGenerationConfig()


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse(text: str, config: AceStepGenerationConfig | None = None) -> list:
    return AceStepScenarioParser().parse(text, config or DEFAULT_CONFIG)


def _pipeline(
    text: str,
    project_id: str = "test_proj",
    config: AceStepGenerationConfig | None = None,
) -> object:
    req = AceStepMusicRequest(
        project_id=project_id,
        scenario_text=text,
        config=config or DEFAULT_CONFIG,
    )
    return AceStepMusicPipeline().run(req)


# ── M1: happy path (EN scenario) ─────────────────────────────────────────────

def test_m1_en_scenario_basic_fields():
    segments = _parse(SCENARIO_EN)
    assert len(segments) == 2

    s1 = segments[0]
    assert s1.segment_id == "segment_01"
    assert "orchestral" in s1.music_prompt.lower()
    assert "cinematic" in s1.style_tags
    assert "epic" in s1.style_tags
    assert s1.duration_sec == pytest.approx(30.0, abs=0.1)

    s2 = segments[1]
    assert "electronic" in s2.music_prompt.lower()
    assert s2.lyrics == "Run, don't look back, the night is alive"
    assert s2.duration_sec == pytest.approx(45.0, abs=0.1)


# ── M2: Cyrillic headers ──────────────────────────────────────────────────────

def test_m2_cyrillic_headers():
    segments = _parse(SCENARIO_RU)
    assert len(segments) == 2
    assert "ambient" in segments[0].style_tags or "sci-fi" in segments[0].style_tags
    assert segments[1].duration_sec == pytest.approx(30.0, abs=0.1)


# ── M3: [AUDIO] alias promoted to music_prompt ───────────────────────────────

def test_m3_audio_alias_promoted():
    segments = _parse(SCENARIO_LTX_AUDIO)
    assert len(segments) == 1
    s = segments[0]
    assert "ambient" in s.music_prompt.lower() or "synth" in s.music_prompt.lower()
    assert s.music_prompt  # must not be empty


# ── M4: [LYRICS] sets job_type to ace_step_vocals ────────────────────────────

def test_m4_lyrics_sets_job_type_vocals():
    resp = _pipeline(SCENARIO_EN)
    job1, job2 = resp.music_jobs
    assert job1.job_type == "ace_step_music"
    assert job2.job_type == "ace_step_vocals"
    assert job2.lyrics == "Run, don't look back, the night is alive"


# ── M5: source_audio sets job_type to ace_step_cover ─────────────────────────

def test_m5_source_audio_sets_job_type_cover():
    segment = AceStepSegment(
        segment_id="segment_01",
        segment_index=0,
        duration_sec=30.0,
        music_prompt="cover of original theme",
        source_audio="path/to/original.wav",
    )
    builder = AceStepJobBuilder()
    job = builder._build_job("proj", segment, DEFAULT_CONFIG)
    assert job.job_type == "ace_step_cover"
    assert job.model_config_ace["source_audio"] == "path/to/original.wav"


# ── M6: style tags are split on comma ────────────────────────────────────────

def test_m6_style_tags_split():
    scenario = """\
## Scene 1 (0:00–0:20)
[MUSIC: tense underscore]
[STYLE: dark ambient, slow burn, minimalist piano]
"""
    segments = _parse(scenario)
    tags = segments[0].style_tags
    assert "dark ambient" in tags
    assert "slow burn" in tags
    assert "minimalist piano" in tags
    assert len(tags) == 3


# ── M7: xl_turbo forces 8 inference steps ────────────────────────────────────

def test_m7_xl_turbo_forces_8_steps():
    config = AceStepGenerationConfig(model_variant="xl_turbo", inference_steps=50)
    resp = _pipeline(SCENARIO_EN, config=config)
    for job in resp.music_jobs:
        assert job.model_config_ace["inference_steps"] == 8


# ── M8: xl_sft keeps user-supplied steps ─────────────────────────────────────

def test_m8_xl_sft_keeps_user_steps():
    config = AceStepGenerationConfig(model_variant="xl_sft", inference_steps=40)
    resp = _pipeline(SCENARIO_EN, config=config)
    for job in resp.music_jobs:
        assert job.model_config_ace["inference_steps"] == 40


# ── M9: VRAM warning ─────────────────────────────────────────────────────────

def test_m9_vram_warning():
    # bf16, no quantisation, no offload → ~9 GB > 6 GB budget
    heavy_config = AceStepGenerationConfig(
        use_quantisation=False,
        use_offload=False,
        vram_budget_gb=6,
    )
    resp = _pipeline(SCENARIO_EN, config=heavy_config)
    assert any("VRAM" in w for w in resp.warnings)


# ── M10: empty scenario ───────────────────────────────────────────────────────

def test_m10_empty_scenario():
    resp = _pipeline("   \n\n  ")
    assert resp.total_segments == 0
    assert not resp.queue_ready
    assert resp.warnings


# ── M11: no timing → 30 s default ────────────────────────────────────────────

def test_m11_no_timing_default_duration():
    scenario = """\
## Scene 1
[MUSIC: ambient drone]
"""
    segments = _parse(scenario)
    assert segments[0].duration_sec == pytest.approx(30.0, abs=0.1)


# ── M12: missing MUSIC/AUDIO tag → first prose line as prompt ────────────────

def test_m12_no_music_tag_fallback():
    scenario = """\
## Scene 1 (0:00–0:15)
A haunting melody echoes through empty halls.
"""
    segments = _parse(scenario)
    assert "haunting" in segments[0].music_prompt.lower()


# ── M13: project_id prefixed in job_id ───────────────────────────────────────

def test_m13_project_id_in_job_id():
    resp = _pipeline(SCENARIO_EN, project_id="bassito_007")
    for job in resp.music_jobs:
        assert job.job_id.startswith("bassito_007_")


# ── M14: full prompt includes style tags ─────────────────────────────────────

def test_m14_full_prompt_includes_style():
    scenario = """\
## Scene 1 (0:00–0:20)
[MUSIC: driving bass line]
[STYLE: industrial, metal, distorted]
"""
    resp = _pipeline(scenario)
    assert len(resp.music_jobs) == 1
    prompt = resp.music_jobs[0].music_prompt
    assert "industrial" in prompt
    assert "metal" in prompt
    assert "driving bass line" in prompt


# ── M15: use_offload → VRAM ≤ 4 GB ──────────────────────────────────────────

def test_m15_offload_vram_estimate():
    config = AceStepGenerationConfig(use_offload=True)
    resp = _pipeline(SCENARIO_EN, config=config)
    assert resp.estimated_vram_gb <= 4.0


# ── M16: quantisation without offload → VRAM ≤ 5 GB ─────────────────────────

def test_m16_quantisation_vram_estimate():
    config = AceStepGenerationConfig(use_quantisation=True, use_offload=False)
    resp = _pipeline(SCENARIO_EN, config=config)
    assert resp.estimated_vram_gb <= 5.0


# ── M17: tool run() returns ToolResult ───────────────────────────────────────

@pytest.mark.asyncio
async def test_m17_tool_run():
    tool = AceStepGenerateTool()
    ctx = ToolUseContext()
    req = AceStepMusicRequest(
        project_id="tool_test",
        scenario_text=SCENARIO_EN,
    )
    result = await tool.run(req, ctx)

    assert result.success
    assert result.output is not None
    assert result.output.total_segments == 2
    assert result.output_to_model is not None
    assert "tool_test" in result.output_to_model
    assert "xl_sft" in result.output_to_model


# ── M18: timing parses correctly ─────────────────────────────────────────────

def test_m18_timing_parse():
    scenario = """\
## Track 1 (1:30–2:45)
[MUSIC: bridge section with key change]
"""
    segments = _parse(scenario)
    assert segments[0].start_sec == pytest.approx(90.0, abs=0.1)
    assert segments[0].end_sec == pytest.approx(165.0, abs=0.1)
    assert segments[0].duration_sec == pytest.approx(75.0, abs=0.1)


# ── M19: multi-segment order ──────────────────────────────────────────────────

def test_m19_multi_segment_order():
    resp = _pipeline(SCENARIO_EN)
    assert resp.total_segments == 2
    assert resp.music_jobs[0].segment_index == 0
    assert resp.music_jobs[1].segment_index == 1
    assert resp.music_jobs[0].segment_id == "segment_01"
    assert resp.music_jobs[1].segment_id == "segment_02"


# ── M20: language propagates into model_config_ace ───────────────────────────

def test_m20_language_propagates():
    config = AceStepGenerationConfig(language="ru")
    resp = _pipeline(SCENARIO_RU, config=config)
    for job in resp.music_jobs:
        assert job.model_config_ace["language"] == "ru"
