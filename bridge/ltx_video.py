"""
bridge/ltx_video.py
===================
LTX 2.3 video generation pipeline for Andrew Swarm.

Two-stage pipeline:
  1. LtxScenarioParser  — converts a ТЗ/scenario text into structured LtxScene objects.
  2. LtxJobBuilder      — converts scenes into ComfyUI-ready LtxSceneJob descriptors.

Supported scenario tag syntax (case-insensitive):
    [VISUAL: ...]   visual action / scene description
    [STYLE:  ...]   visual style (cinematic, neon-tech, etc.)
    [AUDIO:  ...]   audio: voiceover text + sfx description

Scene header patterns accepted:
    ## Scene 1 — Title (0:00–0:15)
    **Сцена 2** (0:15-0:30)
    Scene 3: Title [0:30 - 0:45]
    --- Scene 4 ---
    (plain numbered paragraphs are split on double-newlines as a fallback)
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from bridge.schemas import (
    LtxGenerationConfig,
    LtxKeyframe,
    LtxScene,
    LtxSceneJob,
    LtxVideoJobRequest,
    LtxVideoJobResponse,
)

# ── regex helpers ─────────────────────────────────────────────────────────────

_TAG_RE = re.compile(
    r"\[(?P<tag>VISUAL|STYLE|AUDIO)\s*:\s*(?P<value>[^\]]+)\]",
    re.IGNORECASE,
)

# Detects whether a line is a scene header (scene/shot/Сцена + number).
_SCENE_HEADER_RE = re.compile(
    r"(?:#{1,3}|--+|\*{1,2})?\s*"
    r"(?:scene|shot|clip|\u0441\u0446\u0435\u043d\u0430|\u0441\u0446\u0435\u043d\u0443|\u043a\u0430\u0434\u0440)\s*\d+",
    re.IGNORECASE,
)

# Extracts the time range from anywhere in a header line.
_TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?:\.\d+)?)\s*[-\u2013\u2014]\s*(?P<end>\d{1,2}:\d{2}(?:\.\d+)?)",
)

_TITLE_CLEANUP_RE = re.compile(r"[#\-*\s]+")


def _parse_time_to_sec(t: str) -> float:
    """Convert 'M:SS' or 'H:MM:SS' to seconds."""
    parts = t.strip().split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    except ValueError:
        pass
    return 0.0


def _extract_tags(text: str) -> dict[str, str]:
    """Pull [VISUAL/STYLE/AUDIO: value] tags from a block of text."""
    tags: dict[str, str] = {}
    for m in _TAG_RE.finditer(text):
        key = m.group("tag").upper()
        tags[key] = m.group("value").strip()
    return tags


def _split_into_blocks(scenario_text: str) -> list[str]:
    """
    Split scenario text into per-scene blocks.

    Prefers splitting on scene header lines; falls back to double-newline
    paragraphs when no headers are found.
    """
    lines = scenario_text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        if _SCENE_HEADER_RE.search(line):
            if current:
                blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    # fallback: no headers found → split on blank lines
    if not blocks or (len(blocks) == 1 and not _SCENE_HEADER_RE.search(scenario_text)):
        raw_blocks = [b.strip() for b in re.split(r"\n{2,}", scenario_text) if b.strip()]
        return raw_blocks if raw_blocks else [scenario_text.strip()]

    return [b for b in blocks if b]


# ── parser ────────────────────────────────────────────────────────────────────


class LtxScenarioParser:
    """Parse a free-form ТЗ/scenario text into a list of LtxScene objects."""

    def parse(self, scenario_text: str, config: LtxGenerationConfig) -> list[LtxScene]:
        blocks = _split_into_blocks(scenario_text)
        scenes: list[LtxScene] = []
        cursor_sec = 0.0

        for idx, block in enumerate(blocks):
            scene = self._parse_block(block, idx, cursor_sec, config)
            if scene is not None:
                scenes.append(scene)
                cursor_sec = scene.end_sec

        return scenes

    # ── private ───────────────────────────────────────────────────────────────

    def _parse_block(
        self,
        block: str,
        idx: int,
        cursor_sec: float,
        config: LtxGenerationConfig,
    ) -> LtxScene | None:
        if not block.strip():
            return None

        header_match = _SCENE_HEADER_RE.search(block)

        # timing — scan the first line of the block for a time range
        start_sec = cursor_sec
        end_sec = cursor_sec
        first_line = block.splitlines()[0] if block else ""
        time_match = _TIME_RANGE_RE.search(first_line)
        if time_match:
            start_sec = _parse_time_to_sec(time_match.group("start"))
            end_sec = _parse_time_to_sec(time_match.group("end"))

        # if no timing parsed, infer from max_clip_duration_sec
        duration_sec = end_sec - start_sec
        if duration_sec <= 0:
            duration_sec = min(config.max_clip_duration_sec, 10.0)
            end_sec = start_sec + duration_sec

        duration_sec = min(duration_sec, config.max_clip_duration_sec)

        # title
        title = ""
        if header_match:
            raw_title = first_line.strip()
            title = _TITLE_CLEANUP_RE.sub(" ", raw_title).strip()

        # tags
        tags = _extract_tags(block)
        visual_prompt = tags.get("VISUAL", "")
        style = tags.get("STYLE", "")
        audio_description = tags.get("AUDIO", "")

        # fallback: use first non-header, non-empty line as visual prompt
        if not visual_prompt:
            for line in block.splitlines():
                stripped = line.strip()
                if stripped and not _SCENE_HEADER_RE.search(stripped) and not _TAG_RE.search(stripped):
                    visual_prompt = stripped
                    break

        if not visual_prompt:
            visual_prompt = f"Scene {idx + 1}"

        if style:
            full_prompt = f"{visual_prompt}, {style}"
        else:
            full_prompt = visual_prompt

        scene_id = f"scene_{idx + 1:02d}"
        workflow = (
            "multi_keyframe"
            if duration_sec > config.multi_keyframe_threshold_sec
            else "first_last"
        )

        keyframes = self._build_keyframes(full_prompt, duration_sec, workflow)

        return LtxScene(
            scene_id=scene_id,
            scene_index=idx,
            title=title,
            start_sec=start_sec,
            end_sec=end_sec,
            duration_sec=round(duration_sec, 2),
            visual_prompt=visual_prompt,
            style=style,
            audio_description=audio_description,
            keyframes=keyframes,
            workflow=workflow,
        )

    def _build_keyframes(
        self,
        full_prompt: str,
        duration_sec: float,
        workflow: str,
    ) -> list[LtxKeyframe]:
        """
        Build anchor keyframes from the scene prompt.

        first_last   → 2 keyframes (index 0 and -1).
        multi_keyframe → 3-5 keyframes depending on duration.
        """
        if workflow == "first_last":
            return [
                LtxKeyframe(index=0, frame_type="first", source_prompt=full_prompt),
                LtxKeyframe(index=-1, frame_type="last", source_prompt=full_prompt),
            ]

        # multi_keyframe: distribute middle keyframes linearly
        n_middle = min(3, max(1, int(duration_sec // 5) - 1))
        kfs: list[LtxKeyframe] = [
            LtxKeyframe(index=0, frame_type="first", source_prompt=full_prompt)
        ]
        for i in range(1, n_middle + 1):
            kfs.append(
                LtxKeyframe(
                    index=i,
                    frame_type="middle",
                    source_prompt=full_prompt,
                )
            )
        kfs.append(LtxKeyframe(index=-1, frame_type="last", source_prompt=full_prompt))
        return kfs


# ── job builder ───────────────────────────────────────────────────────────────


class LtxJobBuilder:
    """Convert parsed LtxScene objects into ComfyUI-ready LtxSceneJob descriptors."""

    def build(
        self,
        project_id: str,
        scenes: list[LtxScene],
        config: LtxGenerationConfig,
    ) -> LtxVideoJobResponse:
        warnings: list[str] = []
        jobs: list[LtxSceneJob] = []

        for scene in scenes:
            job = self._build_job(project_id, scene, config)
            jobs.append(job)

        estimated_vram = self._estimate_vram(config)
        if estimated_vram > config.vram_budget_gb:
            warnings.append(
                f"Estimated VRAM ({estimated_vram:.1f} GB) exceeds budget "
                f"({config.vram_budget_gb} GB). Consider enabling double_upscale "
                "or switching to fp8 variant."
            )

        if not scenes:
            warnings.append("No scenes were parsed from the scenario text.")

        return LtxVideoJobResponse(
            project_id=project_id,
            total_scenes=len(jobs),
            scene_jobs=jobs,
            comfyui_queue_ready=len(jobs) > 0,
            estimated_vram_gb=estimated_vram,
            warnings=warnings,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _build_job(
        self,
        project_id: str,
        scene: LtxScene,
        config: LtxGenerationConfig,
    ) -> LtxSceneJob:
        short_id = uuid.uuid4().hex[:8]
        job_id = f"{project_id}_{scene.scene_id}_{short_id}"

        prompts = self._build_prompts(scene)
        model_config_ltx = self._model_config(config)
        audio = self._audio_config(scene, config)

        return LtxSceneJob(
            job_id=job_id,
            scene_id=scene.scene_id,
            scene_index=scene.scene_index,
            job_type="ltx_keyframe",
            workflow=scene.workflow,
            duration_sec=scene.duration_sec,
            prompts=prompts,
            keyframes=scene.keyframes,
            model_config_ltx=model_config_ltx,
            audio=audio,
            status="queued",
        )

    def _build_prompts(self, scene: LtxScene) -> list[str]:
        base = scene.visual_prompt
        if scene.style:
            styled = f"{base}, {scene.style}"
        else:
            styled = base

        if scene.workflow == "first_last":
            return [styled]

        # multi_keyframe: one prompt per inter-keyframe segment
        n_segments = max(1, len(scene.keyframes) - 1)
        return [styled] * n_segments

    def _model_config(self, config: LtxGenerationConfig) -> dict[str, Any]:
        cfg: dict[str, Any] = {
            "variant": config.model_variant,
            "resolution": config.resolution,
            "aspect_ratio": config.aspect_ratio,
            "vram_budget_gb": config.vram_budget_gb,
        }
        if config.use_distilled_lora:
            cfg["lora"] = "distilled"
        if config.double_upscale:
            cfg["upscale_pass"] = "double"
            cfg["generation_resolution"] = "960x544"
        return cfg

    def _audio_config(self, scene: LtxScene, config: LtxGenerationConfig) -> dict[str, Any]:
        return {
            "include_native": config.audio_native,
            "description": scene.audio_description,
        }

    def _estimate_vram(self, config: LtxGenerationConfig) -> float:
        """
        Rough VRAM estimate for LTX 2.3 (22B DiT) based on variant + resolution.

        fp8 + double_upscale fits comfortably on 12 GB.
        bf16 full-res needs ~20 GB.
        """
        base = 8.0 if config.model_variant == "fp8" else 18.0
        if not config.double_upscale and config.resolution == "1920x1080":
            base += 4.0
        return base


# ── convenience facade ────────────────────────────────────────────────────────


class LtxVideoPipeline:
    """Single entry-point: scenario text → LtxVideoJobResponse."""

    def __init__(
        self,
        parser: LtxScenarioParser | None = None,
        builder: LtxJobBuilder | None = None,
    ):
        self._parser = parser or LtxScenarioParser()
        self._builder = builder or LtxJobBuilder()

    def run(self, request: LtxVideoJobRequest) -> LtxVideoJobResponse:
        scenes = self._parser.parse(request.scenario_text, request.config)
        return self._builder.build(request.project_id, scenes, request.config)
