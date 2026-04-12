"""
bridge/ace_step.py
==================
ACE-Step 1.5 XL music generation pipeline for Andrew Swarm.

ACE-Step 1.5 XL is an open-source music foundation model (4B-parameter DiT
decoder + LM planner) that outperforms commercial alternatives such as Suno v5
on SongEval benchmarks while running fully locally.

Model variants
--------------
  xl_base   — full flexibility across all tasks; best for fine-tuning and
               style-transfer or cover generation.
  xl_sft    — supervised fine-tune with classifier-free guidance; highest audio
               quality for text-to-music and lyric-to-song tasks.
  xl_turbo  — distilled to 8 inference steps; generates a full song in < 10 s
               on an RTX 3090.

Benchmark highlights (SongEval, April 2026)
-------------------------------------------
  ACE-Step 1.5-XL  →  overall 8.12  style 6.62
  Suno v5          →  overall 7.87  style 6.51
  MinMax-2.5       →  overall 8.06  style 6.49

Supported scenario tag syntax (case-insensitive)
-------------------------------------------------
    [MUSIC: ...]   core music description / prompt
    [STYLE: ...]   genre, mood, instrumentation tags
    [LYRICS: ...]  vocal/lyrics text to sing
    [AUDIO: ...]   alias for [MUSIC: ...] (compatible with LTX scenario syntax)

Scene header patterns accepted (same as ltx_video.py)
------------------------------------------------------
    ## Scene 1 — Title (0:00–0:15)
    **Сцена 2** (0:15–0:30)
    Scene 3: Title [0:30–0:45]
    --- Scene 4 ---
    (plain numbered paragraphs are split on double-newlines as a fallback)

References
----------
  GitHub : https://github.com/ace-step/ACE-Step-1.5
  Paper  : https://arxiv.org/abs/2602.00744
  HF Hub : ace-step/ACE-Step-1.5-XL-SFT (etc.)
"""
from __future__ import annotations

import re
import uuid
from typing import Any

from bridge.schemas import (
    AceStepGenerationConfig,
    AceStepJob,
    AceStepMusicRequest,
    AceStepMusicResponse,
    AceStepSegment,
)

# ── regex helpers ─────────────────────────────────────────────────────────────

_TAG_RE = re.compile(
    r"\[(?P<tag>MUSIC|STYLE|LYRICS|AUDIO)\s*:\s*(?P<value>[^\]]+)\]",
    re.IGNORECASE,
)

# Detects whether a line is a scene/segment header.
_SEGMENT_HEADER_RE = re.compile(
    r"(?:#{1,3}|--+|\*{1,2})?\s*"
    r"(?:scene|shot|clip|segment|track|\u0441\u0446\u0435\u043d\u0430|\u0441\u0446\u0435\u043d\u0443|\u043a\u0430\u0434\u0440)\s*\d+",
    re.IGNORECASE,
)

# Extracts the time range from anywhere in a header line.
_TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}:\d{2}(?:\.\d+)?)\s*[-\u2013\u2014]\s*(?P<end>\d{1,2}:\d{2}(?:\.\d+)?)",
)

_TITLE_CLEANUP_RE = re.compile(r"[#\-*\s]+")

# Style tags are comma-separated values inside [STYLE: ...] — split on comma/semicolon.
_STYLE_SPLIT_RE = re.compile(r"[,;]+")


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
    """Pull [MUSIC/STYLE/LYRICS/AUDIO: value] tags from a block of text."""
    tags: dict[str, str] = {}
    for m in _TAG_RE.finditer(text):
        key = m.group("tag").upper()
        # AUDIO is an alias for MUSIC (LTX compatibility)
        if key == "AUDIO" and "MUSIC" not in tags:
            key = "MUSIC"
        tags[key] = m.group("value").strip()
    return tags


def _split_style_tags(raw: str) -> list[str]:
    """Split 'dark ambient, cinematic, piano' → ['dark ambient', 'cinematic', 'piano']."""
    return [t.strip() for t in _STYLE_SPLIT_RE.split(raw) if t.strip()]


def _split_into_blocks(scenario_text: str) -> list[str]:
    """
    Split scenario text into per-segment blocks.

    Prefers splitting on scene/segment header lines; falls back to double-newline
    paragraphs when no headers are found.
    """
    lines = scenario_text.splitlines()
    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        if _SEGMENT_HEADER_RE.search(line):
            if current:
                blocks.append("\n".join(current).strip())
            current = [line]
        else:
            current.append(line)

    if current:
        blocks.append("\n".join(current).strip())

    # fallback: no headers found → split on blank lines
    if not blocks or (
        len(blocks) == 1 and not _SEGMENT_HEADER_RE.search(scenario_text)
    ):
        raw_blocks = [b.strip() for b in re.split(r"\n{2,}", scenario_text) if b.strip()]
        return raw_blocks if raw_blocks else [scenario_text.strip()]

    return [b for b in blocks if b]


# ── parser ────────────────────────────────────────────────────────────────────


class AceStepScenarioParser:
    """Parse a free-form scenario/ТЗ text into a list of AceStepSegment objects."""

    def parse(
        self,
        scenario_text: str,
        config: AceStepGenerationConfig,
    ) -> list[AceStepSegment]:
        blocks = _split_into_blocks(scenario_text)
        segments: list[AceStepSegment] = []
        cursor_sec = 0.0

        for idx, block in enumerate(blocks):
            segment = self._parse_block(block, idx, cursor_sec, config)
            if segment is not None:
                segments.append(segment)
                cursor_sec = segment.end_sec

        return segments

    # ── private ───────────────────────────────────────────────────────────────

    def _parse_block(
        self,
        block: str,
        idx: int,
        cursor_sec: float,
        config: AceStepGenerationConfig,
    ) -> AceStepSegment | None:
        if not block.strip():
            return None

        # ── timing ────────────────────────────────────────────────────────────
        start_sec = cursor_sec
        end_sec = cursor_sec
        first_line = block.splitlines()[0] if block else ""
        time_match = _TIME_RANGE_RE.search(first_line)
        if time_match:
            start_sec = _parse_time_to_sec(time_match.group("start"))
            end_sec = _parse_time_to_sec(time_match.group("end"))

        duration_sec = end_sec - start_sec
        if duration_sec <= 0:
            # default: 30 s per untimed segment (typical song section length)
            duration_sec = min(config.max_segment_duration_sec, 30.0)
            end_sec = start_sec + duration_sec

        duration_sec = min(duration_sec, config.max_segment_duration_sec)

        # ── scene reference (for video pairing) ───────────────────────────────
        scene_ref = ""
        header_match = _SEGMENT_HEADER_RE.search(first_line)
        if header_match:
            scene_ref = _TITLE_CLEANUP_RE.sub(" ", first_line).strip()

        # ── tags ──────────────────────────────────────────────────────────────
        tags = _extract_tags(block)
        music_prompt = tags.get("MUSIC", "")
        style_raw = tags.get("STYLE", "")
        lyrics = tags.get("LYRICS", "")

        # fallback: use first non-header, non-empty, non-tag line as music prompt
        if not music_prompt:
            for line in block.splitlines():
                stripped = line.strip()
                if (
                    stripped
                    and not _SEGMENT_HEADER_RE.search(stripped)
                    and not _TAG_RE.search(stripped)
                ):
                    music_prompt = stripped
                    break

        if not music_prompt:
            music_prompt = f"Segment {idx + 1} background music"

        style_tags = _split_style_tags(style_raw)

        segment_id = f"segment_{idx + 1:02d}"

        return AceStepSegment(
            segment_id=segment_id,
            segment_index=idx,
            scene_ref=scene_ref,
            start_sec=round(start_sec, 2),
            end_sec=round(end_sec, 2),
            duration_sec=round(duration_sec, 2),
            music_prompt=music_prompt,
            style_tags=style_tags,
            lyrics=lyrics,
        )


# ── job builder ───────────────────────────────────────────────────────────────


class AceStepJobBuilder:
    """Convert parsed AceStepSegment objects into ACE-Step-ready job descriptors."""

    def build(
        self,
        project_id: str,
        segments: list[AceStepSegment],
        config: AceStepGenerationConfig,
    ) -> AceStepMusicResponse:
        warnings: list[str] = []
        jobs: list[AceStepJob] = []

        for segment in segments:
            job = self._build_job(project_id, segment, config)
            jobs.append(job)

        estimated_vram = self._estimate_vram(config)
        if estimated_vram > config.vram_budget_gb:
            warnings.append(
                f"Estimated VRAM ({estimated_vram:.1f} GB) exceeds budget "
                f"({config.vram_budget_gb} GB). Consider enabling use_quantisation "
                "or use_offload to reduce peak memory."
            )

        if not segments:
            warnings.append("No music segments were parsed from the scenario text.")

        return AceStepMusicResponse(
            project_id=project_id,
            total_segments=len(jobs),
            music_jobs=jobs,
            queue_ready=len(jobs) > 0,
            estimated_vram_gb=estimated_vram,
            warnings=warnings,
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _build_job(
        self,
        project_id: str,
        segment: AceStepSegment,
        config: AceStepGenerationConfig,
    ) -> AceStepJob:
        short_id = uuid.uuid4().hex[:8]
        job_id = f"{project_id}_{segment.segment_id}_{short_id}"

        # Determine job_type from content
        job_type = "ace_step_music"
        if segment.source_audio:
            job_type = "ace_step_cover"
        elif segment.lyrics:
            job_type = "ace_step_vocals"

        model_config_ace = self._model_config(config, segment)

        return AceStepJob(
            job_id=job_id,
            segment_id=segment.segment_id,
            segment_index=segment.segment_index,
            job_type=job_type,
            duration_sec=segment.duration_sec,
            music_prompt=self._full_prompt(segment),
            style_tags=list(segment.style_tags),
            lyrics=segment.lyrics,
            source_audio=segment.source_audio,
            model_config_ace=model_config_ace,
            status="queued",
        )

    def _full_prompt(self, segment: AceStepSegment) -> str:
        """Combine music_prompt and style_tags into the final generation prompt."""
        if segment.style_tags:
            tag_str = ", ".join(segment.style_tags)
            return f"{segment.music_prompt}, {tag_str}"
        return segment.music_prompt

    def _model_config(
        self,
        config: AceStepGenerationConfig,
        segment: AceStepSegment,
    ) -> dict[str, Any]:
        # xl_turbo is distilled to 8 steps; ignore the user-supplied step count
        steps = 8 if config.model_variant == "xl_turbo" else config.inference_steps

        cfg: dict[str, Any] = {
            "variant": config.model_variant,
            "inference_steps": steps,
            "guidance_scale": config.guidance_scale,
            "audio_format": config.audio_format,
            "sample_rate": config.sample_rate,
            "language": config.language,
            "use_quantisation": config.use_quantisation,
            "use_offload": config.use_offload,
            "vram_budget_gb": config.vram_budget_gb,
        }
        if segment.source_audio:
            cfg["source_audio"] = segment.source_audio
        return cfg

    def _estimate_vram(self, config: AceStepGenerationConfig) -> float:
        """
        Rough VRAM estimate for ACE-Step 1.5 XL (4B DiT decoder, bf16 weights ≈ 9 GB).

        - bf16, no quantisation, no offload → ~9 GB
        - with quantisation (int8/fp8)       → ~5 GB
        - with offload                        → keeps GPU use under 4 GB during
                                               the forward pass at cost of speed
        """
        if config.use_offload:
            return 4.0
        if config.use_quantisation:
            return 5.0
        return 9.0


# ── convenience facade ────────────────────────────────────────────────────────


class AceStepMusicPipeline:
    """Single entry-point: scenario text → AceStepMusicResponse."""

    def __init__(
        self,
        parser: AceStepScenarioParser | None = None,
        builder: AceStepJobBuilder | None = None,
    ):
        self._parser = parser or AceStepScenarioParser()
        self._builder = builder or AceStepJobBuilder()

    def run(self, request: AceStepMusicRequest) -> AceStepMusicResponse:
        segments = self._parser.parse(request.scenario_text, request.config)
        return self._builder.build(request.project_id, segments, request.config)
