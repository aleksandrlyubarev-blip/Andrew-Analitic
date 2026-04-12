"""
core/tools/ace_step_generate.py
================================
AbstractTool wrapper for the ACE-Step 1.5 XL music generation pipeline.

Orchestration calls this tool with a project_id + scenario_text and receives
back a fully-populated AceStepMusicResponse (job queue + VRAM estimates).

The tool is concurrency-safe and read-only from the orchestrator's perspective:
it builds job descriptors but does not submit them to the ACE-Step runtime
directly.  Submission is a separate side-effecting step owned by the user's
workflow (e.g. a local ACE-Step server, Hugging Face Spaces, or ComfyUI with
the ACE-Step custom node).
"""
from __future__ import annotations

from bridge.ace_step import AceStepMusicPipeline
from bridge.schemas import (
    AceStepGenerationConfig,
    AceStepMusicRequest,
    AceStepMusicResponse,
)
from core.tools.base import AbstractTool, ToolResult, ToolUseContext


class AceStepGenerateTool(AbstractTool[AceStepMusicRequest, AceStepMusicResponse]):
    """
    Parse a scenario / ТЗ text and emit an ACE-Step 1.5 XL music job queue.

    Input:
        project_id     — unique identifier for the project (matches the video
                          project_id when pairing with LTX video jobs).
        scenario_text  — scenario text with [MUSIC/STYLE/LYRICS/AUDIO] tags.
                          [AUDIO: ...] tags from LTX scenarios are automatically
                          promoted to [MUSIC: ...] for cross-pipeline compatibility.
        config         — optional AceStepGenerationConfig.  Defaults:
                          variant=xl_sft, quantisation=on, offload=on (≤4 GB VRAM).

    Output:
        AceStepMusicResponse with one AceStepJob per parsed segment.

    Job types
    ---------
        ace_step_music   — text-to-music (default)
        ace_step_vocals  — text-to-music with embedded lyrics
        ace_step_cover   — style-transfer / cover from a source audio file

    Model variants
    --------------
        xl_base   — most flexible; good for fine-tuning and cover tasks.
        xl_sft    — highest quality text-to-music (default).
        xl_turbo  — 8-step distilled; fastest (< 10 s / song on RTX 3090).
    """

    name = "ace_step_generate"

    def input_schema(self) -> type[AceStepMusicRequest]:
        return AceStepMusicRequest

    async def prompt(self) -> str:
        return (
            "Parse a video/film scenario text into an ACE-Step 1.5 XL music job queue. "
            "Provide project_id and scenario_text containing [MUSIC/STYLE/LYRICS/AUDIO] tags. "
            "Optionally override config: model_variant='xl_turbo' for fastest generation, "
            "'xl_sft' (default) for highest quality, or 'xl_base' for fine-tuning tasks. "
            "Set use_quantisation=true (default) and use_offload=true (default) to run on "
            "consumer GPUs with ≤12 GB VRAM."
        )

    def is_read_only(self, args: AceStepMusicRequest) -> bool:
        return True

    def is_concurrency_safe(self, args: AceStepMusicRequest) -> bool:
        return True

    async def call(
        self, args: AceStepMusicRequest, context: ToolUseContext
    ) -> ToolResult[AceStepMusicResponse]:
        pipeline = AceStepMusicPipeline()
        response = pipeline.run(args)

        summary_lines = [
            f"ACE-Step 1.5 XL music job queue built for project '{args.project_id}'.",
            f"  Segments parsed : {response.total_segments}",
            f"  VRAM estimate   : {response.estimated_vram_gb:.1f} GB",
            f"  Queue ready     : {response.queue_ready}",
            f"  Model variant   : {args.config.model_variant}",
        ]
        if response.warnings:
            summary_lines.append("  Warnings:")
            for w in response.warnings:
                summary_lines.append(f"    • {w}")

        for job in response.music_jobs:
            tag_str = ", ".join(job.style_tags) if job.style_tags else "—"
            lyrics_flag = " [lyrics]" if job.lyrics else ""
            summary_lines.append(
                f"  [{job.segment_id}] type={job.job_type} "
                f"duration={job.duration_sec}s style=[{tag_str}]{lyrics_flag}"
            )

        return ToolResult(
            output=response,
            output_to_model="\n".join(summary_lines),
        )


def make_ace_step_generate_tool() -> AceStepGenerateTool:
    return AceStepGenerateTool()
