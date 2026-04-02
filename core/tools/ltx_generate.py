"""
core/tools/ltx_generate.py
==========================
AbstractTool wrapper for the LTX 2.3 video generation pipeline.

Orchestration calls this tool with a project_id + scenario_text and receives
back a fully-populated LtxVideoJobResponse (ComfyUI job queue + VRAM estimates).

The tool is concurrency-safe and read-only from the orchestrator's perspective:
it builds job descriptors but does not submit them to ComfyUI directly.
Submission is a separate side-effecting step owned by the user's workflow.
"""
from __future__ import annotations

from bridge.ltx_video import LtxVideoPipeline
from bridge.schemas import (
    LtxGenerationConfig,
    LtxVideoJobRequest,
    LtxVideoJobResponse,
)
from core.tools.base import AbstractTool, ToolResult, ToolUseContext


class LtxGenerateTool(AbstractTool[LtxVideoJobRequest, LtxVideoJobResponse]):
    """
    Parse a Bassito-style ТЗ/scenario and emit a ComfyUI-ready LTX job queue.

    Input:
        project_id     — unique identifier for the video project.
        scenario_text  — full scenario text with [VISUAL/STYLE/AUDIO] tags.
        config         — optional LtxGenerationConfig (defaults: fp8, 12 GB VRAM,
                         double_upscale, 9:16 when aspect_ratio is set).

    Output:
        LtxVideoJobResponse with one LtxSceneJob per parsed scene.
    """

    name = "ltx_generate"

    def input_schema(self) -> type[LtxVideoJobRequest]:
        return LtxVideoJobRequest

    async def prompt(self) -> str:
        return (
            "Parse a video scenario / ТЗ text into a ComfyUI LTX 2.3 job queue. "
            "Provide project_id and scenario_text. "
            "Optionally override config fields such as aspect_ratio='9:16' for Shorts/Reels, "
            "model_variant='fp8' for low-VRAM hardware, or double_upscale=true to save VRAM."
        )

    def is_read_only(self, args: LtxVideoJobRequest) -> bool:
        return True

    def is_concurrency_safe(self, args: LtxVideoJobRequest) -> bool:
        return True

    async def call(
        self, args: LtxVideoJobRequest, context: ToolUseContext
    ) -> ToolResult[LtxVideoJobResponse]:
        pipeline = LtxVideoPipeline()
        response = pipeline.run(args)

        summary_lines = [
            f"LTX job queue built for project '{args.project_id}'.",
            f"  Scenes parsed : {response.total_scenes}",
            f"  VRAM estimate : {response.estimated_vram_gb:.1f} GB",
            f"  Queue ready   : {response.comfyui_queue_ready}",
        ]
        if response.warnings:
            summary_lines.append("  Warnings:")
            for w in response.warnings:
                summary_lines.append(f"    • {w}")

        for job in response.scene_jobs:
            summary_lines.append(
                f"  [{job.scene_id}] workflow={job.workflow} "
                f"duration={job.duration_sec}s keyframes={len(job.keyframes)}"
            )

        return ToolResult(
            output=response,
            output_to_model="\n".join(summary_lines),
        )


def make_ltx_generate_tool() -> LtxGenerateTool:
    return LtxGenerateTool()
