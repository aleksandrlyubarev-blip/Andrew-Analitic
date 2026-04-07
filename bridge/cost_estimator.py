"""
bridge/cost_estimator.py
========================
Estimate cost and generation time for a scenario before dispatching.

Pricing (rough estimates, updated 2025-Q1):
  Google Veo 3.1   ~$0.030 / sec of output video
  xAI Grok 4.2     ~$0.025 / sec of output video
  Higgsfield DOP   ~$0.020 / sec of output video (DOP standard tier)
  Local ComfyUI    $0.00   (electricity only)

Generation time ratios (seconds of wall-clock time per second of video):
  Veo 3.1        ≈  60 s/s  (cloud queue + generation)
  Grok 4.2       ≈  50 s/s  (cloud queue + generation)
  Higgsfield DOP ≈  45 s/s  (cloud queue + generation)
  ComfyUI        ≈ 120 s/s  (local 24 GB GPU, fp8 LTX 2.3)
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from bridge.higgsfield_client import MODEL_MAP as _HF_MODEL_MAP
from bridge.ltx_video import LtxVideoPipeline
from bridge.schemas import LtxGenerationConfig, LtxSceneJob, LtxVideoJobRequest

# ── Pricing table ─────────────────────────────────────────────────────────────

_PRICE_PER_SEC: dict[str, float] = {
    "veo":        0.030,
    "grok":       0.025,
    "higgsfield": 0.020,
    "comfyui":    0.000,
}

# Wall-clock seconds needed to generate 1 second of output video
_TIME_FACTOR: dict[str, float] = {
    "veo":        60.0,
    "grok":       50.0,
    "higgsfield": 45.0,
    "comfyui":    120.0,
}

_VEO_MODELS         = frozenset({"veo3.1", "veo-3.1", "veo3", "veo-3"})
_GROK_MODELS        = frozenset({"grok4.2", "grok-4.2", "grok4", "grok-4"})
_HIGGSFIELD_MODELS  = frozenset(_HF_MODEL_MAP.keys())
_HIGGSFIELD_SLUG_PREFIXES = (
    "higgsfield-ai/", "kling-video/", "bytedance/", "wan/", "alibaba-cloud/",
)


def _backend_key(preferred_model: str) -> Literal["veo", "grok", "higgsfield", "comfyui"]:
    m = preferred_model.strip().lower()
    if m in _VEO_MODELS:
        return "veo"
    if m in _GROK_MODELS:
        return "grok"
    if m in _HIGGSFIELD_MODELS or any(m.startswith(p) for p in _HIGGSFIELD_SLUG_PREFIXES):
        return "higgsfield"
    return "comfyui"


# ── Output models ─────────────────────────────────────────────────────────────

class SceneCostEstimate(BaseModel):
    scene_id: str
    backend: str
    duration_sec: float
    cost_usd: float
    gen_time_sec: float


class BatchCostEstimate(BaseModel):
    scenes: list[SceneCostEstimate]
    total_cost_usd: float
    sequential_time_sec: float  # if run one-by-one
    parallel_time_sec: float    # with max_concurrent concurrent slots
    by_backend: dict[str, float]  # cost breakdown per backend key


# ── Estimator ─────────────────────────────────────────────────────────────────

class CostEstimator:
    """
    Estimate cost and wall-clock time for a list of LtxSceneJobs.

    Parameters
    ----------
    max_concurrent:
        How many jobs run in parallel (mirrors VideoDispatcher.max_concurrent).
    fallback_to_comfyui:
        If True, assume failed cloud scenes fall back to ComfyUI and add
        ComfyUI cost as worst-case overhead.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        fallback_to_comfyui: bool = False,
    ) -> None:
        self._max_concurrent = max_concurrent
        self._fallback = fallback_to_comfyui

    def estimate_jobs(
        self,
        jobs: list[LtxSceneJob],
    ) -> BatchCostEstimate:
        scenes: list[SceneCostEstimate] = []
        by_backend: dict[str, float] = {
            "veo": 0.0, "grok": 0.0, "higgsfield": 0.0, "comfyui": 0.0,
        }

        for job in jobs:
            bk = _backend_key(job.preferred_model)
            cost = job.duration_sec * _PRICE_PER_SEC[bk]
            gen_t = job.duration_sec * _TIME_FACTOR[bk]

            if self._fallback and bk != "comfyui":
                # Worst case: primary fails → ComfyUI fallback adds its cost too
                cost += job.duration_sec * _PRICE_PER_SEC["comfyui"]
                gen_t += job.duration_sec * _TIME_FACTOR["comfyui"]

            scenes.append(SceneCostEstimate(
                scene_id=job.scene_id,
                backend=bk,
                duration_sec=job.duration_sec,
                cost_usd=round(cost, 4),
                gen_time_sec=round(gen_t, 1),
            ))
            by_backend[bk] = round(by_backend[bk] + cost, 4)

        total_cost = round(sum(s.cost_usd for s in scenes), 4)
        seq_time   = round(sum(s.gen_time_sec for s in scenes), 1)

        # Parallel time: pack scenes into concurrent slots, use the slowest slot
        slots = [0.0] * self._max_concurrent
        for s in sorted(scenes, key=lambda x: -x.gen_time_sec):
            slots[slots.index(min(slots))] += s.gen_time_sec
        par_time = round(max(slots), 1)

        return BatchCostEstimate(
            scenes=scenes,
            total_cost_usd=total_cost,
            sequential_time_sec=seq_time,
            parallel_time_sec=par_time,
            by_backend={k: v for k, v in by_backend.items() if v > 0},
        )

    def estimate_scenario(
        self,
        scenario_text: str,
        project_id: str = "estimate",
        config: LtxGenerationConfig | None = None,
    ) -> BatchCostEstimate:
        """Parse a scenario string and return a cost estimate."""
        pipeline = LtxVideoPipeline()
        job_response = pipeline.run(LtxVideoJobRequest(
            project_id=project_id,
            scenario_text=scenario_text,
            config=config or LtxGenerationConfig(),
        ))
        return self.estimate_jobs(job_response.scene_jobs)

    def format_report(self, est: BatchCostEstimate) -> str:
        """Human-readable cost report for CLI output."""
        lines = ["", "  Scene estimate:", "  " + "─" * 54]
        for s in est.scenes:
            backend_label = {
                "veo":        "Veo 3.1   ",
                "grok":       "Grok 4.2  ",
                "higgsfield": "Higgsfield",
                "comfyui":    "ComfyUI   ",
            }[s.backend]
            cost_str = f"${s.cost_usd:.3f}" if s.cost_usd > 0 else "free"
            lines.append(
                f"  {s.scene_id:<12} {s.duration_sec:>4.0f}s  "
                f"→ {backend_label}  {cost_str:>8}  "
                f"~{s.gen_time_sec/60:.1f} min"
            )
        lines.append("  " + "─" * 54)
        lines.append(f"  Total cost  : ${est.total_cost_usd:.3f}")
        lines.append(
            f"  Parallel est: ~{est.parallel_time_sec/60:.1f} min "
            f"({self._max_concurrent} concurrent)"
        )
        if est.by_backend:
            bk_str = "  ,  ".join(
                f"{k}: ${v:.3f}" for k, v in est.by_backend.items()
            )
            lines.append(f"  By backend  : {bk_str}")
        lines.append("")
        return "\n".join(lines)
