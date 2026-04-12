"""
bridge/schemas.py
=================
Pydantic request and response models for the FastAPI bridge endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str
    channel: str = "api"
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class AnalyzeResponse(BaseModel):
    query: str
    narrative: str
    sql_query: Optional[str] = None
    confidence: float
    cost_usd: float
    success: bool
    error: Optional[str] = None
    elapsed_seconds: float
    routing: str
    formatted_message: str


class ScheduleRequest(BaseModel):
    query: str
    cron_schedule: str  # e.g., "0 9 * * 1" for Monday 9am
    name: Optional[str] = None


class HealthResponse(BaseModel):
    andrew: str
    moltis: Dict[str, Any]
    bridge: str


class SceneClipScore(BaseModel):
    visual_quality: int
    continuity_fit: int
    prompt_match: int
    motion_stability: int
    timeline_usefulness: int
    notes: List[str] = Field(default_factory=list)
    risk_flags: List[str] = Field(default_factory=list)
    recommended_action: str = "keep"


class SceneReviewRequest(BaseModel):
    project_id: str
    scene_id: str
    scene_goal: str
    style_profile: str = ""
    editing_mode: str = "hybrid"
    editing_template: str = "cinematic_montage"
    target_duration_sec: float
    actual_duration_sec: float = 0.0
    used_clips: List[str] = Field(default_factory=list)
    rejected_clips: List[str] = Field(default_factory=list)
    clip_scores: Dict[str, SceneClipScore] = Field(default_factory=dict)
    reviews: Dict[str, str] = Field(default_factory=dict)
    bridge_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    regeneration_jobs: List[Dict[str, Any]] = Field(default_factory=list)
    timeline: Optional[Dict[str, Any]] = None


class SceneReviewResponse(BaseModel):
    project_id: str
    scene_id: str
    success: bool
    confidence: float
    summary: str
    warnings: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    quality_breakdown: Dict[str, float] = Field(default_factory=dict)
    formatted_message: str
    hitl_decision: str = "skipped"
    hitl_review_id: Optional[str] = None


class SceneOpsBassitoJobInput(BaseModel):
    job_id: Optional[str] = None
    job_type: str
    status: str = "queued"
    source_clip_id: Optional[str] = None
    artifact_path: Optional[str] = None
    request_path: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SceneOpsAggregateRequest(SceneReviewRequest):
    bassito_jobs: List[SceneOpsBassitoJobInput] = Field(default_factory=list)
    updated_at: Optional[str] = None


class SceneOpsClipScore(BaseModel):
    visualQuality: int
    continuityFit: int
    promptMatch: int
    motionStability: int
    timelineUsefulness: int
    recommendedAction: str = "keep"


class SceneOpsSceneBundle(BaseModel):
    sceneId: str
    sceneGoal: str
    editingTemplate: str
    targetDurationSec: float
    actualDurationSec: float
    usedClips: List[str] = Field(default_factory=list)
    rejectedClips: List[str] = Field(default_factory=list)
    queueState: str


class SceneOpsAndrewReview(BaseModel):
    confidence: float
    summary: str
    warnings: List[str] = Field(default_factory=list)
    recommendedActions: List[str] = Field(default_factory=list)
    qualityBreakdown: Dict[str, float] = Field(default_factory=dict)
    hitlDecision: str = "skipped"


class SceneOpsBassitoJob(BaseModel):
    jobId: str
    jobType: str
    status: str
    sourceClipId: Optional[str] = None
    artifactPath: Optional[str] = None


class SceneOpsSnapshotResponse(BaseModel):
    scene: SceneOpsSceneBundle
    clipScores: Dict[str, SceneOpsClipScore] = Field(default_factory=dict)
    andrew: SceneOpsAndrewReview
    bassitoJobs: List[SceneOpsBassitoJob] = Field(default_factory=list)
    updatedAt: str
    source: str = "api"


# ── LTX 2.3 video generation ──────────────────────────────────────────────────

class LtxKeyframe(BaseModel):
    """Single anchor keyframe for LTX multi-keyframe or first/last workflow."""

    index: int  # 0 = first, -1 = last, >0 = ordered middle keyframe
    frame_type: str  # "first" | "last" | "middle"
    source_prompt: str
    image_path: Optional[str] = None  # path to a pre-rendered still (Flux/SD3/MJ)


class LtxScene(BaseModel):
    """One parsed scene from a ТЗ/scenario."""

    scene_id: str
    scene_index: int
    title: str = ""
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_sec: float = 0.0
    visual_prompt: str
    style: str = ""
    audio_description: str = ""
    keyframes: List[LtxKeyframe] = Field(default_factory=list)
    workflow: str = "first_last"  # "first_last" | "multi_keyframe"


class LtxGenerationConfig(BaseModel):
    """Hardware/quality config for an LTX 2.3 generation run."""

    resolution: str = "1920x1080"
    aspect_ratio: str = "16:9"  # "16:9" | "9:16"
    model_variant: str = "fp8"  # "bf16" | "fp8"
    use_distilled_lora: bool = True
    vram_budget_gb: int = 12
    double_upscale: bool = True
    audio_native: bool = True
    max_clip_duration_sec: float = 15.0
    multi_keyframe_threshold_sec: float = 10.0  # scenes longer than this get multi_keyframe


class LtxVideoJobRequest(BaseModel):
    """Input for the LTX scenario-to-jobs pipeline."""

    project_id: str
    scenario_text: str
    config: LtxGenerationConfig = Field(default_factory=LtxGenerationConfig)


class LtxSceneJob(BaseModel):
    """ComfyUI-ready job descriptor for a single LTX scene clip."""

    job_id: str
    scene_id: str
    scene_index: int
    job_type: str = "ltx_keyframe"
    workflow: str  # "first_last" | "multi_keyframe"
    duration_sec: float
    prompts: List[str]
    keyframes: List[LtxKeyframe]
    model_config_ltx: Dict[str, Any] = Field(default_factory=dict)
    audio: Dict[str, Any] = Field(default_factory=dict)
    status: str = "queued"


class LtxVideoJobResponse(BaseModel):
    """Full pipeline output: parsed scenes + ComfyUI job queue."""

    project_id: str
    total_scenes: int
    scene_jobs: List[LtxSceneJob] = Field(default_factory=list)
    comfyui_queue_ready: bool = True
    estimated_vram_gb: float = 0.0
    warnings: List[str] = Field(default_factory=list)


# ── ACE-Step 1.5 XL music generation ─────────────────────────────────────────

class AceStepGenerationConfig(BaseModel):
    """Hardware/quality config for an ACE-Step 1.5 XL generation run.

    Three model variants:
      xl_base   — full flexibility, suitable for fine-tuning and cover/style tasks.
      xl_sft    — supervised fine-tune, highest audio quality via CFG.
      xl_turbo  — distilled to 8 steps, fastest generation.

    VRAM guidance (bf16 weights, no quantisation):
      xl_base / xl_sft  → ~9 GB (requires ≥12 GB with offload; ≥20 GB without).
      xl_turbo          → same weight size but fewer steps ⇒ faster, not smaller.
    Quantisation (int8/fp8) halves effective weight size.
    """

    model_variant: str = "xl_sft"          # "xl_base" | "xl_sft" | "xl_turbo"
    use_quantisation: bool = True           # int8/fp8 quantisation for low-VRAM
    use_offload: bool = True               # CPU offload to fit in ≤12 GB
    vram_budget_gb: int = 12
    inference_steps: int = 25              # xl_turbo overrides this to 8
    guidance_scale: float = 7.0            # CFG scale (xl_sft benefits most)
    audio_format: str = "wav"              # "wav" | "mp3" | "flac"
    sample_rate: int = 44100
    max_segment_duration_sec: float = 240.0  # ACE-Step supports up to 10-min songs
    language: str = "en"                   # primary lyrics/prompt language


class AceStepSegment(BaseModel):
    """One parsed music segment derived from a scenario block."""

    segment_id: str
    segment_index: int
    scene_ref: str = ""          # optional reference to paired video scene_id
    start_sec: float = 0.0
    end_sec: float = 0.0
    duration_sec: float = 0.0
    music_prompt: str            # core text description of the music
    style_tags: List[str] = Field(default_factory=list)   # genre / mood tags
    lyrics: str = ""             # optional vocals/lyrics text
    source_audio: str = ""       # path/URL for style-transfer or cover source


class AceStepJob(BaseModel):
    """ACE-Step-ready job descriptor for a single music segment."""

    job_id: str
    segment_id: str
    segment_index: int
    job_type: str = "ace_step_music"  # "ace_step_music" | "ace_step_cover" | "ace_step_vocals"
    duration_sec: float
    music_prompt: str
    style_tags: List[str] = Field(default_factory=list)
    lyrics: str = ""
    source_audio: str = ""
    model_config_ace: Dict[str, Any] = Field(default_factory=dict)
    status: str = "queued"


class AceStepMusicRequest(BaseModel):
    """Input for the ACE-Step scenario-to-jobs pipeline."""

    project_id: str
    scenario_text: str
    config: AceStepGenerationConfig = Field(default_factory=AceStepGenerationConfig)


class AceStepMusicResponse(BaseModel):
    """Full pipeline output: parsed segments + ACE-Step job queue."""

    project_id: str
    total_segments: int
    music_jobs: List[AceStepJob] = Field(default_factory=list)
    queue_ready: bool = True
    estimated_vram_gb: float = 0.0
    warnings: List[str] = Field(default_factory=list)
