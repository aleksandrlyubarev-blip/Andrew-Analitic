"""
bridge/schemas.py
=================
Pydantic request and response models for the FastAPI bridge endpoints.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

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
    preferred_model: str = ""   # e.g. "sora2", "wan2.6", "veo3.1", "ltx2.3"
    camera_motion: str = ""     # e.g. "slow zoom", "whip-pan", "tracking"
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
    preferred_model: str = ""   # routing hint for multi-backend dispatchers
    camera_motion: str = ""
    status: str = "queued"


class LtxVideoJobResponse(BaseModel):
    """Full pipeline output: parsed scenes + ComfyUI job queue."""

    project_id: str
    total_scenes: int
    scene_jobs: List[LtxSceneJob] = Field(default_factory=list)
    comfyui_queue_ready: bool = True
    estimated_vram_gb: float = 0.0
    warnings: List[str] = Field(default_factory=list)


# ── ComfyUI submission + polling ──────────────────────────────────────────────

class ComfyUIJobStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    PENDING = "pending"


class ComfyUIOutputFile(BaseModel):
    """A single output file produced by a ComfyUI job."""

    node_id: str
    filename: str
    subfolder: str = ""
    file_type: str = "output"
    url: str = ""  # /view?filename=...&type=output


class ComfyUISubmitResult(BaseModel):
    """Result of submitting one workflow and awaiting its completion."""

    prompt_id: str
    scene_id: str = ""
    status: ComfyUIJobStatus = ComfyUIJobStatus.PENDING
    output_files: List[ComfyUIOutputFile] = Field(default_factory=list)
    error: Optional[str] = None


class ComfyUIBatchResult(BaseModel):
    """Aggregate result of a full batch submission (one entry per scene)."""

    total: int
    succeeded: int
    failed: int
    results: List[ComfyUISubmitResult] = Field(default_factory=list)


class ComfyUISubmitRequest(BaseModel):
    """API request to submit a parsed scenario to local ComfyUI."""

    project_id: str
    scenario_text: str
    config: LtxGenerationConfig = Field(default_factory=LtxGenerationConfig)
    comfyui_host: str = "http://127.0.0.1:8188"
    timeout_sec: float = 600.0


# ── Higgsfield cloud generation ───────────────────────────────────────────────

class HiggsfieldGenerationRequest(BaseModel):
    """
    Per-scene generation request for the Higgsfield API.

    Built by VideoDispatcher from LtxSceneJob fields.
    model must be one of: sora2, wan2.6, veo3.1 (or their dash-separated aliases).
    """

    scene_id: str = ""
    model: str                         # "sora2" | "wan2.6" | "veo3.1"
    prompt: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    duration_sec: int = 8              # Higgsfield accepts integer seconds
    fps: int = 24
    resolution: str = "1920x1080"
    aspect_ratio: str = "16:9"
    guidance_scale: float = 7.5
    seed: Optional[int] = None
    camera_motion: str = ""
    # Start+End frame mode (WAN 2.6 / Veo)
    start_frame_url: Optional[str] = None
    end_frame_url: Optional[str] = None
    # Multi-Image Reference (Veo 3.1)
    reference_image_urls: List[str] = Field(default_factory=list)


class GrokVideoRequest(BaseModel):
    """Per-scene generation request for xAI Grok 4.2 video API."""

    scene_id: str = ""
    prompt: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    duration_sec: int = 8
    fps: int = 24
    resolution: str = "1920x1080"
    aspect_ratio: str = "16:9"
    seed: Optional[int] = None
    camera_motion: str = ""
    start_frame_url: Optional[str] = None
    end_frame_url: Optional[str] = None


class VeoVideoRequest(BaseModel):
    """Per-scene generation request for Google Veo 3.1 API (AI Studio / Vertex)."""

    scene_id: str = ""
    prompt: str
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted"
    duration_sec: int = 8
    fps: int = 24
    resolution: str = "1920x1080"
    aspect_ratio: str = "16:9"
    seed: Optional[int] = None
    camera_motion: str = ""
    start_frame_url: Optional[str] = None    # first frame image (GCS URI or HTTPS)
    end_frame_url: Optional[str] = None      # last frame image
    reference_image_urls: List[str] = Field(default_factory=list)


class VideoDispatchRequest(BaseModel):
    """
    API body for POST /video/dispatch.

    Routes each scene by [MODEL: ...] tag:
      veo3.1              → Google Veo 3.1 (direct AI Studio API)
      grok4.2 / grok-4.2 → xAI Grok 4.2
      ltx2.3 / ""         → local ComfyUI
    """

    project_id: str
    scenario_text: str
    config: LtxGenerationConfig = Field(default_factory=LtxGenerationConfig)
    google_api_key: Optional[str] = None     # falls back to GOOGLE_API_KEY
    veo_timeout_sec: float = 900.0
    xai_api_key: Optional[str] = None        # falls back to XAI_API_KEY
    grok_timeout_sec: float = 900.0
    comfyui_host: str = "http://127.0.0.1:8188"
    comfyui_timeout_sec: float = 600.0
    fallback_to_comfyui: bool = True         # retry cloud failures on local ComfyUI
