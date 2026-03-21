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
