"""
bridge/schemas.py
=================
Pydantic request and response models for the FastAPI bridge endpoints.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


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
