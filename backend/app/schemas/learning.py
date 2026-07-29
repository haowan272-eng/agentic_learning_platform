"""学习领域 Pydantic 请求/响应模型。

对应 models/learning.py 中的 ORM 模型，提供 API 层数据校验和序列化。
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class LearningProfileUpsert(BaseModel):
    target_role: str | None = Field(default=None, max_length=128)
    goal: str | None = Field(default=None, max_length=4000)
    current_level: str = Field(default="unknown", max_length=64)
    weekly_minutes: int = Field(default=300, ge=30, le=10080)
    preferences: dict[str, Any] = Field(default_factory=dict)


class LearningProfileResponse(BaseModel):
    id: int
    target_role: str | None = None
    goal: str | None = None
    current_level: str
    weekly_minutes: int
    preferences: dict[str, Any] = Field(default_factory=dict)
    diagnostic_summary: str | None = None
    readiness_score: float
    updated_at: datetime | None = None


class LearningWeaknessResponse(BaseModel):
    id: int
    topic: str
    category: str
    severity: float
    confidence: float
    evidence: dict[str, Any] = Field(default_factory=dict)
    status: str
    updated_at: datetime | None = None


class LearningPracticeResponse(BaseModel):
    id: int
    task_id: str | None = None
    kb_id: int | None = None
    topic: str
    question: str
    expected_answer: str | None = None
    difficulty: str
    source: dict[str, Any] = Field(default_factory=dict)
    status: str
    created_at: datetime | None = None


class LearningAssessmentCreate(BaseModel):
    practice_id: int | None = None
    task_id: str | None = Field(default=None, max_length=64)
    answer: str | None = Field(default=None, max_length=8000)
    feedback: str = Field(min_length=1, max_length=8000)
    score: float = Field(ge=0.0, le=1.0)
    rubric: dict[str, Any] = Field(default_factory=dict)


class LearningAssessmentResponse(BaseModel):
    id: int
    practice_id: int | None = None
    task_id: str | None = None
    feedback: str
    score: float
    rubric: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class LearningReviewItemResponse(BaseModel):
    id: int
    weakness_id: int | None = None
    topic: str
    prompt: str
    due_at: datetime
    interval_days: int
    status: str


class LearningLongTermAssetUpsert(BaseModel):
    asset_type: str = Field(max_length=64)
    asset_key: str | None = Field(default=None, max_length=160)
    title: str = Field(min_length=1, max_length=255)
    summary: str | None = Field(default=None, max_length=4000)
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float = Field(default=0.6, ge=0.0, le=1.0)
    weight: float = Field(default=1.0, ge=0.0, le=100.0)
    status: str = Field(default="active", max_length=32)
    trend_value: float | None = None


class LearningLongTermAssetResponse(BaseModel):
    id: int
    asset_type: str
    asset_key: str
    title: str
    summary: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence: float
    weight: float
    source: str
    status: str
    task_id: str | None = None
    kb_id: int | None = None
    document_id: int | None = None
    conversation_id: int | None = None
    version: int
    trend_value: float | None = None
    observed_at: datetime | None = None
    updated_at: datetime | None = None


class LearningLongTermAssetSnapshotResponse(BaseModel):
    target_role: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    tech_stack_profile: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    project_experience: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    recurring_error: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    weakness_map: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    interview_answer_version: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    mock_interview_record: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    review_plan: list[LearningLongTermAssetResponse] = Field(default_factory=list)
    capability_trend: list[LearningLongTermAssetResponse] = Field(default_factory=list)


class LearningDashboardResponse(BaseModel):
    active_days_14d: int
    tasks_completed_14d: int
    practice_accuracy: float
    open_weaknesses: int
    weakness_trend: list[dict[str, Any]]
    material_hit_rate: float
    agent_saved_minutes: int
    due_reviews: int
    recent_practices: list[LearningPracticeResponse]
    top_weaknesses: list[LearningWeaknessResponse]
