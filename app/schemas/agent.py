from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["pending", "running", "waiting_user", "completed", "failed", "cancelled"]


class CreateAgentTaskRequest(BaseModel):
    session_id: str | None = None
    user_input: str = Field(min_length=1, max_length=8000)
    task_type: str = "project_upgrade"
    kb_id: int | None = None
    document_id: int | None = None
    conversation_id: int | None = None
    idempotency_key: str | None = Field(default=None, min_length=8, max_length=128)
    deadline_seconds: int | None = Field(default=None, ge=30, le=3600)
    max_steps: int | None = Field(default=None, ge=2, le=16)
    max_tool_calls: int | None = Field(default=None, ge=2, le=32)
    max_total_tokens: int | None = Field(default=None, ge=1000, le=200000)
    max_cost_usd: float | None = Field(default=None, ge=0.01, le=100.0)


class CreateAgentTaskResponse(BaseModel):
    session_id: str
    task_id: str
    run_id: str
    status: TaskStatus


class ResumeAgentTaskRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    user_input: str | None = Field(default=None, min_length=1, max_length=8000)
    note: str | None = Field(default=None, max_length=1000)


class AgentTaskResponse(BaseModel):
    session_id: str
    task_id: str
    run_id: str
    status: TaskStatus
    user_input: str
    task_type: str
    kb_id: int | None = None
    document_id: int | None = None
    conversation_id: int | None = None
    final_answer: str | None = None
    cancel_requested: bool = False
    created_at: datetime
    updated_at: datetime


class AgentEventResponse(BaseModel):
    session_id: str
    task_id: str
    run_id: str | None = None
    event_type: str
    event_index: int
    agent_name: str | None = None
    skill_name: str | None = None
    tool_name: str | None = None
    step_id: str | None = None
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
