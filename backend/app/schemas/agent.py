"""Agent 运行时请求/响应模型。

定义 Agent 任务的生命周期数据契约：
- CreateAgentTask:  创建任务（支持预算、场景、知识库等参数）
- ResumeAgentTask:  暂停后继续（批准/编辑/拒绝）
- AgentTaskResponse: 任务详情响应
- AgentEventResponse: 事件流响应
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal["pending", "running", "waiting_user", "completed", "failed", "cancelled"]


class CreateAgentTaskRequest(BaseModel):
    """创建 Agent 任务请求——定义任务输入、预算和资源范围。"""
    session_id: str | None = None
    user_input: str = Field(min_length=1, max_length=8000)
    task_type: str = "project_upgrade"
    scenario_key: str | None = Field(default=None, max_length=128)
    scenario_inputs: dict[str, Any] = Field(default_factory=dict)
    source_policy: Literal["auto", "local_only"] = "auto"
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
    """创建 Agent 任务响应——返回外部可见的标识信息。"""
    session_id: str
    task_id: str
    run_id: str
    status: TaskStatus


class ResumeAgentTaskRequest(BaseModel):
    """恢复 Agent 任务请求——用户对暂停点做出审批决定。"""
    action: Literal["approve", "edit", "reject"]
    user_input: str | None = Field(default=None, min_length=1, max_length=8000)
    note: str | None = Field(default=None, max_length=1000)


class AgentTaskResponse(BaseModel):
    """Agent 任务详情响应——包含完整的生命周期状态和输出。"""
    session_id: str
    task_id: str
    run_id: str
    status: TaskStatus
    user_input: str
    task_type: str
    scenario_key: str | None = None
    scenario: dict[str, Any] = Field(default_factory=dict)
    kb_id: int | None = None
    document_id: int | None = None
    conversation_id: int | None = None
    final_answer: str | None = None
    cancel_requested: bool = False
    created_at: datetime
    updated_at: datetime


class AgentEventResponse(BaseModel):
    """Agent 事件响应——单条事件记录，用于前端事件流展示。"""
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
