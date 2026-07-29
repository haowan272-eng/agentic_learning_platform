"""Agent 运行时 ORM 模型。

涵盖 Agent 任务的完整生命周期：
- AgentSession / AgentTask / AgentRun: 会话、任务和执行实例
- AgentPlan / AgentStep: 计划与步骤
- AgentEvent / AgentToolCall / AgentVerification: 事件、工具调用和验证
- AgentTool / AgentToolPermission: 工具注册与权限
- MemoryEvent / SessionSummary: 记忆事件与会话摘要
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from app.core.database import Base


class AgentTask(Base):
    __tablename__ = "agent_tasks"
    __table_args__ = (UniqueConstraint("task_id", name="uq_agent_tasks_task_id"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    task_id = Column(String(64), nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)

    user_id = Column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    username = Column(String(128), nullable=False, index=True)
    parent_task_id = Column(String(64), ForeignKey("agent_tasks.task_id", ondelete="SET NULL"), nullable=True, index=True)

    task_type = Column(String(64), nullable=False, default="project_upgrade", index=True)
    goal = Column(Text, nullable=True)
    intent = Column(String(128), nullable=True, index=True)
    status = Column(String(32), nullable=False, default="pending", index=True)
    user_input = Column(Text, nullable=False)
    final_answer = Column(Text, nullable=True)

    kb_id = Column(Integer, ForeignKey("knowledge_bases.id"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=True, index=True)
    conversation_id = Column(Integer, ForeignKey("rag_conversations.id"), nullable=True, index=True)

    state_json = Column(Text, nullable=True)
    kb_scope_json = Column(Text, nullable=True)
    budget_json = Column(Text, nullable=True)
    idempotency_key = Column(String(128), nullable=True, unique=True, index=True)
    cancel_requested = Column(Integer, nullable=False, default=0, index=True)
    lease_owner = Column(String(128), nullable=True, index=True)
    lease_until = Column(DateTime, nullable=True, index=True)
    last_error = Column(Text, nullable=True)
    resume_payload_json = Column(Text, nullable=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime, nullable=True)


class AgentSession(Base):
    __tablename__ = "agent_sessions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    title = Column(String(255), nullable=False, default="New Session")
    active_task_id = Column(String(64), nullable=True, index=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    status = Column(String(32), nullable=False, default="running", index=True)
    current_node = Column(String(128), nullable=True)
    current_step_id = Column(String(128), nullable=True)
    checkpoint_json = Column(Text, nullable=True)
    started_at = Column(DateTime, server_default=func.now())
    finished_at = Column(DateTime, nullable=True)


class AgentEvent(Base):
    __tablename__ = "agent_events"
    __table_args__ = (UniqueConstraint("task_id", "event_index", name="uq_agent_event_index"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    task_id = Column(String(64), nullable=False, index=True)
    run_id = Column(String(64), nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    event_index = Column(Integer, nullable=False)

    agent_name = Column(String(64), nullable=True)
    skill_name = Column(String(64), nullable=True)
    tool_name = Column(String(128), nullable=True)
    step_id = Column(String(64), nullable=True)
    message = Column(Text, nullable=False)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class AgentPlan(Base):
    __tablename__ = "agent_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    plan_version = Column(Integer, nullable=False, default=1, index=True)
    source = Column(String(32), nullable=False, default="fallback", index=True)
    status = Column(String(32), nullable=False, default="active", index=True)
    goal = Column(Text, nullable=False)
    intent = Column(String(128), nullable=False, index=True)
    plan_json = Column(Text, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class AgentStep(Base):
    __tablename__ = "agent_steps"
    __table_args__ = (UniqueConstraint("run_id", "step_id", name="uq_agent_step_run_step"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    plan_id = Column(Integer, ForeignKey("agent_plans.id", ondelete="SET NULL"), nullable=True, index=True)
    task_id = Column(String(64), ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    run_id = Column(String(64), nullable=False, index=True)
    step_id = Column(String(64), nullable=False, index=True)
    step_index = Column(Integer, nullable=False, default=0, index=True)

    agent_name = Column(String(64), nullable=True, index=True)
    skill_name = Column(String(128), nullable=True, index=True)
    tool_name = Column(String(128), nullable=True, index=True)
    objective = Column(Text, nullable=True)

    status = Column(String(32), nullable=False, default="pending", index=True)
    input_json = Column(Text, nullable=True)
    output_json = Column(Text, nullable=True)
    error_json = Column(Text, nullable=True)
    error_type = Column(String(128), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentToolCall(Base):
    __tablename__ = "tool_calls"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), nullable=False, index=True)
    run_id = Column(String(64), nullable=True, index=True)
    step_id = Column(String(64), nullable=True, index=True)
    agent_name = Column(String(64), nullable=True, index=True)
    skill_name = Column(String(128), nullable=True, index=True)
    tool_name = Column(String(128), nullable=False, index=True)
    idempotency_key = Column(String(128), nullable=True, unique=True, index=True)
    status = Column(String(32), nullable=False, default="completed", index=True)
    input_json = Column(Text, nullable=True)
    output_json = Column(Text, nullable=True)
    result_json = Column(Text, nullable=True)
    ok = Column(Integer, nullable=False, default=1, index=True)
    error_type = Column(String(128), nullable=True, index=True)
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    latency_ms = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class AgentVerification(Base):
    __tablename__ = "verifications"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), nullable=False, index=True)
    run_id = Column(String(64), nullable=True, index=True)
    step_id = Column(String(64), nullable=True, index=True)
    target_type = Column(String(64), nullable=True, index=True)
    target_id = Column(String(128), nullable=True, index=True)
    status = Column(String(32), nullable=False, index=True)
    score = Column(Float, nullable=True)
    evidence_json = Column(Text, nullable=True)
    issues_json = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class AgentTool(Base):
    __tablename__ = "tools"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, unique=True, index=True)
    description = Column(Text, nullable=True)
    category = Column(String(64), nullable=False, index=True)
    input_schema_json = Column(Text, nullable=True)
    output_schema_json = Column(Text, nullable=True)
    enabled = Column(Integer, nullable=False, default=1, index=True)
    timeout_ms = Column(Integer, nullable=False, default=10000)
    max_retries = Column(Integer, nullable=False, default=1)
    fallback_tool = Column(String(128), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class AgentToolPermission(Base):
    __tablename__ = "tool_permissions"
    __table_args__ = (UniqueConstraint("tool_name", "agent_name", name="uq_tool_permission_agent"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    tool_name = Column(String(128), ForeignKey("tools.name", ondelete="CASCADE"), nullable=False, index=True)
    agent_name = Column(String(128), nullable=False, index=True)
    allowed = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, server_default=func.now())


class MemoryEvent(Base):
    __tablename__ = "memory_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    session_id = Column(String(64), nullable=True, index=True)
    task_id = Column(String(64), nullable=True, index=True)
    event_type = Column(String(128), nullable=False, index=True)
    category = Column(String(64), nullable=True, index=True)
    content = Column(Text, nullable=False)
    source = Column(String(64), nullable=False, default="agent", index=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class SessionSummary(Base):
    __tablename__ = "session_summaries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    summary = Column(Text, nullable=False)
    summary_until_event_id = Column(Integer, nullable=True)
    metadata_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class AgentOutbox(Base):
    __tablename__ = "agent_outbox"
    __table_args__ = (UniqueConstraint("task_id", "event_type", name="uq_agent_outbox_task_event"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, default="agent_task_requested", index=True)
    payload_json = Column(Text, nullable=False)
    status = Column(String(32), nullable=False, default="pending", index=True)
    attempts = Column(Integer, nullable=False, default=0)
    available_at = Column(DateTime, server_default=func.now(), index=True)
    dispatched_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
