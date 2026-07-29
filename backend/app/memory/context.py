"""Agent 上下文组装器——为 Planner/Executor/Verifier 构建统一上下文包。

汇聚用户画像、会话摘要、近期事件、任务状态和记忆事件五大数据源。
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from app.models.agent_runtime import AgentEvent, AgentRun, AgentTask, MemoryEvent, SessionSummary

from .profile import load_user_profile
from .short_term import load_recent_events


def _json_load(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _event_to_dict(row: AgentEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "event_index": row.event_index,
        "event_type": row.event_type,
        "agent_name": row.agent_name,
        "skill_name": row.skill_name,
        "tool_name": row.tool_name,
        "step_id": row.step_id,
        "message": row.message,
        "payload": _json_load(row.payload_json, {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _memory_event_to_dict(row: MemoryEvent) -> dict[str, Any]:
    return {
        "id": row.id,
        "event_type": row.event_type,
        "category": row.category,
        "content": row.content,
        "source": row.source,
        "metadata": _json_load(row.metadata_json, {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _latest_summary(db: Session, user_id: int | None, session_id: str | None) -> dict[str, Any] | None:
    if user_id is None or not session_id:
        return None
    row = (
        db.query(SessionSummary)
        .filter(SessionSummary.user_id == int(user_id), SessionSummary.session_id == session_id)
        .order_by(SessionSummary.id.desc())
        .first()
    )
    if not row:
        return None
    return {
        "id": row.id,
        "summary": row.summary,
        "summary_until_event_id": row.summary_until_event_id,
        "metadata": _json_load(row.metadata_json, {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def build_agent_context(db: Session, state: dict[str, Any]) -> dict[str, Any]:
    """Build the context package consumed by Planner/Executor/Verifier/RAG."""
    user_id = state.get("user_id")
    session_id = state.get("session_id")
    task_id = state.get("task_id")
    run_id = state.get("run_id")

    task = db.query(AgentTask).filter(AgentTask.task_id == task_id).first() if task_id else None
    run = db.query(AgentRun).filter(AgentRun.run_id == run_id).first() if run_id else None

    redis_events = load_recent_events(user_id, session_id, limit=20)
    if not redis_events and session_id:
        rows = (
            db.query(AgentEvent)
            .filter(AgentEvent.session_id == session_id)
            .order_by(AgentEvent.id.desc())
            .limit(20)
            .all()
        )
        redis_events = [_event_to_dict(row) for row in reversed(rows)]

    memory_rows = []
    if user_id is not None:
        query = db.query(MemoryEvent).filter(MemoryEvent.user_id == int(user_id))
        if task_id:
            query = query.filter((MemoryEvent.task_id == task_id) | (MemoryEvent.session_id == session_id))
        memory_rows = query.order_by(MemoryEvent.id.desc()).limit(20).all()

    profile = load_user_profile(db, int(user_id) if user_id is not None else None)
    return {
        "profile": profile,
        "profile_item_count": len(profile.get("items", [])),
        "session": {
            "session_id": session_id,
            "summary": _latest_summary(db, int(user_id) if user_id is not None else None, session_id),
            "recent_events": redis_events,
        },
        "task": {
            "task_id": task_id,
            "run_id": run_id,
            "goal": state.get("goal") or (task.goal if task else None),
            "intent": state.get("intent") or (task.intent if task else None),
            "status": state.get("status") or (task.status if task else None),
            "state": _json_load(task.state_json, {}) if task else {},
            "checkpoint": _json_load(run.checkpoint_json, {}) if run else {},
            "current_step_id": run.current_step_id if run else None,
        },
        "memory_events": [_memory_event_to_dict(row) for row in reversed(memory_rows)],
    }
