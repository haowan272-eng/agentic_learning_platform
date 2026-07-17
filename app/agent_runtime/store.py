from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import func
from sqlalchemy.exc import IntegrityError

from app.core.database import SessionLocal
from app.core.config import (
    AGENT_TASK_DEFAULT_DEADLINE_SECONDS,
    AGENT_TASK_DEFAULT_MAX_COST_USD,
    AGENT_TASK_DEFAULT_MAX_STEPS,
    AGENT_TASK_DEFAULT_MAX_TOKENS,
    AGENT_TASK_DEFAULT_MAX_TOOL_CALLS,
)
from app.models.agent_runtime import AgentEvent as AgentEventModel
from app.models.agent_runtime import AgentOutbox, AgentPlan, AgentRun, AgentSession, AgentStep, AgentTask, AgentToolCall, AgentVerification

from .schemas import AgentEvent


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return str(value)


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_load(raw: str | None, default: Any) -> Any:
    if not raw:
        return deepcopy(default)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return deepcopy(default)


def _dt(value: datetime | None) -> datetime:
    return value or utcnow()


def _task_to_dict(task: AgentTask) -> dict[str, Any]:
    return {
        "session_id": task.session_id,
        "task_id": task.task_id,
        "run_id": task.run_id,
        "status": task.status,
        "username": task.username,
        "user_id": task.user_id,
        "parent_task_id": task.parent_task_id,
        "user_input": task.user_input,
        "task_type": task.task_type,
        "goal": task.goal,
        "intent": task.intent,
        "kb_id": task.kb_id,
        "document_id": task.document_id,
        "conversation_id": task.conversation_id,
        "state": _json_load(task.state_json, {}),
        "kb_scope": _json_load(task.kb_scope_json, {}),
        "version": task.version,
        "final_answer": task.final_answer,
        "created_at": _dt(task.created_at),
        "updated_at": _dt(task.updated_at),
        "completed_at": task.completed_at,
        "cancel_requested": bool(task.cancel_requested),
        "budget": _json_load(task.budget_json, {}),
        "resume_payload": _json_load(task.resume_payload_json, None),
    }


def _event_to_dict(event: AgentEventModel) -> dict[str, Any]:
    return {
        "session_id": event.session_id,
        "task_id": event.task_id,
        "run_id": event.run_id,
        "event_type": event.event_type,
        "event_index": event.event_index,
        "agent_name": event.agent_name,
        "skill_name": event.skill_name,
        "tool_name": event.tool_name,
        "step_id": event.step_id,
        "message": event.message,
        "payload": _json_load(event.payload_json, {}),
        "created_at": _dt(event.created_at),
    }


class AgentRuntimeStore:
    """Database-backed runtime store for Agent task orchestration.

    The public methods intentionally mirror the old in-memory store, so the
    LangGraph-style runtime and FastAPI layer stay decoupled from persistence.
    """

    def create_task(self, payload: dict[str, Any]) -> dict[str, Any]:
        now = utcnow()
        budget = {
            "deadline_seconds": int(payload.get("deadline_seconds") or AGENT_TASK_DEFAULT_DEADLINE_SECONDS),
            "max_steps": int(payload.get("max_steps") or AGENT_TASK_DEFAULT_MAX_STEPS),
            "max_tool_calls": int(payload.get("max_tool_calls") or AGENT_TASK_DEFAULT_MAX_TOOL_CALLS),
            "max_total_tokens": int(payload.get("max_total_tokens") or AGENT_TASK_DEFAULT_MAX_TOKENS),
            "max_cost_usd": float(payload.get("max_cost_usd") or AGENT_TASK_DEFAULT_MAX_COST_USD),
            "started_at": None,
        }
        idempotency_key = payload.get("idempotency_key")
        if idempotency_key:
            with SessionLocal() as db:
                existing = db.query(AgentTask).filter(AgentTask.idempotency_key == idempotency_key).first()
                if existing:
                    if existing.user_id != payload.get("user_id"):
                        raise PermissionError("idempotency key belongs to another user")
                    return _task_to_dict(existing)
        task = AgentTask(
            session_id=payload.get("session_id") or str(uuid4()),
            task_id=str(uuid4()),
            run_id=str(uuid4()),
            status="pending",
            username=payload.get("username") or "admin",
            user_id=payload.get("user_id"),
            parent_task_id=payload.get("parent_task_id"),
            user_input=payload["user_input"],
            task_type=payload.get("task_type", "project_upgrade"),
            goal=payload.get("goal") or payload.get("user_input"),
            intent=payload.get("intent"),
            kb_id=payload.get("kb_id"),
            document_id=payload.get("document_id"),
            conversation_id=payload.get("conversation_id"),
            state_json=_json_dump({}),
            kb_scope_json=_json_dump(payload.get("kb_scope") or {}),
            budget_json=_json_dump(budget),
            idempotency_key=idempotency_key,
            cancel_requested=0,
            version=1,
            final_answer=None,
            created_at=now,
            updated_at=now,
        )
        with SessionLocal() as db:
            db.add(task)
            session = db.query(AgentSession).filter(AgentSession.session_id == task.session_id).first()
            if session is None:
                session = AgentSession(
                    session_id=task.session_id,
                    user_id=task.user_id,
                    title=(task.user_input[:80] or "New Session"),
                    active_task_id=task.task_id,
                    metadata_json=_json_dump({"created_by": "agent_runtime"}),
                    created_at=now,
                    updated_at=now,
                )
                db.add(session)
            else:
                if session.user_id != task.user_id:
                    raise PermissionError("session_id belongs to another user")
                session.active_task_id = task.task_id
                session.updated_at = now

            db.add(
                AgentRun(
                    task_id=task.task_id,
                    run_id=task.run_id,
                    user_id=task.user_id,
                    status="running",
                    checkpoint_json=_json_dump({}),
                    started_at=now,
                )
            )
            db.add(
                AgentOutbox(
                    task_id=task.task_id,
                    event_type="agent_task_requested",
                    payload_json=_json_dump({"task_id": task.task_id}),
                    status="pending",
                    attempts=0,
                    available_at=now,
                )
            )
            db.commit()
            db.refresh(task)
            return _task_to_dict(task)

    def get_task(self, task_id: str, user_id: int | None = None) -> dict[str, Any] | None:
        with SessionLocal() as db:
            query = db.query(AgentTask).filter(AgentTask.task_id == task_id)
            if user_id is not None:
                query = query.filter(AgentTask.user_id == user_id)
            task = query.first()
            return _task_to_dict(task) if task else None

    def list_tasks(self, session_id: str | None = None, user_id: int | None = None) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            query = db.query(AgentTask)
            if user_id is not None:
                query = query.filter(AgentTask.user_id == user_id)
            if session_id:
                query = query.filter(AgentTask.session_id == session_id)
            rows = query.order_by(AgentTask.created_at.desc(), AgentTask.id.desc()).all()
            return [_task_to_dict(row) for row in rows]

    def save_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        with SessionLocal() as db:
            task = db.query(AgentTask).filter(AgentTask.task_id == task_id).first()
            if not task:
                return
            task.state_json = _json_dump(state)
            task.status = state.get("status", task.status)
            task.final_answer = state.get("final_answer")
            if "budget" in state:
                task.budget_json = _json_dump(state.get("budget") or {})
            task.intent = state.get("intent", task.intent)
            task.goal = state.get("goal", task.goal)
            if "kb_scope" in state:
                task.kb_scope_json = _json_dump(state.get("kb_scope") or {})
            task.version = int(task.version or 1) + 1
            task.updated_at = utcnow()
            if task.status in {"completed", "failed", "cancelled"} and task.completed_at is None:
                task.completed_at = utcnow()
            run = db.query(AgentRun).filter(AgentRun.run_id == task.run_id).first()
            if run:
                run.status = task.status
                run.current_step_id = (state.get("current_step") or {}).get("step_id")
                run.current_node = state.get("graph_route") or state.get("next_action")
                run.checkpoint_json = _json_dump(state)
                if task.status in {"completed", "failed", "cancelled"} and run.finished_at is None:
                    run.finished_at = utcnow()
            db.commit()

    def request_cancel(self, task_id: str, user_id: int) -> bool:
        with SessionLocal() as db:
            task = db.query(AgentTask).filter(AgentTask.task_id == task_id, AgentTask.user_id == user_id).first()
            if not task or task.status in {"completed", "failed", "cancelled"}:
                return False
            task.cancel_requested = 1
            task.updated_at = utcnow()
            db.commit()
            return True

    def request_resume(self, task_id: str, user_id: int, payload: dict[str, Any]) -> dict[str, Any] | None:
        """Queue a durable HITL response for the existing LangGraph thread."""
        with SessionLocal() as db:
            task = (
                db.query(AgentTask)
                .filter(AgentTask.task_id == task_id, AgentTask.user_id == user_id)
                .with_for_update()
                .first()
            )
            if not task or task.status != "waiting_user":
                return None
            task.resume_payload_json = _json_dump(payload)
            task.status = "pending"
            task.updated_at = utcnow()
            db.add(
                AgentOutbox(
                    task_id=task.task_id,
                    # Each human turn is independently dispatchable; a task
                    # can legitimately pause more than once.
                    event_type=f"agent_task_resumed:{uuid4().hex[:12]}",
                    payload_json=_json_dump({"task_id": task.task_id}),
                    status="pending",
                    attempts=0,
                    available_at=utcnow(),
                )
            )
            db.commit()
            db.refresh(task)
            return _task_to_dict(task)

    def clear_resume_payload(self, task_id: str) -> None:
        with SessionLocal() as db:
            task = db.query(AgentTask).filter(AgentTask.task_id == task_id).first()
            if task:
                task.resume_payload_json = None
                db.commit()

    def is_cancel_requested(self, task_id: str) -> bool:
        with SessionLocal() as db:
            value = db.query(AgentTask.cancel_requested).filter(AgentTask.task_id == task_id).scalar()
            return bool(value)

    def acquire_task_lease(self, task_id: str, owner: str, ttl_seconds: int) -> dict[str, Any] | None:
        now = utcnow()
        until = now + timedelta(seconds=max(30, int(ttl_seconds)))
        with SessionLocal() as db:
            task = db.query(AgentTask).filter(AgentTask.task_id == task_id).with_for_update().first()
            if not task or task.status in {"completed", "failed", "cancelled"} or task.cancel_requested:
                return None
            if task.lease_until and task.lease_until > now and task.lease_owner != owner:
                return None
            task.lease_owner = owner
            task.lease_until = until
            task.status = "running"
            task.updated_at = now
            budget = _json_load(task.budget_json, {})
            if not budget.get("started_at"):
                budget["started_at"] = now.isoformat()
                task.budget_json = _json_dump(budget)
            db.commit()
            return _task_to_dict(task)

    def release_task_lease(self, task_id: str, owner: str, *, error: str | None = None) -> None:
        with SessionLocal() as db:
            task = db.query(AgentTask).filter(AgentTask.task_id == task_id, AgentTask.lease_owner == owner).first()
            if not task:
                return
            task.lease_owner = None
            task.lease_until = None
            if error:
                task.last_error = error[:4000]
            db.commit()

    def pending_outbox(self, limit: int = 100) -> list[dict[str, Any]]:
        now = utcnow()
        with SessionLocal() as db:
            rows = (
                db.query(AgentOutbox)
                .filter(AgentOutbox.status == "pending", AgentOutbox.available_at <= now)
                .order_by(AgentOutbox.id.asc())
                .limit(max(1, min(int(limit), 500)))
                .all()
            )
            return [{"id": row.id, "task_id": row.task_id, "attempts": row.attempts} for row in rows]

    def mark_outbox_dispatched(self, outbox_id: int) -> None:
        with SessionLocal() as db:
            row = db.query(AgentOutbox).filter(AgentOutbox.id == outbox_id).first()
            if row:
                row.status = "dispatched"
                row.dispatched_at = utcnow()
                row.attempts = int(row.attempts or 0) + 1
                db.commit()

    def mark_outbox_failed(self, outbox_id: int, error: str) -> None:
        with SessionLocal() as db:
            row = db.query(AgentOutbox).filter(AgentOutbox.id == outbox_id).first()
            if row:
                row.attempts = int(row.attempts or 0) + 1
                row.last_error = error[:4000]
                row.available_at = utcnow() + timedelta(seconds=min(300, 2 ** min(row.attempts, 8)))
                db.commit()

    def append_event(self, event: AgentEvent) -> int | None:
        payload = dict(event)
        # Parallel Send branches can finish concurrently.  Allocate a stable
        # per-task sequence with a short retry rather than silently dropping a
        # duplicate event index.
        for _ in range(4):
            with SessionLocal() as db:
                event_index = payload.get("event_index")
                if event_index is None:
                    max_index = (
                        db.query(func.max(AgentEventModel.event_index))
                        .filter(AgentEventModel.task_id == payload["task_id"])
                        .scalar()
                    )
                    event_index = int(max_index or 0) + 1
                row = AgentEventModel(
                    session_id=payload["session_id"], task_id=payload["task_id"], run_id=payload.get("run_id"),
                    event_type=payload["event_type"], event_index=int(event_index), agent_name=payload.get("agent_name"),
                    skill_name=payload.get("skill_name"), tool_name=payload.get("tool_name"), step_id=payload.get("step_id"),
                    message=payload["message"], payload_json=_json_dump(payload.get("payload") or {}),
                    created_at=payload.get("created_at") or utcnow(),
                )
                db.add(row)
                try:
                    db.commit()
                    return int(event_index)
                except IntegrityError:
                    db.rollback()
                    payload.pop("event_index", None)
        return None

    def list_events(self, task_id: str, after_index: int = 0) -> list[dict[str, Any]]:
        with SessionLocal() as db:
            rows = (
                db.query(AgentEventModel)
                .filter(
                    AgentEventModel.task_id == task_id,
                    AgentEventModel.event_index > after_index,
                )
                .order_by(AgentEventModel.event_index.asc(), AgentEventModel.id.asc())
                .all()
            )
            return [_event_to_dict(row) for row in rows]

    def save_plan(self, payload: dict[str, Any]) -> None:
        with SessionLocal() as db:
            db.add(
                AgentPlan(
                    task_id=payload["task_id"],
                    run_id=payload["run_id"],
                    plan_version=int(payload.get("plan_version") or 1),
                    source=payload.get("source") or "fallback",
                    status=payload.get("status") or "active",
                    goal=payload.get("goal") or "",
                    intent=payload.get("intent") or "general",
                    plan_json=_json_dump(payload.get("steps") or []),
                    error_message=payload.get("error_message"),
                    created_at=utcnow(),
                )
            )
            db.commit()

    def upsert_step(self, payload: dict[str, Any]) -> None:
        with SessionLocal() as db:
            row = (
                db.query(AgentStep)
                .filter(
                    AgentStep.run_id == payload["run_id"],
                    AgentStep.step_id == payload["step_id"],
                )
                .first()
            )
            if row is None:
                row = AgentStep(
                    plan_id=payload.get("plan_id"),
                    task_id=payload["task_id"],
                    run_id=payload["run_id"],
                    step_id=payload["step_id"],
                    step_index=int(payload.get("step_index") or 0),
                    created_at=utcnow(),
                )
                db.add(row)

            row.agent_name = payload.get("agent_name", row.agent_name)
            row.skill_name = payload.get("skill_name", row.skill_name)
            row.tool_name = payload.get("tool_name", row.tool_name)
            row.objective = payload.get("objective", row.objective)
            row.status = payload.get("status", row.status or "pending")
            row.plan_id = payload.get("plan_id", row.plan_id)
            if "input" in payload:
                row.input_json = _json_dump(payload.get("input") or {})
            if "output" in payload:
                row.output_json = _json_dump(payload.get("output") or {})
            if "error" in payload:
                error = payload.get("error") or {}
                row.error_json = _json_dump(error)
                if isinstance(error, dict):
                    row.error_type = error.get("type") or error.get("error_type") or row.error_type
                    row.error_message = error.get("message") or error.get("error_message") or row.error_message
            if payload.get("started_at"):
                row.started_at = payload["started_at"]
            if payload.get("finished_at"):
                row.finished_at = payload["finished_at"]
            row.updated_at = utcnow()
            db.commit()

    def save_tool_call(self, payload: dict[str, Any]) -> None:
        with SessionLocal() as db:
            db.add(
                AgentToolCall(
                    task_id=payload["task_id"],
                    run_id=payload.get("run_id"),
                    step_id=payload.get("step_id"),
                    agent_name=payload.get("agent_name"),
                    skill_name=payload.get("skill_name"),
                    tool_name=payload["tool_name"],
                    input_json=_json_dump(payload.get("input") or {}),
                    output_json=_json_dump(payload.get("output", payload.get("result") or {})),
                    result_json=_json_dump(payload.get("result", payload.get("output") or {})),
                    ok=1 if payload.get("ok", True) else 0,
                    error_type=payload.get("error_type"),
                    error_message=payload.get("error_message"),
                    retry_count=int(payload.get("retry_count") or 0),
                    latency_ms=payload.get("latency_ms"),
                    prompt_tokens=payload.get("prompt_tokens"),
                    completion_tokens=payload.get("completion_tokens"),
                    created_at=utcnow(),
                )
            )
            db.commit()

    def save_verification(self, payload: dict[str, Any]) -> None:
        with SessionLocal() as db:
            db.add(
                AgentVerification(
                    task_id=payload["task_id"],
                    run_id=payload.get("run_id"),
                    step_id=payload.get("step_id"),
                    target_type=payload.get("target_type"),
                    target_id=payload.get("target_id"),
                    status=payload.get("status", "unknown"),
                    score=payload.get("score"),
                    evidence_json=_json_dump(payload.get("evidence") or []),
                    issues_json=_json_dump(payload.get("issues") or []),
                    payload_json=_json_dump(payload),
                    created_at=utcnow(),
                )
            )
            db.commit()


agent_store = AgentRuntimeStore()
