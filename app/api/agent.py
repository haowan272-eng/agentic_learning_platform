from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.agent_runtime.store import agent_store
from app.agent_runtime.event_bus import read_task_events
from app.agent_runtime.tools import list_tools
from app.api.deps import check_kb_role, get_current_user
from app.core.database import get_db
from app.models import Document, User
from app.schemas.agent import (
    AgentEventResponse,
    AgentTaskResponse,
    CreateAgentTaskRequest,
    CreateAgentTaskResponse,
    ResumeAgentTaskRequest,
)
from app.tasks.agent_runtime import run_agent_task_task


router = APIRouter(prefix="/agent", tags=["Agent Runtime"])


def _resolve_user_id(db: Session, username: str) -> int:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return int(user.id)


def _to_task_response(task: dict[str, Any]) -> AgentTaskResponse:
    return AgentTaskResponse(
        session_id=task["session_id"],
        task_id=task["task_id"],
        run_id=task["run_id"],
        status=task["status"],
        user_input=task["user_input"],
        task_type=task["task_type"],
        kb_id=task.get("kb_id"),
        document_id=task.get("document_id"),
        conversation_id=task.get("conversation_id"),
        final_answer=task.get("final_answer"),
        cancel_requested=bool(task.get("cancel_requested")),
        created_at=task["created_at"],
        updated_at=task["updated_at"],
    )


@router.post("/tasks", response_model=CreateAgentTaskResponse)
def create_agent_task(
    body: CreateAgentTaskRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CreateAgentTaskResponse:
    user_id = _resolve_user_id(db, current_user)
    if body.kb_id is not None:
        check_kb_role(db, current_user, body.kb_id, "viewer")
    if body.document_id is not None:
        document = db.query(Document).filter(Document.id == body.document_id).first()
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        if document.kb_id is not None:
            check_kb_role(db, current_user, document.kb_id, "viewer")
        elif document.user_id != user_id:
            raise HTTPException(status_code=403, detail="Document is outside your personal scope")
        if body.kb_id is not None and document.kb_id != body.kb_id:
            raise HTTPException(status_code=400, detail="document_id does not belong to kb_id")
    task = agent_store.create_task({**body.model_dump(), "username": current_user, "user_id": user_id})
    # The durable outbox is the source of truth. A temporary broker outage no
    # longer turns a valid task into a failed task.
    for outbox in agent_store.pending_outbox(limit=20):
        if outbox["task_id"] != task["task_id"]:
            continue
        try:
            run_agent_task_task.delay(task["task_id"])
            agent_store.mark_outbox_dispatched(outbox["id"])
        except Exception as exc:  # noqa: BLE001
            agent_store.mark_outbox_failed(outbox["id"], str(exc))
        break
    return CreateAgentTaskResponse(
        session_id=task["session_id"],
        task_id=task["task_id"],
        run_id=task["run_id"],
        status=task["status"],
    )


@router.post("/tasks/{task_id}/cancel", response_model=AgentTaskResponse)
def cancel_agent_task(
    task_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentTaskResponse:
    user_id = _resolve_user_id(db, current_user)
    if not agent_store.request_cancel(task_id, user_id):
        raise HTTPException(status_code=404, detail="Agent task not found or already finished")
    task = agent_store.get_task(task_id, user_id=user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Agent task not found")
    agent_store.append_event({
        "session_id": task["session_id"], "task_id": task_id, "run_id": task["run_id"],
        "event_type": "task.cancel_requested", "agent_name": "api", "message": "Cancellation requested by user.", "payload": {},
    })
    return _to_task_response(task)


@router.post("/tasks/{task_id}/resume", response_model=AgentTaskResponse)
def resume_agent_task(
    task_id: str,
    body: ResumeAgentTaskRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentTaskResponse:
    """Resume a LangGraph interrupt using the same task/thread id."""
    user_id = _resolve_user_id(db, current_user)
    payload = body.model_dump(exclude_none=True)
    task = agent_store.request_resume(task_id, user_id, payload)
    if not task:
        raise HTTPException(status_code=409, detail="Agent task is not waiting for user approval")
    agent_store.append_event({
        "session_id": task["session_id"], "task_id": task_id, "run_id": task["run_id"],
        "event_type": "approval.submitted", "agent_name": "api", "message": "User submitted an approval response.", "payload": payload,
    })
    for outbox in agent_store.pending_outbox(limit=20):
        if outbox["task_id"] != task_id:
            continue
        try:
            run_agent_task_task.delay(task_id)
            agent_store.mark_outbox_dispatched(outbox["id"])
        except Exception as exc:  # noqa: BLE001
            agent_store.mark_outbox_failed(outbox["id"], str(exc))
        break
    return _to_task_response(task)


@router.get("/tasks", response_model=list[AgentTaskResponse])
def list_agent_tasks(
    session_id: str | None = None,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AgentTaskResponse]:
    user_id = _resolve_user_id(db, current_user)
    return [_to_task_response(task) for task in agent_store.list_tasks(session_id=session_id, user_id=user_id)]


@router.get("/tasks/{task_id}", response_model=AgentTaskResponse)
def get_agent_task(
    task_id: str,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> AgentTaskResponse:
    user_id = _resolve_user_id(db, current_user)
    task = agent_store.get_task(task_id, user_id=user_id)
    if not task:
        raise HTTPException(status_code=404, detail="Agent task not found")
    return _to_task_response(task)


@router.get("/tasks/{task_id}/events", response_model=list[AgentEventResponse])
def list_agent_events(
    task_id: str,
    after_index: int = 0,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> list[AgentEventResponse]:
    user_id = _resolve_user_id(db, current_user)
    if not agent_store.get_task(task_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Agent task not found")
    return [AgentEventResponse(**event) for event in agent_store.list_events(task_id, after_index=after_index)]


@router.get("/tasks/{task_id}/stream")
def stream_agent_events(
    task_id: str,
    after_index: int = 0,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> StreamingResponse:
    user_id = _resolve_user_id(db, current_user)
    if not agent_store.get_task(task_id, user_id=user_id):
        raise HTTPException(status_code=404, detail="Agent task not found")

    async def event_generator():
        cursor = after_index
        stream_cursor = "$"
        idle_ticks = 0
        while idle_ticks < 120:
            events = agent_store.list_events(task_id, after_index=cursor)
            if events:
                idle_ticks = 0
                for event in events:
                    cursor = max(cursor, int(event["event_index"]))
                    yield f"event: {event['event_type']}\ndata: {json.dumps(_json_ready(event), ensure_ascii=False)}\n\n"
            else:
                idle_ticks += 1
                stream_cursor, pushed = await asyncio.to_thread(read_task_events, task_id, stream_cursor, block_ms=500)
                if pushed:
                    idle_ticks = 0
                    for event in pushed:
                        # Token chunks are intentionally transient: persisting
                        # every token would bloat the event table. Redis Stream
                        # cursoring already prevents duplicate delivery.
                        if event.get("event_type") == "llm.token":
                            yield f"event: llm.token\ndata: {json.dumps(_json_ready(event), ensure_ascii=False)}\n\n"
                            continue
                        if int(event.get("event_index") or 0) <= cursor:
                            continue
                        cursor = int(event["event_index"])
                        yield f"event: {event['event_type']}\ndata: {json.dumps(_json_ready(event), ensure_ascii=False)}\n\n"
                else:
                    yield ": heartbeat\n\n"
            task = agent_store.get_task(task_id, user_id=user_id)
            if task and task["status"] in {"completed", "failed", "cancelled"} and not events:
                break
            await asyncio.sleep(0.5)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tools")
def get_agent_tools(current_user: str = Depends(get_current_user)) -> list[dict[str, Any]]:
    return list_tools()


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
