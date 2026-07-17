from __future__ import annotations

from typing import Any

from app.core.database import SessionLocal

from .context import build_agent_context
from .consolidator import consolidate_memory_events
from .events import write_memory_event
from .profile import load_user_profile
from .summarizer import summarize_session


def read_profile_for_user(user_id: int | None) -> dict[str, Any]:
    with SessionLocal() as db:
        return load_user_profile(db, user_id)


def build_context_for_state(state: dict[str, Any]) -> dict[str, Any]:
    with SessionLocal() as db:
        return build_agent_context(db, state)


def record_memory_event(
    *,
    user_id: int | None,
    session_id: str | None,
    task_id: str | None,
    event_type: str,
    category: str | None,
    content: str,
    source: str = "agent",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    with SessionLocal() as db:
        row = write_memory_event(
            db,
            user_id=user_id,
            session_id=session_id,
            task_id=task_id,
            event_type=event_type,
            category=category,
            content=content,
            source=source,
            metadata=metadata,
        )
        db.commit()
        if row is None:
            return None
        return {"id": row.id, "event_type": row.event_type, "category": row.category, "content": row.content}


def consolidate_task_memory(state: dict[str, Any]) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        result = consolidate_memory_events(
            db,
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
            task_id=state.get("task_id"),
        )
        db.commit()
        return result


def summarize_task_session(state: dict[str, Any]) -> dict[str, Any] | None:
    with SessionLocal() as db:
        result = summarize_session(
            db,
            user_id=state.get("user_id"),
            session_id=state.get("session_id"),
            task_id=state.get("task_id"),
        )
        db.commit()
        return result
