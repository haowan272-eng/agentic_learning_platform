"""记忆事件写入器。

将 Agent 运行过程中产生的记忆事件持久化到 PostgreSQL MemoryEvent 表，
同时推送到 Redis 热窗口供后续合并消费。自动清洗 PII/密钥等敏感信息。
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any

from sqlalchemy.orm import Session

from app.models.agent_runtime import MemoryEvent
from app.core.config import AGENT_MEMORY_MAX_EVENT_CHARS

from .short_term import append_recent_event


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def sanitize_memory_content(content: str) -> str:
    """Prevent common PII/secrets from becoming durable learner profile data."""
    value = " ".join((content or "").split())[:AGENT_MEMORY_MAX_EVENT_CHARS]
    value = re.sub(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[redacted-email]", value)
    value = re.sub(r"(?<!\d)1[3-9]\d{9}(?!\d)", "[redacted-phone]", value)
    value = re.sub(r"(?i)\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*\S+", "[redacted-secret]", value)
    return value


def write_memory_event(
    db: Session,
    *,
    user_id: int | None,
    session_id: str | None,
    task_id: str | None,
    event_type: str,
    category: str | None,
    content: str,
    source: str = "agent",
    metadata: dict[str, Any] | None = None,
) -> MemoryEvent | None:
    """Write a traceable memory event.

    user_id is required for durable memory; anonymous events are intentionally
    ignored because long-term learner profile must be user-isolated.
    """
    if user_id is None:
        return None
    clean_content = sanitize_memory_content(content)
    if not clean_content:
        return None
    metadata = dict(metadata or {})
    metadata.setdefault("memory_approved", event_type in {
        "user_goal_set", "weak_point_detected", "topic_mastered", "constraint_confirmed",
        "preference_detected", "resume_feedback_received", "project_context_updated",
    })
    row = MemoryEvent(
        user_id=int(user_id),
        session_id=session_id,
        task_id=task_id,
        event_type=event_type,
        category=category,
        content=clean_content,
        source=source,
        metadata_json=json.dumps(metadata, ensure_ascii=False, default=_json_default),
    )
    db.add(row)
    db.flush()
    append_recent_event(
        user_id,
        session_id,
        {
            "event_type": event_type,
            "category": category,
            "content": row.content,
            "source": source,
            "task_id": task_id,
            "memory_event_id": row.id,
            "metadata": metadata,
        },
    )
    return row
