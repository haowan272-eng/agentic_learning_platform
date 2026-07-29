"""记忆合并器——将 MemoryEvent 转换为 UserMemory 持久化条目。

按 CONSOLIDATION_POLICY 将事件类型映射为记忆分类和基础权重，
支持 PostgreSQL / SQLite 双后端 upsert，并应用衰减系数。
"""
from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import UserMemory
from app.models.agent_runtime import MemoryEvent
from app.core.config import AGENT_MEMORY_DECAY_ENABLED, AGENT_MEMORY_DECAY_HALF_LIFE_DAYS, AGENT_MEMORY_MIN_CONFIDENCE


CONSOLIDATION_POLICY: dict[str, tuple[str, float]] = {
    "user_goal_set": ("learning_goal", 1.2),
    "weak_point_detected": ("weak_point", 1.5),
    "topic_mastered": ("mastered_topic", 1.1),
    "project_gap_found": ("project_gap", 1.4),
    "constraint_confirmed": ("constraint", 1.3),
    "preference_detected": ("preference", 1.1),
    "resume_feedback_received": ("resume_feedback", 1.4),
    "project_context_updated": ("project_context", 1.0),
    "task_completed": ("project_context", 0.6),
    "verification_failed": ("weak_point", 0.8),
}


def _slug(text: str, *, max_len: int = 128) -> str:
    compact = re.sub(r"\s+", "_", (text or "").strip().lower())
    compact = re.sub(r"[^\w\u4e00-\u9fff.-]+", "_", compact).strip("_")
    return (compact or "memory")[:max_len]


def _load_metadata(row: MemoryEvent) -> dict[str, Any]:
    try:
        return json.loads(row.metadata_json or "{}")
    except json.JSONDecodeError:
        return {}


def _compute_existing_weight(db: Session, *, user_id: int, memory_key: str, category: str) -> float:
    """Return the decay-adjusted weight of an existing memory, or 0.0."""
    from .profile import apply_memory_decay

    row = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == user_id,
            UserMemory.keyword == memory_key,
            UserMemory.category == category,
        )
        .first()
    )
    if row is None:
        return 0.0
    return apply_memory_decay(row)


def _upsert_memory(
    db: Session,
    *,
    user_id: int,
    memory_key: str,
    category: str,
    value: str,
    weight: float,
    confidence: float,
    source_event_id: int,
    source_task_id: str | None,
) -> None:
    # Apply decay to the existing stored weight so stale memories are
    # naturally down-weighted at each consolidation cycle.
    old_weight = _compute_existing_weight(db, user_id=user_id, memory_key=memory_key, category=category)
    effective_weight = old_weight + weight

    values = {
        "user_id": user_id,
        "memory_type": "learner_profile",
        "key": memory_key,
        "keyword": memory_key,
        "memory_key": memory_key,
        "category": category,
        "value": value,
        "weight": effective_weight,
        "confidence": confidence,
        "source": "agent_consolidator",
        "is_active": 1,
        "source_event_id": source_event_id,
        "source_task_id": source_task_id,
    }
    dialect = db.get_bind().dialect.name
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert

        statement = insert(UserMemory).values(**values)
        statement = statement.on_conflict_do_update(
            constraint="uq_user_memory_keyword",
            set_={
                "memory_key": statement.excluded.memory_key,
                "key": statement.excluded.key,
                "value": statement.excluded.value,
                "weight": statement.excluded.weight,
                "confidence": func.greatest(UserMemory.confidence, statement.excluded.confidence),
                "source": statement.excluded.source,
                "is_active": 1,
                "source_event_id": statement.excluded.source_event_id,
                "source_task_id": statement.excluded.source_task_id,
                "updated_at": func.now(),
            },
        )
        db.execute(statement)
        return

    existing = (
        db.query(UserMemory)
        .filter(
            UserMemory.user_id == user_id,
            UserMemory.keyword == memory_key,
            UserMemory.category == category,
        )
        .first()
    )
    if existing:
        existing.memory_key = memory_key
        existing.key = memory_key
        existing.value = value
        existing.weight = effective_weight
        existing.confidence = max(float(existing.confidence or 0.0), confidence)
        existing.source_event_id = source_event_id
        existing.source_task_id = source_task_id
    else:
        db.add(UserMemory(**values))


def consolidate_memory_events(
    db: Session,
    *,
    user_id: int | None,
    session_id: str | None = None,
    task_id: str | None = None,
    limit: int = 80,
) -> list[dict[str, Any]]:
    if user_id is None:
        return []

    query = db.query(MemoryEvent).filter(MemoryEvent.user_id == int(user_id))
    if task_id:
        query = query.filter(MemoryEvent.task_id == task_id)
    elif session_id:
        query = query.filter(MemoryEvent.session_id == session_id)

    rows = query.order_by(MemoryEvent.id.desc()).limit(max(1, min(int(limit), 200))).all()
    consolidated = []
    for row in reversed(rows):
        category, base_weight = CONSOLIDATION_POLICY.get(
            row.event_type,
            (row.category or "other", 0.4),
        )
        if category not in {
            "learning_goal",
            "career_goal",
            "project_context",
            "weak_point",
            "mastered_topic",
            "preference",
            "constraint",
            "tech_stack",
            "resume_feedback",
            "project_gap",
            "other",
        }:
            category = "other"
        metadata = _load_metadata(row)
        if not bool(metadata.get("memory_approved")):
            continue
        value = " ".join((row.content or "").split())
        if not value:
            continue
        memory_key = str(metadata.get("memory_key") or _slug(f"{category}_{value[:60]}"))
        confidence = float(metadata.get("confidence") or 0.75)
        if confidence < AGENT_MEMORY_MIN_CONFIDENCE:
            continue
        weight = float(metadata.get("weight") or base_weight)
        _upsert_memory(
            db,
            user_id=int(user_id),
            memory_key=memory_key,
            category=category,
            value=value,
            weight=weight,
            confidence=confidence,
            source_event_id=row.id,
            source_task_id=row.task_id,
        )
        consolidated.append(
            {
                "memory_key": memory_key,
                "category": category,
                "value": value,
                "source_event_id": row.id,
            }
        )
    db.flush()
    return consolidated
