from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import AGENT_MEMORY_DECAY_ENABLED, AGENT_MEMORY_DECAY_HALF_LIFE_DAYS
from app.models import UserMemory


PROFILE_CATEGORIES = (
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
)


def _memory_key(row: UserMemory) -> str:
    return str(row.memory_key or row.key or row.keyword or "").strip()


def _memory_value(row: UserMemory) -> str:
    return str(row.value or row.keyword or row.memory_key or row.key or "").strip()


def _as_item(row: UserMemory) -> dict[str, Any]:
    return {
        "id": row.id,
        "key": _memory_key(row),
        "category": row.category or "other",
        "value": _memory_value(row),
        "weight": float(row.weight or 0.0),
        "confidence": float(row.confidence) if row.confidence is not None else None,
        "source": row.source or "agent",
        "source_task_id": row.source_task_id,
        "source_event_id": row.source_event_id,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _compute_decay_factor(
    updated_at: datetime | None, *, now: datetime | None = None,
) -> float:
    """Exponential decay: 0.5^(age_days / half_life_days).

    A memory that hasn't been updated for *half_life_days* gets its weight
    halved.  Memories updated recently (age ≈ 0) retain their full weight.
    """
    if not AGENT_MEMORY_DECAY_ENABLED:
        return 1.0
    if updated_at is None:
        return 0.5  # never-updated memories are treated as aged
    ref = now or datetime.now(timezone.utc)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (ref - updated_at).total_seconds() / 86400.0)
    half_life = max(1, AGENT_MEMORY_DECAY_HALF_LIFE_DAYS)
    return 0.5 ** (age_days / half_life)


def apply_memory_decay(row: UserMemory, *, now: datetime | None = None) -> float:
    """Return the decay-adjusted effective weight for a memory row.

    The stored weight is left untouched (soft decay); callers use the
    effective weight for ranking without mutating the database.
    """
    stored = float(row.weight or 0.0)
    factor = _compute_decay_factor(row.updated_at, now=now)
    return round(stored * factor, 6)


def load_user_profile(db: Session, user_id: int | None, *, limit: int = 80) -> dict[str, Any]:
    """Load long-term learner profile from user_memories.

    The table still keeps RAG's legacy keyword memory, so this reader normalizes
    both old keyword rows and the new structured memory rows into one profile.
    """
    if user_id is None:
        return {"user_id": None, "items": [], **{category: [] for category in PROFILE_CATEGORIES}}

    now = datetime.now(timezone.utc)
    rows = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == int(user_id))
        .filter((UserMemory.is_active == 1) | (UserMemory.is_active.is_(None)))
        .order_by(UserMemory.weight.desc(), UserMemory.updated_at.desc(), UserMemory.id.desc())
        .limit(max(1, min(int(limit), 200)))
        .all()
    )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    items: list[dict[str, Any]] = []
    for row in rows:
        item = _as_item(row)
        if not item["value"]:
            continue
        effective = apply_memory_decay(row, now=now)
        item["effective_weight"] = effective
        category = item["category"] if item["category"] in PROFILE_CATEGORIES else "other"
        grouped[category].append(item)
        items.append(item)

    # Re-sort by decay-adjusted effective weight so fresh/relevant
    # memories surface above stale ones, even when stored weight is high.
    items.sort(key=lambda i: float(i.get("effective_weight", 0)), reverse=True)
    for cat in grouped:
        grouped[cat].sort(key=lambda i: float(i.get("effective_weight", 0)), reverse=True)

    profile: dict[str, Any] = {"user_id": int(user_id), "items": items}
    for category in PROFILE_CATEGORIES:
        profile[category] = grouped.get(category, [])

    # Friendly aliases used by Planner/Verifier prompts.
    profile["goals"] = grouped.get("learning_goal", []) + grouped.get("career_goal", [])
    profile["weak_points"] = grouped.get("weak_point", [])
    profile["mastered_topics"] = grouped.get("mastered_topic", [])
    profile["constraints"] = grouped.get("constraint", [])
    profile["preferences"] = grouped.get("preference", [])
    return profile
