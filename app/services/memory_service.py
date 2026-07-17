"""RAG主链路的私有长期记忆读写服务"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import (
    RAG_MEMORY_EXTRACTION_INTERVAL_MESSAGES,
    RAG_MEMORY_EXTRACTION_MIN_CHARS,
)
from app.models import UserMemory
from app.services.memory_extractor import extract_from_conversation, extract_from_message
from app.services.short_term_memory import load_short_term_messages


@dataclass(frozen=True)
class MemorySnapshot:
    keyword: str
    category: str
    weight: float

    def as_dict(self) -> dict:
        return {
            "keyword": self.keyword,
            "category": self.category,
            "weight": self.weight,
        }


def _upsert_candidates(
    db: Session,
    user_id: int,
    conversation_id: int,
    extracted: list[dict],
) -> list[str]:
    dialect = db.get_bind().dialect.name
    for item in extracted:
        values = {
            "user_id": user_id,
            "keyword": item["keyword"][:128],
            "category": item.get("category") or "other",
            "weight": float(item.get("weight", 1.0)),
            "source_conversation_id": conversation_id,
        }
        if dialect == "postgresql":
            from sqlalchemy.dialects.postgresql import insert

            statement = insert(UserMemory).values(**values)
            statement = statement.on_conflict_do_update(
                constraint="uq_user_memory_keyword",
                set_={
                    "weight": UserMemory.weight + statement.excluded.weight,
                    "source_conversation_id": conversation_id,
                    "updated_at": func.now(),
                },
            )
            db.execute(statement)
        elif dialect == "sqlite":
            from sqlalchemy.dialects.sqlite import insert

            statement = insert(UserMemory).values(**values)
            statement = statement.on_conflict_do_update(
                index_elements=["user_id", "keyword", "category"],
                set_={
                    "weight": UserMemory.weight + statement.excluded.weight,
                    "source_conversation_id": conversation_id,
                    "updated_at": func.now(),
                },
            )
            db.execute(statement)
        else:
            existing = db.query(UserMemory).filter(
                UserMemory.user_id == user_id,
                UserMemory.keyword == values["keyword"],
                UserMemory.category == values["category"],
            ).first()
            if existing:
                existing.weight += values["weight"]
                existing.source_conversation_id = conversation_id
            else:
                db.add(UserMemory(**values))
    db.flush()
    return [item["keyword"] for item in extracted]


def _is_effective_memory_candidate(content: str) -> bool:
    text = " ".join((content or "").split())
    short_commands = {
        "好", "好的", "可以", "行", "嗯", "继续", "执行", "开始", "不要", "是", "不是",
        "ok", "yes", "no",
    }
    return bool(text) and text.lower() not in short_commands


def _meets_memory_batch_threshold(content: str) -> bool:
    text = " ".join((content or "").split())
    return _is_effective_memory_candidate(text) and len(text) >= RAG_MEMORY_EXTRACTION_MIN_CHARS


def remember_user_message(
    db: Session,
    user_id: int,
    conversation_id: int,
    content: str,
) -> list[str]:
    """兼容单消息提取入口；新问答主链使用短期窗口入口"""
    if not _meets_memory_batch_threshold(content):
        return []
    return _upsert_candidates(
        db,
        user_id,
        conversation_id,
        extract_from_message(content),
    )


def remember_short_term_window(
    db: Session,
    user_id: int,
    conversation_id: int,
) -> list[str]:
    """从Redis短期窗口处理尚未沉淀的消息，缓存不可用时回源PostgreSQL"""
    from app.models import RagConversation, RagMessage

    recent = load_short_term_messages(db, user_id, conversation_id)
    cached_content = {row["message_id"]: row["content"] for row in recent}
    pending = (
        db.query(RagMessage)
        .join(RagConversation, RagConversation.id == RagMessage.conversation_id)
        .filter(
            RagMessage.conversation_id == conversation_id,
            RagConversation.user_id == user_id,
            RagMessage.role == "user",
            RagMessage.memory_extracted.is_(False),
        )
        .order_by(RagMessage.id)
        .all()
    )
    if not pending:
        return []

    has_existing_memory = db.query(UserMemory.id).filter(UserMemory.user_id == user_id).first() is not None
    candidate = _meets_memory_batch_threshold if has_existing_memory else _is_effective_memory_candidate
    effective_rows = [
        row for row in pending
        if candidate(cached_content.get(row.id, row.content))
    ]
    if has_existing_memory and len(effective_rows) < RAG_MEMORY_EXTRACTION_INTERVAL_MESSAGES:
        return []
    if not has_existing_memory and not effective_rows:
        return []

    messages = [
        {
            "role": "user",
            "content": cached_content.get(row.id, row.content),
        }
        for row in effective_rows
    ]
    keywords = _upsert_candidates(
        db,
        user_id,
        conversation_id,
        extract_from_conversation(messages),
    )
    for row in pending:
        row.memory_extracted = True
    db.flush()
    return keywords


def load_memory(db: Session, user_id: int, limit: int = 12) -> list[MemorySnapshot]:
    rows = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user_id)
        .order_by(UserMemory.weight.desc(), UserMemory.updated_at.desc())
        .limit(max(0, min(limit, 30)))
        .all()
    )
    return [
        MemorySnapshot(row.keyword, row.category or "other", float(row.weight or 0.0))
        for row in rows
    ]


def memory_context(memories: list[MemorySnapshot]) -> str:
    if not memories:
        return "（无长期记忆）"
    labels = {
        "destination": "关注地点",
        "preference": "偏好",
        "budget": "预算",
        "constraint": "限制",
        "project": "项目",
        "topic": "主题",
        "other": "其他",
    }
    return "\n".join(
        f"- {labels.get(item.category, item.category)}：{item.keyword}"
        for item in memories
    )


def retrieval_query(question: str, memories: list[MemorySnapshot]) -> str:
    """保持用户原问题为主体，只追加短记忆条件，避免长期记忆淹没Query"""
    if not memories:
        return question
    keywords = ", ".join(item.keyword for item in memories[:8])
    return f"{question}\n用户偏好条件：{keywords}"


__all__ = [
    "MemorySnapshot",
    "load_memory",
    "memory_context",
    "remember_user_message",
    "remember_short_term_window",
    "retrieval_query",
]
