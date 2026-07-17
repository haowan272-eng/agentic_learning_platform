"""Redis短期对话窗口；PostgreSQL始终是完整消息的事实源"""
from __future__ import annotations

from datetime import datetime
import json
import logging
from typing import Any

from redis.exceptions import RedisError
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import (
    SHORT_TERM_MEMORY_MESSAGES,
    SHORT_TERM_MEMORY_TTL_SECONDS,
    SHORT_TERM_MESSAGE_MAX_CHARS,
)
from app.models import RagConversation, RagMessage
from app.core.redis import get_redis

logger = logging.getLogger(__name__)


def short_term_key(user_id: int, conversation_id: int) -> str:
    """user_id进入Key，避免不同用户会话缓存发生碰撞"""
    return f"rag:short_term:{user_id}:{conversation_id}"


def _payload(
    message_id: int,
    role: str,
    content: str,
    created_at: Any = None,
) -> dict:
    timestamp = created_at.isoformat() if isinstance(created_at, datetime) else str(created_at or "")
    return {
        "message_id": int(message_id),
        "role": role,
        "content": " ".join((content or "").split())[:SHORT_TERM_MESSAGE_MAX_CHARS],
        "created_at": timestamp,
    }


def append_short_term_message(
    user_id: int,
    conversation_id: int,
    message_id: int,
    role: str,
    content: str,
    created_at: Any = None,
) -> bool:
    """尽力更新Redis热窗口；失败仅影响性能，不影响消息持久化"""
    client = get_redis()
    if client is None:
        return False
    key = short_term_key(user_id, conversation_id)
    encoded = json.dumps(
        _payload(message_id, role, content, created_at),
        ensure_ascii=False,
    )
    try:
        with client.pipeline(transaction=True) as pipeline:
            pipeline.rpush(key, encoded)
            pipeline.ltrim(key, -SHORT_TERM_MEMORY_MESSAGES, -1)
            pipeline.expire(key, SHORT_TERM_MEMORY_TTL_SECONDS)
            pipeline.execute()
        return True
    except RedisError as exc:
        logger.warning("写入短期记忆缓存失败，将继续使用PostgreSQL: %s", exc)
        return False


def _decode_rows(rows: list[Any]) -> list[dict]:
    decoded: list[dict] = []
    for raw in rows:
        try:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            item = json.loads(raw)
            message_id = int(item["message_id"])
            role = str(item["role"])
            content = str(item["content"])
            if role not in {"user", "assistant", "system"}:
                raise ValueError("invalid role")
            decoded.append({
                "message_id": message_id,
                "role": role,
                "content": content,
                "created_at": str(item.get("created_at") or ""),
            })
        except (TypeError, ValueError, KeyError, json.JSONDecodeError, UnicodeDecodeError):
            return []
    return decoded


def _load_from_postgres(
    db: Session,
    user_id: int,
    conversation_id: int,
    limit: int,
) -> list[dict]:
    rows = (
        db.query(RagMessage)
        .join(RagConversation, RagConversation.id == RagMessage.conversation_id)
        .filter(
            RagMessage.conversation_id == conversation_id,
            RagConversation.user_id == user_id,
        )
        .order_by(RagMessage.id.desc())
        .limit(limit)
        .all()
    )
    return [
        _payload(row.id, row.role, row.content, row.created_at)
        for row in reversed(rows)
    ]


def _replace_cache(user_id: int, conversation_id: int, messages: list[dict]) -> None:
    client = get_redis()
    if client is None or not messages:
        return
    key = short_term_key(user_id, conversation_id)
    try:
        with client.pipeline(transaction=True) as pipeline:
            pipeline.delete(key)
            pipeline.rpush(
                key,
                *(json.dumps(item, ensure_ascii=False) for item in messages),
            )
            pipeline.ltrim(key, -SHORT_TERM_MEMORY_MESSAGES, -1)
            pipeline.expire(key, SHORT_TERM_MEMORY_TTL_SECONDS)
            pipeline.execute()
    except RedisError as exc:
        logger.warning("重建短期记忆缓存失败: %s", exc)


def load_short_term_messages(
    db: Session,
    user_id: int,
    conversation_id: int,
    limit: int = SHORT_TERM_MEMORY_MESSAGES,
) -> list[dict]:
    """优先读Redis；未命中、损坏或异常时回源PostgreSQL并重建缓存"""
    limit = max(1, min(int(limit), SHORT_TERM_MEMORY_MESSAGES))
    client = get_redis()
    if client is not None:
        key = short_term_key(user_id, conversation_id)
        try:
            cached = _decode_rows(client.lrange(key, -limit, -1))
            if cached:
                latest_id = (
                    db.query(func.max(RagMessage.id))
                    .join(RagConversation, RagConversation.id == RagMessage.conversation_id)
                    .filter(
                        RagMessage.conversation_id == conversation_id,
                        RagConversation.user_id == user_id,
                    )
                    .scalar()
                )
                if latest_id is not None and cached[-1]["message_id"] == int(latest_id):
                    client.expire(key, SHORT_TERM_MEMORY_TTL_SECONDS)
                    return cached
        except RedisError as exc:
            logger.warning("读取短期记忆缓存失败，回源PostgreSQL: %s", exc)

    messages = _load_from_postgres(db, user_id, conversation_id, limit)
    _replace_cache(user_id, conversation_id, messages)
    return messages


def delete_short_term_memory(user_id: int, conversation_id: int) -> None:
    """删除对话后清理热缓存；Redis失败不影响PostgreSQL删除结果"""
    client = get_redis()
    if client is None:
        return
    try:
        client.delete(short_term_key(user_id, conversation_id))
    except RedisError as exc:
        logger.warning("删除短期记忆缓存失败，等待TTL自动回收: %s", exc)


__all__ = [
    "append_short_term_message",
    "delete_short_term_memory",
    "load_short_term_messages",
    "short_term_key",
]
