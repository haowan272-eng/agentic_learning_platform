from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from redis.exceptions import RedisError

from app.core.config import SHORT_TERM_MEMORY_TTL_SECONDS
from app.core.redis import get_redis

logger = logging.getLogger(__name__)

MAX_AGENT_RECENT_EVENTS = 40


def recent_event_key(user_id: int | str | None, session_id: str) -> str:
    return f"agent:short_term:{user_id or 'anonymous'}:{session_id}"


def _json_ready(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def append_recent_event(user_id: int | str | None, session_id: str | None, event: dict[str, Any]) -> bool:
    if not session_id:
        return False
    client = get_redis()
    if client is None:
        return False
    key = recent_event_key(user_id, session_id)
    try:
        payload = json.dumps(_json_ready(event), ensure_ascii=False)
        with client.pipeline(transaction=True) as pipe:
            pipe.rpush(key, payload)
            pipe.ltrim(key, -MAX_AGENT_RECENT_EVENTS, -1)
            pipe.expire(key, SHORT_TERM_MEMORY_TTL_SECONDS)
            pipe.execute()
        return True
    except RedisError as exc:
        logger.warning("写入 Agent 短期记忆失败，将继续使用 PostgreSQL: %s", exc)
        return False


def load_recent_events(user_id: int | str | None, session_id: str | None, *, limit: int = 20) -> list[dict[str, Any]]:
    if not session_id:
        return []
    client = get_redis()
    if client is None:
        return []
    key = recent_event_key(user_id, session_id)
    try:
        rows = client.lrange(key, -max(1, min(int(limit), MAX_AGENT_RECENT_EVENTS)), -1)
    except RedisError as exc:
        logger.warning("读取 Agent 短期记忆失败，将回源 PostgreSQL: %s", exc)
        return []

    decoded = []
    for row in rows:
        try:
            decoded.append(json.loads(row))
        except (TypeError, json.JSONDecodeError):
            return []
    return decoded
