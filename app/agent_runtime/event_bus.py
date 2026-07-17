"""Best-effort Redis Stream fan-out; PostgreSQL remains the replay source."""
from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any

from app.core.redis import get_redis


def _ready(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_ready(item) for item in value]
    return value


def stream_key(task_id: str) -> str:
    return f"agent:events:{task_id}"


def publish_task_event(event: dict[str, Any]) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        key = stream_key(str(event["task_id"]))
        client.xadd(key, {"payload": json.dumps(_ready(event), ensure_ascii=False)}, maxlen=1000, approximate=True)
        client.expire(key, 86400)
    except Exception:  # noqa: BLE001
        return


def read_task_events(task_id: str, cursor: str, *, block_ms: int = 1000) -> tuple[str, list[dict[str, Any]]]:
    client = get_redis()
    if client is None:
        return cursor, []
    try:
        rows = client.xread({stream_key(task_id): cursor}, count=100, block=max(1, block_ms))
    except Exception:  # noqa: BLE001
        return cursor, []
    events: list[dict[str, Any]] = []
    latest = cursor
    for _, messages in rows:
        for message_id, values in messages:
            latest = message_id
            try:
                events.append(json.loads(values["payload"]))
            except (KeyError, TypeError, json.JSONDecodeError):
                continue
    return latest, events
