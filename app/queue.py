"""Celery文档任务投递，以及Redis进度、锁和最终失败记录"""
from __future__ import annotations

import json
import redis
from redis import ConnectionPool

from app.core.config import (
    CELERY_BROKER_URL,
    CELERY_DOCUMENT_QUEUE,
    DOC_INDEX_LOCK_TTL_SECONDS,
    REDIS_URL,
)

DOC_INDEX_DEAD_QUEUE = "queue:doc_index:dead"

_pool: ConnectionPool | None = None
_broker_pool: ConnectionPool | None = None


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool.from_url(
            REDIS_URL,
            max_connections=10,
            decode_responses=True,
        )
    return _pool


def _get_client() -> redis.Redis | None:
    try:
        return redis.Redis(connection_pool=_get_pool())
    except Exception:
        return None


def _get_broker_client() -> redis.Redis | None:
    """队列指标必须读取Broker所在的Redis DB，而不是应用缓存DB"""
    global _broker_pool
    try:
        if _broker_pool is None:
            _broker_pool = ConnectionPool.from_url(
                CELERY_BROKER_URL,
                max_connections=5,
                decode_responses=True,
            )
        return redis.Redis(connection_pool=_broker_pool)
    except Exception:
        return None


def enqueue_document_index_task(
    document_id: int,
    user_id: int,
    kb_id: int | None = None,
) -> str:
    """通过Celery投递文档索引任务，并返回Celery task_id"""
    from app.tasks.document_index import index_document_task

    result = index_document_task.apply_async(
        kwargs={
            "document_id": document_id,
            "user_id": user_id,
            "kb_id": kb_id,
        },
        queue=CELERY_DOCUMENT_QUEUE,
    )
    task_id = str(result.id)
    store_doc_index_progress(
        document_id,
        "queued",
        task_id=task_id,
        attempt=0,
    )
    return task_id


def acquire_doc_index_lock(document_id: int, task_id: str) -> bool | None:
    client = _get_client()
    if client is None:
        return None
    return bool(
        client.set(
            f"lock:doc_index:{document_id}",
            task_id,
            nx=True,
            ex=DOC_INDEX_LOCK_TTL_SECONDS,
        )
    )


def release_doc_index_lock(document_id: int, task_id: str) -> None:
    """只释放当前Celery任务持有的文档锁"""
    client = _get_client()
    if client is None:
        return
    script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
      return redis.call('del', KEYS[1])
    end
    return 0
    """
    client.eval(script, 1, f"lock:doc_index:{document_id}", task_id)


def refresh_doc_index_lock(document_id: int, task_id: str) -> bool:
    """只为当前Celery任务持有的锁续期"""
    client = _get_client()
    if client is None:
        return False
    script = """
    if redis.call('get', KEYS[1]) == ARGV[1] then
        redis.call('expire', KEYS[1], ARGV[2])
        return 1
    end
    return 0
    """
    return bool(
        client.eval(
            script,
            1,
            f"lock:doc_index:{document_id}",
            task_id,
            DOC_INDEX_LOCK_TTL_SECONDS,
        )
    )


def store_doc_index_progress(
    document_id: int,
    status: str,
    error_message: str | None = None,
    **details,
) -> None:
    client = _get_client()
    if client is None:
        return
    payload = {"status": status, **details}
    if error_message:
        payload["error_message"] = error_message[:1000]
    try:
        client.setex(
            f"doc_progress:{document_id}",
            1800,
            json.dumps(payload, ensure_ascii=False),
        )
    except redis.RedisError:
        # Progress is side-channel information; do not fail the task on Redis errors.
        return


def get_doc_index_progress(document_id: int) -> dict:
    client = _get_client()
    if client is None:
        return {"status": "unknown"}
    try:
        raw = client.get(f"doc_progress:{document_id}")
        return json.loads(raw) if raw else {"status": "not_found"}
    except (redis.RedisError, json.JSONDecodeError):
        return {"status": "unknown"}


def get_doc_index_queue_length() -> int:
    """Redis Broker中指定Celery队列的待消费任务数"""
    client = _get_broker_client()
    if client is None:
        return 0
    try:
        return int(client.llen(CELERY_DOCUMENT_QUEUE))
    except redis.RedisError:
        return 0


def record_dead_document_task(payload: dict) -> None:
    """Celery重试耗尽后保留一份便于管理端检查的失败记录"""
    client = _get_client()
    if client is not None:
        try:
            client.lpush(
                DOC_INDEX_DEAD_QUEUE,
                json.dumps(payload, ensure_ascii=False),
            )
        except redis.RedisError:
            return


__all__ = [
    "acquire_doc_index_lock",
    "enqueue_document_index_task",
    "get_doc_index_progress",
    "get_doc_index_queue_length",
    "record_dead_document_task",
    "refresh_doc_index_lock",
    "release_doc_index_lock",
    "store_doc_index_progress",
]
