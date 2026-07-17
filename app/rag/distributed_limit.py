"""基于Redis租约的跨Worker视觉模型全局并发限制"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager

import redis

from app.core.config import (
    REDIS_URL,
    VISION_GLOBAL_ACQUIRE_TIMEOUT_SECONDS,
    VISION_GLOBAL_CONCURRENCY,
    VISION_GLOBAL_SLOT_TTL_SECONDS,
)

logger = logging.getLogger(__name__)

_VISION_SLOTS_KEY = "semaphore:rag:vision"
_ACQUIRE_SCRIPT = """
local now_parts = redis.call('TIME')
local now_ms = now_parts[1] * 1000 + math.floor(now_parts[2] / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now_ms)
if redis.call('ZCARD', KEYS[1]) < tonumber(ARGV[2]) then
    redis.call('ZADD', KEYS[1], now_ms + tonumber(ARGV[3]), ARGV[1])
    redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[3]) + 1000)
    return 1
end
return 0
"""
_RELEASE_SCRIPT = "return redis.call('ZREM', KEYS[1], ARGV[1])"

_client: redis.Redis | None = None
_client_lock = threading.Lock()


def _get_client() -> redis.Redis:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = redis.Redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=3,
                    socket_timeout=5,
                    health_check_interval=30,
                )
    return _client


def _try_acquire(token: str) -> bool:
    return bool(
        _get_client().eval(
            _ACQUIRE_SCRIPT,
            1,
            _VISION_SLOTS_KEY,
            token,
            VISION_GLOBAL_CONCURRENCY,
            int(VISION_GLOBAL_SLOT_TTL_SECONDS * 1000),
        )
    )


def _release(token: str) -> None:
    _get_client().eval(_RELEASE_SCRIPT, 1, _VISION_SLOTS_KEY, token)


@asynccontextmanager
async def vision_global_slot():
    """等待一个跨进程VL配额；租约到期可自动回收崩溃Worker占用的槽位"""
    token = uuid.uuid4().hex
    deadline = time.monotonic() + VISION_GLOBAL_ACQUIRE_TIMEOUT_SECONDS
    acquired = False
    try:
        while time.monotonic() < deadline:
            try:
                acquired = await asyncio.to_thread(_try_acquire, token)
            except redis.RedisError as exc:
                raise RuntimeError(f"Redis 全局视觉限流不可用: {exc}") from exc
            if acquired:
                break
            await asyncio.sleep(0.1)
        if not acquired:
            raise TimeoutError(
                f"等待视觉模型全局并发槽位超过{VISION_GLOBAL_ACQUIRE_TIMEOUT_SECONDS:g}秒"
            )
        yield
    finally:
        if acquired:
            try:
                await asyncio.to_thread(_release, token)
            except redis.RedisError as exc:
                # Lease will expire automatically; release failure must not hide the original result.
                logger.warning("释放视觉模型全局槽位失败: %s", exc)


__all__ = ["vision_global_slot"]
