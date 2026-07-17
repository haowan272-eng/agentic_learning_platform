"""固定窗口频率限制：基于 Redis INCR + TTL 实现。"""
from app.core.redis import get_redis
from app.core.config import (
    RATE_LIMIT_LOGIN_PER_MIN,
    RATE_LIMIT_ENABLED,
)


class RateLimiter:
    """固定窗口计数器。key 格式: ratelimit:{prefix}:{identifier}"""

    def __init__(self, key_prefix: str, max_requests: int, window_seconds: int = 60):
        self.key_prefix = key_prefix
        self.max_requests = max_requests
        self.window_seconds = window_seconds

    def is_allowed(self, identifier: str) -> bool:
        """返回 True 表示允许本次请求"""
        if not RATE_LIMIT_ENABLED:
            return True
        redis = get_redis()
        if redis is None:
            return True  # Redis 不可用时放行，避免误杀
        key = f"ratelimit:{self.key_prefix}:{identifier}"
        try:
            count = redis.incr(key)
            if count == 1:
                redis.expire(key, self.window_seconds)
            return count <= self.max_requests
        except Exception:
            return True  # Redis 异常时放行

# ---- 预设限制 ----

login_limiter = RateLimiter("login", RATE_LIMIT_LOGIN_PER_MIN, 60)
