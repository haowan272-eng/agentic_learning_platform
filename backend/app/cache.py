"""缓存辅助函数：用户缓存。"""
import logging

from app.core.redis import get_redis
from app.core.config import CACHE_USER_TTL

logger = logging.getLogger(__name__)


def get_cached_user(username: str) -> dict | None:
    """从 Redis Hash 读取缓存的用户信息；失败返回 None。"""
    redis = get_redis()
    if redis is None:
        return None
    try:
        data = redis.hgetall(f"user:{username}")
        return data if data else None
    except Exception as e:
        logger.warning(f"Redis get_cached_user failed: {e}")
        return None


def set_cached_user(user):
    """将用户基本信息写入 Redis Hash，不缓存密码，避免安全风险。"""
    redis = get_redis()
    if redis is None:
        return
    try:
        key = f"user:{user.username}"
        redis.hset(key, mapping={
            "id": str(user.id),
            "username": user.username,
        })
        redis.expire(key, CACHE_USER_TTL)
    except Exception as e:
        logger.warning(f"Redis set_cached_user failed: {e}")
