"""Redis 客户端：连接池管理，生命周期由 main.py 控制。"""
import logging

import redis
from redis import ConnectionPool

from app.core.config import REDIS_URL

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None
_client: redis.Redis | None = None


def init_redis():
    """启动时调用：创建连接池和客户端，验证连通"""
    global _pool, _client
    _pool = ConnectionPool.from_url(
        REDIS_URL,
        max_connections=20,
        decode_responses=True,
    )
    _client = redis.Redis(connection_pool=_pool)
    _client.ping()  # 失败会报错，让启动流程感知。
    logger.info("redis connected")


def close_redis():
    """关闭时调用：释放连接"""
    if _client:
        _client.close()
        logger.info("redis connection closed")


def get_redis() -> redis.Redis | None:
    """获取共享 Redis 客户端；若未初始化则返回 None，调用方应优雅降级。"""
    return _client
