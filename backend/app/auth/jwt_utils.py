"""JWT token 生成与验证工具。

提供短期 access token 和长期 refresh token 的生成函数。
access token 用于 API 鉴权，refresh token 用于无感续期。
"""
from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import (
    SECRET_KEY,
    ALGORITHM,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_SECRET_KEY,
    REFRESH_TOKEN_EXPIRE_MINUTES,
)


def create_access_token(username: str) -> str:
    """生成短期 access token。

    以用户名作为 subject，有效期由 ACCESS_TOKEN_EXPIRE_MINUTES 控制。
    用于每次 API 请求的 Bearer 鉴权。
    """
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def create_refresh_token(username: str) -> str:
    """生成长期 refresh token。

    使用独立的 REFRESH_SECRET_KEY 签名，有效期由 REFRESH_TOKEN_EXPIRE_MINUTES 控制。
    客户端在 access token 过期后可使用此 token 换取新的 token 对。
    """
    payload = {
        "sub": username,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES),
    }
    return jwt.encode(payload, REFRESH_SECRET_KEY, algorithm=ALGORITHM)
