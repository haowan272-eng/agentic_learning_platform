"""认证相关的 Pydantic 请求/响应模型。"""
from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    """登录请求"""
    username: str = Field(..., min_length=1, max_length=64, description="用户")
    password: str = Field(..., min_length=1, max_length=128, description="密码")


class TokenResponse(BaseModel):
    """登录成功返回"""
    access_token: str = Field(..., description="访问令牌 (JWT)")
    refresh_token: str = Field(..., description="刷新令牌")
    token_type: str = Field(default="bearer", description="令牌类型")


class RefreshRequest(BaseModel):
    """刷新 token 请求"""
    refresh_token: str = Field(..., description="刷新令牌")
