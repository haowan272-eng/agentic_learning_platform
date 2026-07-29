"""认证路由：登录、注册、刷新 token。"""
import json

import bcrypt
import jwt
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.errors import AppError, ErrorCode
from ..models import User
from ..schemas import LoginRequest, TokenResponse, RefreshRequest
from ..auth import create_access_token, create_refresh_token
from app.core.config import (
    REFRESH_SECRET_KEY, ALGORITHM,
    ADMIN_USERNAMES,
    AUTO_CREATE_DEFAULT_KB, DEFAULT_KB_NAME,
    DEFAULT_KB_CHUNK_SIZE, DEFAULT_KB_CHUNK_OVERLAP,
)
from ..cache import get_cached_user, set_cached_user
from ..rate_limiter import login_limiter

router = APIRouter(tags=["认证"])


def _ensure_default_kb(db: Session, user: User) -> None:
    """首次登录/注册时自动创建默认知识库（幂等）"""
    if not AUTO_CREATE_DEFAULT_KB:
        return
    from ..models.knowledge_base import KnowledgeBase
    from ..models.knowledge_base_member import KnowledgeBaseMember

    existing = (
        db.query(KnowledgeBaseMember)
        .join(KnowledgeBase, KnowledgeBaseMember.kb_id == KnowledgeBase.id)
        .filter(
            KnowledgeBaseMember.user_id == user.id,
            KnowledgeBase.name == DEFAULT_KB_NAME,
        )
        .first()
    )
    if existing:
        return

    default_config = json.dumps({
        "chunk_size": DEFAULT_KB_CHUNK_SIZE,
        "chunk_overlap": DEFAULT_KB_CHUNK_OVERLAP,
    })
    kb = KnowledgeBase(
        name=DEFAULT_KB_NAME,
        description="自动创建的默认知识库",
        chunk_config=default_config,
        created_by=user.id,
    )
    db.add(kb)
    db.flush()
    db.add(KnowledgeBaseMember(kb_id=kb.id, user_id=user.id, role="owner"))
    db.commit()


def _verify_password(plain: str, hashed: str) -> bool:
    """兼容 bcrypt 哈希和旧的明文密码"""
    if hashed.startswith("$2b$") or hashed.startswith("$2a$"):
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    return plain == hashed


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    """登录：验证用户名密码，返回 access_token 和 refresh_token。"""
    if not login_limiter.is_allowed(req.username):
        raise AppError(
            code=ErrorCode.AUTH_LOGIN_RATE_LIMITED,
            message="登录过于频繁，请稍后再试",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            retryable=True,
        )

    cached = get_cached_user(req.username)
    if cached:
        # 缓存命中说明用户存在，但仍需从DB取密码验证（密码不缓存）
        user = db.query(User).filter(User.username == req.username).first()
        if not user:
            raise AppError(code=ErrorCode.AUTH_USER_NOT_FOUND, message="用户不存在，请注册", status_code=status.HTTP_404_NOT_FOUND)
        if not _verify_password(req.password, user.password):
            raise AppError(code=ErrorCode.AUTH_PASSWORD_INVALID, message="密码错误", status_code=status.HTTP_401_UNAUTHORIZED)
    else:
        user = db.query(User).filter(User.username == req.username).first()
        if not user:
            raise AppError(code=ErrorCode.AUTH_USER_NOT_FOUND, message="用户不存在，请注册", status_code=status.HTTP_404_NOT_FOUND)
        if not _verify_password(req.password, user.password):
            raise AppError(code=ErrorCode.AUTH_PASSWORD_INVALID, message="密码错误", status_code=status.HTTP_401_UNAUTHORIZED)
        set_cached_user(user)

    _ensure_default_kb(db, user)

    return TokenResponse(
        access_token=create_access_token(req.username),
        refresh_token=create_refresh_token(req.username),
    )


@router.post("/register")
def register(req: LoginRequest, db: Session = Depends(get_db)):
    """注册：创建新用户，使用 bcrypt 哈希密码。"""
    if db.query(User).filter(User.username == req.username).first():
        raise AppError(
            code=ErrorCode.AUTH_USERNAME_EXISTS,
            message="用户名已存在",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    if req.username in ADMIN_USERNAMES:
        raise AppError(
            code=ErrorCode.AUTH_ADMIN_USERNAME_RESERVED,
            message="Admin usernames are reserved; configure an existing user or CREATE_SEED_ADMIN to create one",
            status_code=status.HTTP_403_FORBIDDEN,
        )
    hashed = bcrypt.hashpw(req.password.encode(), bcrypt.gensalt()).decode()
    user = User(username=req.username, password=hashed)
    db.add(user)
    db.commit()
    set_cached_user(user)
    _ensure_default_kb(db, user)
    return {"message": "注册成功", "username": req.username}


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest):
    """刷新 token：使用 refresh_token 换取新的 access_token。"""
    try:
        payload = jwt.decode(req.refresh_token, REFRESH_SECRET_KEY, algorithms=[ALGORITHM])
        username = payload["sub"]
        return TokenResponse(
            access_token=create_access_token(username),
            refresh_token=create_refresh_token(username),
        )
    except jwt.ExpiredSignatureError:
        raise AppError(
            code=ErrorCode.AUTH_TOKEN_EXPIRED,
            message="Refresh token expired",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    except jwt.InvalidTokenError:
        raise AppError(
            code=ErrorCode.AUTH_TOKEN_INVALID,
            message="Invalid refresh token",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
