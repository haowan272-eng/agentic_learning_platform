"""FastAPI 共享依赖：JWT 鉴权、用户解析、知识库 RBAC 权限校验。

is_platform_admin     —— 判断是否为平台管理员
get_current_user      —— Bearer token → 用户名
check_kb_role         —— 知识库成员身份 + 角色等级校验
"""
from types import SimpleNamespace

import jwt
from fastapi import Depends, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.config import ADMIN_USERNAMES, ALGORITHM, SECRET_KEY
from app.core.errors import AppError, ErrorCode

security = HTTPBearer()

# 角色层级：数值越大权限越高
ROLE_HIERARCHY = {"viewer": 1, "editor": 2, "admin": 3, "owner": 4}


def is_platform_admin(username: str) -> bool:
    """判断用户是否为平台管理员。"""
    return username in ADMIN_USERNAMES


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """从 Bearer token 中解析当前用户名。

    验证 JWT 签名和有效期，失败时返回 401。
    成功时返回 payload 中的 sub 字段（即用户名）。
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise AppError(
            code=ErrorCode.AUTH_TOKEN_EXPIRED,
            message="Token expired",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
    except jwt.InvalidTokenError:
        raise AppError(
            code=ErrorCode.AUTH_TOKEN_INVALID,
            message="Invalid token",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


def check_kb_role(db: Session, current_user: str, kb_id: int, min_role: str):
    """校验用户对指定知识库的访问权限。

    参数：
        db:           数据库会话
        current_user: 当前用户名
        kb_id:        目标知识库 ID
        min_role:     所需最低角色（viewer/editor/admin/owner）

    返回：
        用户在该知识库的成员记录或合成成员对象

    逻辑：
        - shared 知识库对所有已认证用户开放 viewer 权限
        - 平台管理员对 shared 知识库自动拥有 owner 权限
        - 角色不足时抛出 403
    """
    from app.models import KnowledgeBase, KnowledgeBaseMember, User

    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise AppError(
            code=ErrorCode.AUTH_USER_NOT_FOUND,
            message="User not found",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise AppError(
            code=ErrorCode.KB_NOT_FOUND,
            message="Knowledge base not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )

    required_level = ROLE_HIERARCHY.get(min_role, 0)
    is_shared = getattr(kb, "visibility", "private") == "shared"

    membership = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == user.id,
        )
        .first()
    )

    if is_shared and required_level <= ROLE_HIERARCHY["viewer"]:
        return membership or SimpleNamespace(kb_id=kb_id, user_id=user.id, role="viewer")

    if not membership:
        if is_shared and is_platform_admin(current_user):
            return SimpleNamespace(kb_id=kb_id, user_id=user.id, role="owner")
        raise AppError(
            code=ErrorCode.KB_FORBIDDEN,
            message="You are not a member of this knowledge base",
            status_code=status.HTTP_403_FORBIDDEN,
        )

    user_level = ROLE_HIERARCHY.get(membership.role, 0)
    if is_shared and is_platform_admin(current_user):
        user_level = ROLE_HIERARCHY["owner"]
    if user_level < required_level:
        raise AppError(
            code=ErrorCode.KB_ROLE_REQUIRED,
            message=f"Requires {min_role} or higher permission",
            status_code=status.HTTP_403_FORBIDDEN,
            details={"required_role": min_role, "current_role": membership.role},
        )

    return membership
