"""FastAPI 共享依赖：认证、数据库会话、知识库权限"""
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.core.config import SECRET_KEY, ALGORITHM

security = HTTPBearer()

ROLE_HIERARCHY = {"viewer": 1, "editor": 2, "admin": 3, "owner": 4}


def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    从 Bearer token 解析当前用户。

    用法: current_user: str = Depends(get_current_user)
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload["sub"]
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def check_kb_role(db: Session, current_user: str, kb_id: int, min_role: str):
    """
    验证当前用户是指定知识库成员且角色 >= min_role。

    失败时抛出 HTTPException(403)，成功时返回 membership 行。
    用法: membership = check_kb_role(db, current_user, kb_id, "editor")
    """
    from app.models import User, KnowledgeBaseMember

    user = db.query(User).filter(User.username == current_user).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存")

    membership = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == user.id,
        )
        .first()
    )
    if not membership:
        raise HTTPException(status_code=403, detail="您不是该知识库的成员")

    required_level = ROLE_HIERARCHY.get(min_role, 0)
    user_level = ROLE_HIERARCHY.get(membership.role, 0)
    if user_level < required_level:
        raise HTTPException(status_code=403, detail=f"需要 {min_role} 或更高权限")

    return membership

