"""用户长期关键词记忆 API。"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from ..models import User, UserMemory

router = APIRouter(prefix="/memory", tags=["用户记忆"])


def _user(db: Session, username: str) -> User:
    from fastapi import HTTPException
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存")
    return user


@router.get("")
def get_memories(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
    limit: int = 30,
):
    """获取当前用户的关键词记忆（按权重降序），登录后前端拉取并注入 RAG"""
    user = _user(db, current_user)
    rows = (
        db.query(UserMemory)
        .filter(UserMemory.user_id == user.id)
        .order_by(UserMemory.weight.desc(), UserMemory.created_at.desc())
        .limit(limit)
        .all()
    )
    return {
        "keywords": [
            {
                "id": r.id,
                "keyword": r.keyword,
                "category": r.category,
                "weight": r.weight,
            }
            for r in rows
        ],
        "count": len(rows),
    }


@router.delete("/{memory_id}")
def delete_memory(
    memory_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除当前用户的一条长期记忆"""
    user = _user(db, current_user)
    row = db.query(UserMemory).filter(
        UserMemory.id == memory_id,
        UserMemory.user_id == user.id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="记忆不存")
    db.delete(row)
    db.commit()
    return {"message": "记忆已删除"}


@router.delete("")
def clear_memories(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """清空当前用户的全部长期记忆，不影响私有对话历史"""
    user = _user(db, current_user)
    deleted = db.query(UserMemory).filter(UserMemory.user_id == user.id).delete()
    db.commit()
    return {"message": "记忆已清", "deleted": deleted}
