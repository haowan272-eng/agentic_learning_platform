"""知识库 CRUD 与成员管理 API。"""
import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.database import get_db
from ..models import User, KnowledgeBase, KnowledgeBaseMember
from ..schemas.knowledge_base import (
    KnowledgeBaseCreate,
    KnowledgeBaseUpdate,
    KnowledgeBaseResponse,
    AddMemberRequest,
    UpdateMemberRequest,
    MemberResponse,
)
from app.api.deps import get_current_user, check_kb_role

router = APIRouter(prefix="/kb", tags=["知识"])


def _get_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存")
    return user


# ==================== 知识库 CRUD ====================


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(
    body: KnowledgeBaseCreate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """创建知识库，创建者自动成为 owner。"""
    user = _get_user(db, current_user)

    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        chunk_config=body.chunk_config,
        created_by=user.id,
    )
    db.add(kb)
    db.flush()

    db.add(KnowledgeBaseMember(kb_id=kb.id, user_id=user.id, role="owner"))
    db.commit()
    db.refresh(kb)

    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        chunk_config=kb.chunk_config,
        created_by=kb.created_by,
        created_at=str(kb.created_at),
        updated_at=str(kb.updated_at),
        member_count=1,
        role="owner",
    )


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_kbs(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """列出共享知识库；成员角色仅用于管理权限"""
    user = _get_user(db, current_user)

    memberships = {
        member.kb_id: member.role
        for member in (
        db.query(KnowledgeBaseMember)
        .filter(KnowledgeBaseMember.user_id == user.id)
        .all()
        )
    }
    member_counts = dict(
        db.query(KnowledgeBaseMember.kb_id, func.count(KnowledgeBaseMember.id))
        .group_by(KnowledgeBaseMember.kb_id)
        .all()
    )

    result = []
    if not memberships:
        return result
    for kb in (
        db.query(KnowledgeBase)
        .filter(KnowledgeBase.id.in_(list(memberships)))
        .order_by(KnowledgeBase.created_at.desc())
        .all()
    ):
        result.append(
            KnowledgeBaseResponse(
                id=kb.id,
                name=kb.name,
                description=kb.description,
                chunk_config=kb.chunk_config,
                created_by=kb.created_by,
                created_at=str(kb.created_at),
                updated_at=str(kb.updated_at),
                member_count=member_counts.get(kb.id, 0),
                role=memberships.get(kb.id),
            )
        )
    return result


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_kb(
    kb_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """获取共享知识库详情；管理操作仍要求成员角色"""
    user = _get_user(db, current_user)

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    membership = db.query(KnowledgeBaseMember).filter(
        KnowledgeBaseMember.kb_id == kb_id,
        KnowledgeBaseMember.user_id == user.id,
    ).first()
    if not membership:
        raise HTTPException(status_code=403, detail="您不是该知识库的成员")

    member_count = (
        db.query(KnowledgeBaseMember)
        .filter(KnowledgeBaseMember.kb_id == kb.id)
        .count()
    )

    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        chunk_config=kb.chunk_config,
        created_by=kb.created_by,
        created_at=str(kb.created_at),
        updated_at=str(kb.updated_at),
        member_count=member_count,
        role=membership.role if membership else None,
    )


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_kb(
    kb_id: int,
    body: KnowledgeBaseUpdate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """更新知识库（需 admin+ 权限"""
    check_kb_role(db, current_user, kb_id, "admin")

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    if body.chunk_config is not None:
        kb.chunk_config = body.chunk_config
    db.commit()
    db.refresh(kb)

    member_count = (
        db.query(KnowledgeBaseMember)
        .filter(KnowledgeBaseMember.kb_id == kb.id)
        .count()
    )

    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        chunk_config=kb.chunk_config,
        created_by=kb.created_by,
        created_at=str(kb.created_at),
        updated_at=str(kb.updated_at),
        member_count=member_count,
        role="admin",
    )


@router.delete("/{kb_id}")
def delete_kb(
    kb_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """删除知识库（需 owner 权限）。文档保留，kb_id 置空"""
    check_kb_role(db, current_user, kb_id, "owner")

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="知识库不存在")

    # 将该 KB 下的文档 kb_id 置空，不删除文档。
    from ..models import Document
    db.query(Document).filter(Document.kb_id == kb_id).update({"kb_id": None})

    db.delete(kb)
    db.commit()

    return {"ok": True, "detail": f"知识库 '{kb.name}' 已删除，文档已保留"}


# ==================== 成员管理 ====================


@router.get("/{kb_id}/members", response_model=list[MemberResponse])
def list_members(
    kb_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """列出知识库所有成员（需 viewer+ 权限"""
    check_kb_role(db, current_user, kb_id, "viewer")

    members = (
        db.query(KnowledgeBaseMember, User.username)
        .join(User, KnowledgeBaseMember.user_id == User.id)
        .filter(KnowledgeBaseMember.kb_id == kb_id)
        .all()
    )

    return [
        MemberResponse(
            user_id=m.KnowledgeBaseMember.user_id,
            username=m.username,
            role=m.KnowledgeBaseMember.role,
            created_at=str(m.KnowledgeBaseMember.created_at),
        )
        for m in members
    ]


@router.post("/{kb_id}/members", response_model=MemberResponse)
def add_member(
    kb_id: int,
    body: AddMemberRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """添加成员到知识库（需 admin+ 权限"""
    check_kb_role(db, current_user, kb_id, "admin")

    target_user = db.query(User).filter(User.username == body.username).first()
    if not target_user:
        raise HTTPException(status_code=404, detail=f"用户 '{body.username}' 不存")

    existing = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == target_user.id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="该用户已经是知识库成")

    member = KnowledgeBaseMember(kb_id=kb_id, user_id=target_user.id, role=body.role)
    db.add(member)
    db.commit()
    db.refresh(member)

    return MemberResponse(
        user_id=member.user_id,
        username=body.username,
        role=member.role,
        created_at=str(member.created_at),
    )


@router.put("/{kb_id}/members/{user_id}", response_model=MemberResponse)
def update_member_role(
    kb_id: int,
    user_id: int,
    body: UpdateMemberRequest,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """修改成员角色（需 admin+ 权限，不能修改 owner）。"""
    check_kb_role(db, current_user, kb_id, "admin")

    member = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == user_id,
        )
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="该用户不是知识库成员")

    if member.role == "owner":
        raise HTTPException(status_code=403, detail="不能修改 owner 的角")

    member.role = body.role
    db.commit()

    target_user = db.query(User).filter(User.id == user_id).first()

    return MemberResponse(
        user_id=member.user_id,
        username=target_user.username if target_user else "unknown",
        role=member.role,
        created_at=str(member.created_at),
    )


@router.delete("/{kb_id}/members/{user_id}")
def remove_member(
    kb_id: int,
    user_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """移除成员（需 admin+ 权限，不能移除 owner）。"""
    check_kb_role(db, current_user, kb_id, "admin")

    member = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == user_id,
        )
        .first()
    )
    if not member:
        raise HTTPException(status_code=404, detail="该用户不是知识库成员")

    if member.role == "owner":
        raise HTTPException(status_code=403, detail="不能移除 owner")

    db.delete(member)
    db.commit()

    return {"ok": True, "detail": f"已移除用户 {user_id}"}
