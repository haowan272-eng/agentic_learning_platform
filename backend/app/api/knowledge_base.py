"""知识库 CRUD 与成员管理 API。

提供知识库的创建、列表、详情、更新、删除，以及成员增删改查。
支持 private / shared 两种可见性，shared 知识库对所有已认证用户开放 viewer 权限。
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.api.deps import ROLE_HIERARCHY, check_kb_role, get_current_user, is_platform_admin
from app.core.database import get_db
from ..models import KnowledgeBase, KnowledgeBaseMember, User
from ..schemas.knowledge_base import (
    AddMemberRequest,
    KnowledgeBaseCreate,
    KnowledgeBaseResponse,
    KnowledgeBaseUpdate,
    MemberResponse,
    UpdateMemberRequest,
)

router = APIRouter(prefix="/kb", tags=["Knowledge Base"])


def _get_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _member_counts(db: Session) -> dict[int, int]:
    return dict(
        db.query(KnowledgeBaseMember.kb_id, func.count(KnowledgeBaseMember.id))
        .group_by(KnowledgeBaseMember.kb_id)
        .all()
    )


def _role_for_kb(
    db: Session,
    user: User,
    username: str,
    kb: KnowledgeBase,
) -> Optional[str]:
    member = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb.id,
            KnowledgeBaseMember.user_id == user.id,
        )
        .first()
    )
    if member:
        return member.role
    if kb.visibility == "shared":
        return "owner" if is_platform_admin(username) else "viewer"
    return None


def _can_upload(username: str, kb: KnowledgeBase, role: Optional[str]) -> bool:
    if kb.visibility == "shared":
        return is_platform_admin(username)
    return ROLE_HIERARCHY.get(role or "", 0) >= ROLE_HIERARCHY["editor"]


def _kb_response(
    kb: KnowledgeBase,
    *,
    role: Optional[str],
    member_count: int,
    can_upload: bool,
) -> KnowledgeBaseResponse:
    return KnowledgeBaseResponse(
        id=kb.id,
        name=kb.name,
        description=kb.description,
        visibility=kb.visibility,
        chunk_config=kb.chunk_config,
        created_by=kb.created_by,
        created_at=str(kb.created_at),
        updated_at=str(kb.updated_at),
        member_count=member_count,
        role=role,
        can_upload=can_upload,
    )


@router.post("", response_model=KnowledgeBaseResponse)
def create_kb(
    body: KnowledgeBaseCreate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _get_user(db, current_user)
    if body.visibility == "shared" and not is_platform_admin(current_user):
        raise HTTPException(status_code=403, detail="Only administrators can create shared knowledge bases")

    kb = KnowledgeBase(
        name=body.name,
        description=body.description,
        visibility=body.visibility,
        chunk_config=body.chunk_config,
        created_by=user.id,
    )
    db.add(kb)
    db.flush()
    db.add(KnowledgeBaseMember(kb_id=kb.id, user_id=user.id, role="owner"))
    db.commit()
    db.refresh(kb)

    return _kb_response(
        kb,
        role="owner",
        member_count=1,
        can_upload=_can_upload(current_user, kb, "owner"),
    )


@router.get("", response_model=list[KnowledgeBaseResponse])
def list_kbs(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _get_user(db, current_user)
    memberships = {
        member.kb_id: member.role
        for member in (
            db.query(KnowledgeBaseMember)
            .filter(KnowledgeBaseMember.user_id == user.id)
            .all()
        )
    }
    counts = _member_counts(db)
    filters = [KnowledgeBase.visibility == "shared"]
    if memberships:
        filters.append(KnowledgeBase.id.in_(list(memberships)))

    result: list[KnowledgeBaseResponse] = []
    for kb in (
        db.query(KnowledgeBase)
        .filter(or_(*filters))
        .order_by(KnowledgeBase.visibility.desc(), KnowledgeBase.created_at.desc())
        .all()
    ):
        role = memberships.get(kb.id)
        if role is None and kb.visibility == "shared":
            role = "owner" if is_platform_admin(current_user) else "viewer"
        result.append(
            _kb_response(
                kb,
                role=role,
                member_count=counts.get(kb.id, 0),
                can_upload=_can_upload(current_user, kb, role),
            )
        )
    return result


@router.get("/{kb_id}", response_model=KnowledgeBaseResponse)
def get_kb(
    kb_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _get_user(db, current_user)
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    role = _role_for_kb(db, user, current_user, kb)
    if role is None:
        raise HTTPException(status_code=403, detail="You are not a member of this knowledge base")

    return _kb_response(
        kb,
        role=role,
        member_count=(
            db.query(KnowledgeBaseMember)
            .filter(KnowledgeBaseMember.kb_id == kb.id)
            .count()
        ),
        can_upload=_can_upload(current_user, kb, role),
    )


@router.put("/{kb_id}", response_model=KnowledgeBaseResponse)
def update_kb(
    kb_id: int,
    body: KnowledgeBaseUpdate,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")
    check_kb_role(db, current_user, kb_id, "admin")

    if body.visibility is not None and body.visibility != kb.visibility:
        if not is_platform_admin(current_user):
            raise HTTPException(status_code=403, detail="Only administrators can change shared visibility")
        kb.visibility = body.visibility
    if body.name is not None:
        kb.name = body.name
    if body.description is not None:
        kb.description = body.description
    if body.chunk_config is not None:
        kb.chunk_config = body.chunk_config
    db.commit()
    db.refresh(kb)

    user = _get_user(db, current_user)
    role = _role_for_kb(db, user, current_user, kb)
    return _kb_response(
        kb,
        role=role,
        member_count=(
            db.query(KnowledgeBaseMember)
            .filter(KnowledgeBaseMember.kb_id == kb.id)
            .count()
        ),
        can_upload=_can_upload(current_user, kb, role),
    )


@router.delete("/{kb_id}")
def delete_kb(
    kb_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    check_kb_role(db, current_user, kb_id, "owner")

    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise HTTPException(status_code=404, detail="Knowledge base not found")

    from ..models import Document

    db.query(Document).filter(Document.kb_id == kb_id).update({"kb_id": None})
    db.delete(kb)
    db.commit()

    return {"ok": True, "detail": f"Knowledge base '{kb.name}' deleted; documents were kept"}


@router.get("/{kb_id}/members", response_model=list[MemberResponse])
def list_members(
    kb_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
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
    check_kb_role(db, current_user, kb_id, "admin")

    target_user = db.query(User).filter(User.username == body.username).first()
    if not target_user:
        raise HTTPException(status_code=404, detail=f"User '{body.username}' not found")

    existing = (
        db.query(KnowledgeBaseMember)
        .filter(
            KnowledgeBaseMember.kb_id == kb_id,
            KnowledgeBaseMember.user_id == target_user.id,
        )
        .first()
    )
    if existing:
        raise HTTPException(status_code=400, detail="User is already a knowledge base member")

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
        raise HTTPException(status_code=404, detail="User is not a knowledge base member")
    if member.role == "owner":
        raise HTTPException(status_code=403, detail="Owner role cannot be changed")

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
        raise HTTPException(status_code=404, detail="User is not a knowledge base member")
    if member.role == "owner":
        raise HTTPException(status_code=403, detail="Owner cannot be removed")

    db.delete(member)
    db.commit()

    return {"ok": True, "detail": f"Removed user {user_id}"}
