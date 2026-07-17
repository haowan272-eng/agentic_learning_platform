"""RAG主链路产生的私有对话只读/删除API"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.api.deps import get_current_user
from ..models import RagConversation, RagMessage, User
from ..schemas.conversation import ConversationResponse, MessageResponse

router = APIRouter(prefix="/conversations", tags=["私有对话"])


def _user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存")
    return user


def _conversation(db: Session, conversation_id: int, user_id: int) -> RagConversation:
    row = db.query(RagConversation).filter(
        RagConversation.id == conversation_id,
        RagConversation.user_id == user_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="对话不存")
    return row


def _conversation_response(row: RagConversation) -> ConversationResponse:
    return ConversationResponse(
        id=row.id,
        title=row.title,
        kb_id=row.kb_id,
        created_at=str(row.created_at),
        updated_at=str(row.updated_at),
    )


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _user(db, current_user)
    rows = db.query(RagConversation).filter(RagConversation.user_id == user.id).order_by(
        RagConversation.updated_at.desc()
    ).all()
    return [_conversation_response(row) for row in rows]


@router.get("/{conversation_id}/messages", response_model=list[MessageResponse])
def list_messages(
    conversation_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _user(db, current_user)
    _conversation(db, conversation_id, user.id)
    rows = db.query(RagMessage).filter(RagMessage.conversation_id == conversation_id).order_by(
        RagMessage.created_at, RagMessage.id
    ).all()
    return [
        MessageResponse(
            id=row.id,
            role=row.role,
            content=row.content,
            citations_json=row.citations_json,
            memory_json=row.memory_json,
            degraded=bool(row.degraded),
            created_at=str(row.created_at),
        )
        for row in rows
    ]


@router.delete("/{conversation_id}")
def delete_conversation(
    conversation_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user = _user(db, current_user)
    conversation = _conversation(db, conversation_id, user.id)
    db.query(RagMessage).filter(RagMessage.conversation_id == conversation.id).delete()
    db.delete(conversation)
    db.commit()
    from app.services.short_term_memory import delete_short_term_memory

    delete_short_term_memory(user.id, conversation_id)
    return {"ok": True}
