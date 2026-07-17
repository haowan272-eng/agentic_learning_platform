"""用户私有RAG对话与消息"""
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, func

from app.core.database import Base


class RagConversation(Base):
    __tablename__ = "rag_conversations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(255), nullable=False, default="新对")
    summary = Column(Text, nullable=True)
    summary_until_message_id = Column(Integer, nullable=True)
    task_state_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class RagMessage(Base):
    __tablename__ = "rag_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    conversation_id = Column(
        Integer,
        ForeignKey("rag_conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    citations_json = Column(Text, nullable=True)
    memory_json = Column(Text, nullable=True)
    memory_extracted = Column(Boolean, nullable=False, default=False)
    degraded = Column(Boolean, nullable=False, default=False)
    created_at = Column(DateTime, server_default=func.now())
