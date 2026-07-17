"""用户长期记忆：只由 RAG 回答主链路产生和消费。"""
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func

from app.core.database import Base


class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "keyword", "category", name="uq_user_memory_keyword"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    memory_type = Column(String(64), nullable=False, default="profile", index=True)
    key = Column(String(128), nullable=True, index=True)
    keyword = Column(String(128), nullable=False, index=True)
    memory_key = Column(String(128), nullable=True, index=True)
    category = Column(String(32), nullable=False, default="other", comment="destination / preference / budget / constraint / other")
    value = Column(Text, nullable=True)
    weight = Column(Float, default=1.0, comment="重复提取时递增权重")
    confidence = Column(Float, nullable=True)
    source = Column(String(64), nullable=True, default="agent", index=True)
    is_active = Column(Integer, nullable=True, default=1, index=True)
    source_conversation_id = Column(Integer, ForeignKey("rag_conversations.id", ondelete="SET NULL"), nullable=True)
    source_event_id = Column(Integer, ForeignKey("memory_events.id", ondelete="SET NULL"), nullable=True, index=True)
    source_task_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
