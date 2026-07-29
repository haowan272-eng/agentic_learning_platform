"""用户长期记忆：由 Agent 运行时记忆合并器产生和消费，RAG 不再写入此表。"""
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func

from app.core.database import Base


class UserMemory(Base):
    __tablename__ = "user_memories"
    __table_args__ = (
        UniqueConstraint("user_id", "keyword", "category", name="uq_user_memory_keyword"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    memory_type = Column(String(64), nullable=False, default="agent_profile", index=True)
    key = Column(String(128), nullable=True, index=True, comment="DEPRECATED: RAG legacy keyword column.")
    keyword = Column(String(128), nullable=True, index=True, comment="DEPRECATED: RAG legacy keyword column.")
    memory_key = Column(String(128), nullable=True, index=True)
    category = Column(String(32), nullable=False, default="other", comment="Agent categories: learning_goal, weak_point, mastered_topic, project_gap, preference, constraint")
    value = Column(Text, nullable=True)
    weight = Column(Float, default=1.0, comment="Agent 合并时递增权重")
    confidence = Column(Float, nullable=True)
    source = Column(String(64), nullable=True, default="agent", index=True)
    is_active = Column(Integer, nullable=True, default=1, index=True)
    source_conversation_id = Column(Integer, ForeignKey("rag_conversations.id", ondelete="SET NULL"), nullable=True, comment="DEPRECATED: RAG memory system removed; no longer populated.")
    source_event_id = Column(Integer, ForeignKey("memory_events.id", ondelete="SET NULL"), nullable=True, index=True)
    source_task_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
