"""知识库成员 RBAC 模型。"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.sql import func

from app.core.database import Base


class KnowledgeBaseMember(Base):
    __tablename__ = "knowledge_base_members"

    id = Column(Integer, primary_key=True, autoincrement=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False, default="viewer", comment="owner / admin / editor / viewer")

    created_at = Column(DateTime, server_default=func.now())

    __table_args__ = (
        UniqueConstraint("kb_id", "user_id", name="uq_kb_member"),
    )
