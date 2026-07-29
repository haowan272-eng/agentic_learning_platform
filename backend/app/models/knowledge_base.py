"""知识库 ORM 模型。

知识库是文档的逻辑分组单元，支持 private/shared 两种可见性。
shared 知识库对所有已认证用户开放，private 仅成员可见。
"""
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.sql import func

from app.core.database import Base


class KnowledgeBase(Base):
    __tablename__ = "knowledge_bases"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    visibility = Column(String(20), nullable=False, default="private", index=True)
    chunk_config = Column(Text, nullable=True, comment="JSON: {chunk_size, chunk_overlap, strategy}")
    created_by = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
