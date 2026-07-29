"""Embedding 向量 ORM 模型。"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base


class ChunkEmbedding(Base):
    """
    记录每个分块所用的向量模型和维度；实际向量只存储在 Qdrant。
    ``embedding`` 列已在 20260730_0018 迁移中删除。"""

    __tablename__ = "chunk_embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(
        Integer,
        ForeignKey("documents_chunks.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        comment="关联 documents_chunks.id，一个分块只有一个向量",
    )
    dim = Column(Integer, nullable=False, comment="向量维度")
    model_name = Column(String(256), nullable=False, comment="生成该向量的模型名称")
    created_at = Column(DateTime, server_default=func.now(), comment="创建时间")

    # 可选：反向关系
    chunk = relationship("DocumentChunk", backref="embedding_row", uselist=False)
