"""文档检索子表 ORM 模型。"""
from sqlalchemy import Column, ForeignKey, Index, Integer, String, Text

from app.core.database import Base


class DocumentChunk(Base):
    __tablename__ = "documents_chunks"
    __table_args__ = (
        Index("ix_documents_chunks_document_chunk_index", "document_id", "chunk_index"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    # Text sent to embedding; display content stays readable.
    embedding_content = Column(Text, nullable=True)
    chunk_index = Column(Integer, nullable=False)
    modality = Column(String(16), nullable=False, default="text", server_default="text")
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    # JSON string for heading path and source location.
    metadata_json = Column(Text, nullable=True)
