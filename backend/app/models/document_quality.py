"""文档质量报告持久化模型——存储自动检测的文档完备性和质量评分。"""
from sqlalchemy import Column, DateTime, ForeignKey, Integer, String, Text, func

from app.core.database import Base


class DocumentQualityReport(Base):
    __tablename__ = "document_quality_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False, index=True)
    doc_type = Column(String(64), nullable=False, index=True)
    quality_score = Column(Integer, nullable=False)
    level = Column(String(32), nullable=False, index=True)
    summary_json = Column(Text, nullable=True)
    issues_json = Column(Text, nullable=True)
    suggestions_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
