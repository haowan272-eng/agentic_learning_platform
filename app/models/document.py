"""文档生命周期元数据 ORM 模型。

PostgreSQL 不存储上传文件正文，也不存储用户原始本地路径。
- source_retained=True: 原始文件体仍保留在受控存储中。
- source_retained=False: 原始文件体在解析/索引后已删除，元数据、结构化字段和 Qdrant 向量仍可用。
- storage_key: 系统生成的存储引用，非用户原始文件路径。
"""
from sqlalchemy import Column, Integer, String, DateTime, Text, Boolean, ForeignKey, func
from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # user_id records the uploader; document corpus is shared among logged-in users.
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    kb_id = Column(Integer, nullable=True, index=True, comment="所属知识库，NULL=未归")

    # ---- 原始文件信息 ----
    original_file_name = Column(String, nullable=True)
    content_type = Column(String, nullable=True)
    file_size = Column(Integer, nullable=True)

    # ---- 旧字段（保留兼容） ----
    file_name = Column(String, nullable=True)
    file_path = Column(String, nullable=True)

    # ---- 存储管理 ----
    source_retained = Column(Boolean, default=False)
    storage_backend = Column(String, nullable=True)
    storage_key = Column(String, nullable=True)

    # ---- 生命周期状态 ----
    status = Column(String, default="uploaded", index=True)  # uploaded / indexing / indexed / failed
    error_message = Column(Text, nullable=True)
    pipeline_version = Column(String, nullable=True, comment="上一次索引使用的分块和模型配置指")

    # ---- 时间戳 ----
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
