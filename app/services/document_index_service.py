"""文档索引业务服务；由Celery任务调用，不包含Broker重试策略"""
from __future__ import annotations

import asyncio
import json
import os
import threading

from app.core.config import (
    DOC_INDEX_LOCK_HEARTBEAT_SECONDS,
    RAG_DOCUMENT_TIMEOUT_SECONDS,
)
from app.core.database import SessionLocal
from app.models import Document
from app.queue import (
    acquire_doc_index_lock,
    refresh_doc_index_lock,
    release_doc_index_lock,
    store_doc_index_progress,
)
from app.rag.chunker import CHUNK_TOKENS, OVERLAP_TOKENS


class DocumentIndexNotFound(RuntimeError):
    pass


class DocumentIndexLockBusy(RuntimeError):
    pass


def _run_document_chunking(
    document_id: int,
    db,
    chunk_size: int,
    overlap: int,
):
    """Celery同步任务与异步文档管道之间唯一的事件循环边界"""
    from app.rag.chunker import chunk_document_async

    async def run():
        try:
            return await asyncio.wait_for(
                chunk_document_async(document_id, db, chunk_size, overlap),
                timeout=RAG_DOCUMENT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as exc:
            raise TimeoutError(
                f"文档解析超过{RAG_DOCUMENT_TIMEOUT_SECONDS:g}秒"
            ) from exc

    return asyncio.run(run())


def run_document_index(
    document_id: int,
    user_id: int,
    task_id: str,
) -> dict:
    """执行一次幂等索引尝试；异常交给Celery决定重试或最终失败"""
    lock_acquired = False
    heartbeat_stop = threading.Event()
    lock_lost = threading.Event()
    heartbeat_thread = None
    db = SessionLocal()
    try:
        lock_status = acquire_doc_index_lock(document_id, task_id)
        if lock_status is None:
            raise RuntimeError("Redis unavailable while acquiring document lock")
        lock_acquired = lock_status
        if not lock_acquired:
            raise DocumentIndexLockBusy(
                f"Document {document_id} is locked by another indexing task"
            )

        def heartbeat():
            while not heartbeat_stop.wait(DOC_INDEX_LOCK_HEARTBEAT_SECONDS):
                try:
                    if not refresh_doc_index_lock(document_id, task_id):
                        lock_lost.set()
                        return
                except Exception:
                    lock_lost.set()
                    return

        def ensure_lock():
            if lock_lost.is_set():
                raise RuntimeError("文档索引锁已丢失，停止任务以避免重复写入")

        heartbeat_thread = threading.Thread(
            target=heartbeat,
            name=f"doc-lock-{document_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        document = db.query(Document).filter(
            Document.id == document_id,
            Document.user_id == user_id,
        ).first()
        if not document:
            raise DocumentIndexNotFound(f"文档 {document_id} 不存在或上传者不匹配")

        document.status = "indexing"
        document.error_message = None
        db.commit()
        store_doc_index_progress(document_id, "indexing", task_id=task_id)

        file_path = document.storage_key or document.file_path
        if not file_path or not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")

        chunk_size = CHUNK_TOKENS
        overlap = OVERLAP_TOKENS
        if document.kb_id is not None:
            from app.models.knowledge_base import KnowledgeBase

            kb = db.query(KnowledgeBase).filter(
                KnowledgeBase.id == document.kb_id
            ).first()
            if kb and kb.chunk_config:
                try:
                    config = json.loads(kb.chunk_config)
                    chunk_size = max(
                        100,
                        min(2000, int(config.get("chunk_size", CHUNK_TOKENS))),
                    )
                    overlap = max(
                        0,
                        min(
                            chunk_size - 1,
                            int(config.get("chunk_overlap", OVERLAP_TOKENS)),
                        ),
                    )
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass

        from app.rag.embeddings import get_embedder
        from app.rag.vectorstore import get_qdrant_store

        embedder = get_embedder()
        pipeline_hash = get_qdrant_store().compute_pipeline_hash(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            model_name=embedder.model_name,
            dim=embedder.dim,
        )

        store_doc_index_progress(document_id, "parsing", task_id=task_id)
        chunks = _run_document_chunking(
            document_id,
            db,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        ensure_lock()

        store_doc_index_progress(
            document_id,
            "embedding",
            task_id=task_id,
            total_chunks=len(chunks),
        )
        embedding_count = embedder.generate_for_chunks(
            db,
            document_id=document_id,
            strict_vectorstore=True,
        )
        ensure_lock()

        document.status = "indexed"
        document.source_retained = True
        document.error_message = None
        document.pipeline_version = pipeline_hash
        db.commit()
        store_doc_index_progress(
            document_id,
            "indexed",
            task_id=task_id,
            total_chunks=len(chunks),
            total_embeddings=embedding_count,
        )
        return {
            "status": "indexed",
            "document_id": document_id,
            "chunks": len(chunks),
            "embeddings": embedding_count,
            "pipeline_version": pipeline_hash,
        }
    except Exception:
        db.rollback()
        raise
    finally:
        heartbeat_stop.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        if lock_acquired:
            try:
                release_doc_index_lock(document_id, task_id)
            except Exception:
                pass
        db.close()


__all__ = [
    "DocumentIndexLockBusy",
    "DocumentIndexNotFound",
    "_run_document_chunking",
    "run_document_index",
]
