"""Celery文档索引任务：ACK、重试、最终失败与死信记录"""
from __future__ import annotations

from app.core.celery import celery_app
from app.core.config import DOC_INDEX_MAX_RETRIES
from app.core.database import SessionLocal
from app.models import Document
from app.queue import record_dead_document_task, store_doc_index_progress
from app.services.document_index_service import (
    DocumentIndexLockBusy,
    DocumentIndexNotFound,
    run_document_index,
)


def _update_failed_document(
    document_id: int,
    status: str,
    error_message: str,
) -> None:
    db = SessionLocal()
    try:
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.status = status
            document.error_message = error_message[:2000]
            db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


@celery_app.task(
    bind=True,
    name="app.tasks.document_index.index_document",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=DOC_INDEX_MAX_RETRIES,
)
def index_document_task(
    self,
    document_id: int,
    user_id: int,
    kb_id: int | None = None,
) -> dict:
    task_id = str(self.request.id)
    attempt = int(self.request.retries or 0)
    try:
        return run_document_index(document_id, user_id, task_id)
    except DocumentIndexLockBusy as exc:
        message = str(exc)
        store_doc_index_progress(
            document_id,
            "waiting_lock",
            message,
            task_id=task_id,
            attempt=attempt + 1,
        )
        raise self.retry(
            exc=exc,
            countdown=min(60, max(5, 2 ** attempt)),
        )
    except DocumentIndexNotFound as exc:
        message = str(exc)
        _update_failed_document(document_id, "failed", message)
        store_doc_index_progress(
            document_id,
            "failed",
            message,
            task_id=task_id,
            attempt=attempt,
        )
        return {"status": "failed", "document_id": document_id, "error": message}
    except Exception as exc:
        message = str(exc)
        if attempt < DOC_INDEX_MAX_RETRIES:
            _update_failed_document(document_id, "uploaded", message)
            store_doc_index_progress(
                document_id,
                "retrying",
                message,
                task_id=task_id,
                attempt=attempt + 1,
            )
            raise self.retry(
                exc=exc,
                countdown=min(60, 2 ** attempt),
            )

        _update_failed_document(document_id, "failed", message)
        payload = {
            "task_id": task_id,
            "document_id": document_id,
            "user_id": user_id,
            "kb_id": kb_id,
            "attempt": attempt,
            "last_error": message[:1000],
        }
        record_dead_document_task(payload)
        store_doc_index_progress(
            document_id,
            "failed",
            message,
            task_id=task_id,
            attempt=attempt,
        )
        raise


__all__ = ["index_document_task"]
