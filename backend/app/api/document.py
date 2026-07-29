"""多格式共享文档 API：上传、列表、删除与异步索引。

上传流程：
  1. 认证用户上传文件字节。
  2. 创建 Document 生命周期记录，user_id 仅表示上传者。
  3. 生成 document_id 和 storage_key。
  4. 临时保存文件到受控存储。
  5. 推送 document_index 任务到 Redis/Celery。
  6. 立即返回 document_id 和 status。
"""
import os
import shutil
import uuid
import json
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Query
from sqlalchemy import false, or_
from sqlalchemy.orm import Session
from typing import Optional

from app.core.database import get_db
from app.core.errors import AppError, ErrorCode
from ..models import User, Document, KnowledgeBase, KnowledgeBaseMember
from app.api.deps import get_current_user
from ..queue import enqueue_document_index_task, get_doc_index_progress
from app.rag.chunker import SUPPORTED_DOCUMENT_EXTENSIONS
from app.core.config import MAX_DOCUMENT_SIZE_MB, DOCUMENT_UPLOAD_CHUNK_BYTES, UPLOAD_DIR
from app.models.document_quality import DocumentQualityReport
from app.services.document_quality import (
    assess_document_chunks,
    list_document_type_specs,
    normalize_document_type,
    report_to_dict,
    save_quality_report,
)

os.makedirs(UPLOAD_DIR, exist_ok=True)

router = APIRouter(prefix="/document", tags=["Document"])

_ZIP_FORMATS = {".docx", ".pptx", ".xlsx", ".epub"}


def _has_valid_signature(extension: str, header: bytes) -> bool:
    """校验高风险二进制格式的文件头；文本类格式交由解析器和编码检查"""
    if extension == ".pdf":
        return header.startswith(b"%PDF-")
    if extension in {".jpg", ".jpeg"}:
        return header.startswith(b"\xff\xd8\xff")
    if extension == ".png":
        return header.startswith(b"\x89PNG\r\n\x1a\n")
    if extension == ".gif":
        return header.startswith((b"GIF87a", b"GIF89a"))
    if extension == ".webp":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP"
    if extension in _ZIP_FORMATS:
        return header.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))
    if extension in {".doc", ".xls"}:
        return header.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")
    return True

def _get_user(db: Session, username: str) -> User:
    """根据用户名获取用户，不存在则 401"""
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="用户不存在")
    return user


def _get_scoped_document(
    db: Session,
    current_user: str,
    document_id: int,
    kb_role: str = "viewer",
) -> tuple[User, Document]:
    user = _get_user(db, current_user)
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found")
    if doc.kb_id is None:
        if doc.user_id != user.id:
            raise HTTPException(status_code=403, detail="Document is outside your personal scope")
    else:
        from app.api.deps import check_kb_role

        check_kb_role(db, current_user, doc.kb_id, kb_role)
    return user, doc


def _progress_percent(status_value: str) -> int:
    return {
        "uploaded": 5,
        "queued": 10,
        "waiting_lock": 12,
        "retrying": 18,
        "indexing": 25,
        "parsing": 45,
        "embedding": 75,
        "indexed": 100,
        "completed": 100,
        "failed": 100,
        "unknown": 0,
        "not_found": 0,
    }.get(status_value, 0)


def _split_form_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        loaded = json.loads(value)
        if isinstance(loaded, list):
            return [str(item).strip() for item in loaded if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in value.split(",") if item.strip()]


def _latest_quality_report(db: Session, document_id: int) -> DocumentQualityReport | None:
    return (
        db.query(DocumentQualityReport)
        .filter(DocumentQualityReport.document_id == document_id)
        .order_by(DocumentQualityReport.created_at.desc(), DocumentQualityReport.id.desc())
        .first()
    )


@router.get("/types")
def document_types():
    return {"types": list_document_type_specs()}


@router.get("/list")
def list_documents(
    kb_id: Optional[int] = Query(None, description="按知识库筛"),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """列出共享文档语料，可选按知识库筛选"""
    user = _get_user(db, current_user)

    if kb_id is not None:
        from app.api.deps import check_kb_role
        check_kb_role(db, current_user, kb_id, "viewer")
        q = db.query(Document).filter(Document.kb_id == kb_id)
    else:
        member_kb_ids = [
            row.kb_id
            for row in db.query(KnowledgeBaseMember.kb_id)
            .filter(KnowledgeBaseMember.user_id == user.id)
            .all()
        ]
        shared_kb_ids = [
            row.id
            for row in db.query(KnowledgeBase.id)
            .filter(KnowledgeBase.visibility == "shared")
            .all()
        ]
        accessible_kb_ids = sorted(set(member_kb_ids + shared_kb_ids))
        q = db.query(Document).filter(
            or_(
                (Document.kb_id.is_(None)) & (Document.user_id == user.id),
                Document.kb_id.in_(accessible_kb_ids) if accessible_kb_ids else false(),
            )
        )
    docs = q.order_by(Document.created_at.desc()).all()
    return [
        {
            "id": d.id,
            "file_name": d.original_file_name or d.file_name,
            "content_type": d.content_type,
            "file_size": d.file_size,
            "status": d.status,
            "source_retained": d.source_retained,
            "doc_type": d.doc_type,
            "title": d.title,
            "tags": _split_form_list(d.tags_json),
            "target_roles": _split_form_list(d.target_roles_json),
            "seniority": d.seniority,
            "quality_score": d.quality_score,
            "quality_level": d.quality_level,
            "created_at": str(d.created_at),
            "uploaded_by": d.user_id,
            "kb_id": d.kb_id,
        }
        for d in docs
    ]


@router.get("/{document_id}/progress")
def document_index_progress(
    document_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _, doc = _get_scoped_document(db, current_user, document_id, "viewer")
    progress = get_doc_index_progress(document_id)
    progress_status = str(progress.get("status") or "")
    effective_status = (
        progress_status
        if progress_status and progress_status not in {"not_found", "unknown"}
        else str(doc.status or "unknown")
    )
    percent = _progress_percent(effective_status)
    if doc.status in {"indexed", "completed"}:
        effective_status = str(doc.status)
        percent = 100
    if doc.status == "failed":
        effective_status = "failed"
        percent = 100
    return {
        "document_id": doc.id,
        "status": effective_status,
        "document_status": doc.status,
        "percent": percent,
        "task_id": progress.get("task_id"),
        "attempt": progress.get("attempt"),
        "total_chunks": progress.get("total_chunks"),
        "total_embeddings": progress.get("total_embeddings"),
        "error_message": progress.get("error_message") or doc.error_message,
        "updated_at": str(doc.updated_at) if doc.updated_at else None,
    }


@router.get("/{document_id}/quality")
def get_document_quality(
    document_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _, doc = _get_scoped_document(db, current_user, document_id, "viewer")
    report = _latest_quality_report(db, document_id)
    return {
        "document_id": doc.id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "quality_score": doc.quality_score,
        "quality_level": doc.quality_level,
        "report": report_to_dict(report) if report else None,
    }


@router.post("/{document_id}/quality/recheck")
def recheck_document_quality(
    document_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _, doc = _get_scoped_document(db, current_user, document_id, "editor")
    report_payload = assess_document_chunks(db, doc)
    report = save_quality_report(db, doc, report_payload)
    db.commit()
    db.refresh(report)
    db.refresh(doc)
    return {
        "document_id": doc.id,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "quality_score": doc.quality_score,
        "quality_level": doc.quality_level,
        "report": report_to_dict(report),
    }


@router.post("/{document_id}/reindex", status_code=202)
def reindex_document(
    document_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user, doc = _get_scoped_document(db, current_user, document_id, "editor")
    if doc.kb_id is not None:
        from app.api.deps import is_platform_admin

        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == doc.kb_id).first()
        if kb and kb.visibility == "shared" and not is_platform_admin(current_user):
            raise HTTPException(status_code=403, detail="Only administrators can reindex shared knowledge base documents")
    progress_status = str(get_doc_index_progress(document_id).get("status") or "")
    if doc.status in {"queued", "indexing"} or progress_status in {"queued", "waiting_lock", "retrying", "indexing", "parsing", "embedding"}:
        raise AppError(
            code=ErrorCode.DOCUMENT_ALREADY_INDEXING,
            message="Document is already being indexed",
            status_code=409,
        )
    file_path = doc.storage_key or doc.file_path
    if not file_path or not os.path.exists(file_path):
        raise AppError(
            code=ErrorCode.DOCUMENT_SOURCE_MISSING,
            message="Original document file is not available for reindexing",
            status_code=409,
        )
    doc.status = "uploaded"
    doc.error_message = None
    db.commit()
    try:
        task_id = enqueue_document_index_task(doc.id, user.id, doc.kb_id)
    except Exception as exc:
        doc.status = "failed"
        doc.error_message = f"Failed to enqueue reindex task: {exc}"
        db.commit()
        raise AppError(
            code=ErrorCode.DOCUMENT_INDEX_QUEUE_UNAVAILABLE,
            message="Index queue is unavailable",
            status_code=503,
            retryable=True,
            details={"document_id": doc.id},
        ) from exc
    return {
        "id": doc.id,
        "file_name": doc.original_file_name or doc.file_name,
        "status": doc.status,
        "task_id": task_id,
        "kb_id": doc.kb_id,
    }


@router.delete("/{document_id}")
def delete_document(
    document_id: int,
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    删除指定文档及其所有关联数据（上传者或 KB 管理员）：

    1. 删除关联分块和 chunk_embeddings。
    2. 删除 Qdrant 中的向量。
    3. 删除 documents 表记录。
    4. 删除磁盘文件和派生资源（如果存在）。
    """
    user = _get_user(db, current_user)

    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档 {document_id} 不存在")
    if doc.user_id != user.id:
        if doc.kb_id is None:
            raise HTTPException(status_code=403, detail="只有上传者可以删除该文档")
        from app.api.deps import check_kb_role
        check_kb_role(db, current_user, doc.kb_id, "admin")

    # Delete related chunks; cascade removes chunk embeddings.
    from app.rag.chunk_models import DocumentChunk

    chunk_count = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document_id)
        .delete()
    )

    # 3. 删除 Qdrant 中的向量
    qdrant_deleted = 0
    try:
        from app.rag.vectorstore import get_qdrant_store

        qdrant_deleted = get_qdrant_store().delete_by_document_id(document_id, strict=True)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=503, detail=f"向量删除失败，文档未删除: {e}") from e

    # 4. 删除文档记录
    db.delete(doc)
    db.commit()

    # Delete disk file if file_path or storage_key exists.
    file_deleted = False
    for path_candidate in [doc.file_path, doc.storage_key]:
        if path_candidate and os.path.exists(path_candidate):
            asset_dir = Path(path_candidate).parent / "rag_assets" / Path(path_candidate).stem
            caption_cache = Path(path_candidate).with_suffix(Path(path_candidate).suffix + ".caption.json")
            os.remove(path_candidate)
            file_deleted = True
            if caption_cache.exists():
                caption_cache.unlink()
            if asset_dir.exists():
                shutil.rmtree(asset_dir)

    return {
        "ok": True,
        "document_id": document_id,
        "file_name": doc.original_file_name or doc.file_name,
        "file_deleted": file_deleted,
        "chunks_deleted": chunk_count,
        "qdrant_points_deleted": qdrant_deleted,
    }


@router.post("/upload", status_code=202)
async def upload_document(
    file: UploadFile = File(...),
    doc_type: Optional[str] = Form(None, description="业务资料类型"),
    title: Optional[str] = Form(None, description="资料标题"),
    tags: Optional[str] = Form(None, description="逗号分隔或 JSON 数组标签"),
    target_roles: Optional[str] = Form(None, description="逗号分隔或 JSON 数组目标岗位"),
    seniority: Optional[str] = Form(None, description="目标级别"),
    source: Optional[str] = Form("manual_upload", description="资料来源"),
    kb_id: Optional[int] = Form(None, description="目标知识库 ID"),
    current_user: str = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """上传受支持文档并触发异步多模态索引任务。

    流程：
      1. 验证文件类型和 KB 权限。
      2. 创建 Document 生命周期记录。
      3. 保存文件到受控存储。
      4. 推送 document_index 任务到 Redis/Celery。
      5. 立即返回，不等待索引完成。
    """
    safe_filename = Path(file.filename or "").name
    extension = Path(safe_filename).suffix.lower()
    if not safe_filename or extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_DOCUMENT_EXTENSIONS))
        raise AppError(
            code=ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            message=f"不支持的文件格式，当前支持：{supported}",
            status_code=400,
            details={"supported": sorted(SUPPORTED_DOCUMENT_EXTENSIONS)},
        )

    normalized_doc_type = normalize_document_type(doc_type)
    if doc_type and normalized_doc_type == "unknown":
        supported_types = ", ".join(spec["code"] for spec in list_document_type_specs())
        raise AppError(
            code=ErrorCode.DOCUMENT_TYPE_UNSUPPORTED,
            message=f"Unsupported document type. Supported: {supported_types}",
            status_code=400,
            details={"supported_types": [spec["code"] for spec in list_document_type_specs()]},
        )

    user = _get_user(db, current_user)

    # 0. 校验 KB 权限
    if kb_id is not None:
        from app.api.deps import check_kb_role, is_platform_admin

        kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
        if not kb:
            raise HTTPException(status_code=404, detail="Knowledge base not found")
        if kb.visibility == "shared":
            if not is_platform_admin(current_user):
                raise HTTPException(status_code=403, detail="Only administrators can upload to shared knowledge bases")
            check_kb_role(db, current_user, kb_id, "admin")
        else:
            check_kb_role(db, current_user, kb_id, "editor")

    # 1. 生成系统存储引用
    unique_name = f"{uuid.uuid4().hex}_{safe_filename}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    # 2. 流式保存，避免大文件一次性占用 API 进程内存。
    file_size = 0
    header = bytearray()
    max_bytes = MAX_DOCUMENT_SIZE_MB * 1024 * 1024
    try:
        with open(save_path, "wb") as output:
            while chunk := await file.read(DOCUMENT_UPLOAD_CHUNK_BYTES):
                file_size += len(chunk)
                if file_size > max_bytes:
                    raise AppError(
                        code=ErrorCode.DOCUMENT_FILE_TOO_LARGE,
                        message=f"文件超过 {MAX_DOCUMENT_SIZE_MB}MB 限制",
                        status_code=413,
                        details={"max_mb": MAX_DOCUMENT_SIZE_MB},
                    )
                if len(header) < 32:
                    header.extend(chunk[: 32 - len(header)])
                output.write(chunk)
        if file_size == 0:
            raise AppError(code=ErrorCode.DOCUMENT_FILE_EMPTY, message="上传文件为空", status_code=400)
        if not _has_valid_signature(extension, bytes(header)):
            raise AppError(code=ErrorCode.DOCUMENT_SIGNATURE_INVALID, message="文件内容与扩展名不匹配", status_code=400)
    except Exception:
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

    # 3. 创建 Document 生命周期记录
    doc = Document(
        user_id=user.id,
        kb_id=kb_id,
        original_file_name=safe_filename,
        content_type=file.content_type or "application/octet-stream",
        file_size=file_size,
        file_name=safe_filename,          # 兼容旧字段
        file_path=save_path,              # 兼容旧字段
        source_retained=True,
        storage_backend="local",
        storage_key=save_path,
        doc_type=normalized_doc_type,
        title=(title or safe_filename).strip()[:255],
        source=(source or "manual_upload").strip()[:64],
        tags_json=json.dumps(_split_form_list(tags), ensure_ascii=False),
        target_roles_json=json.dumps(_split_form_list(target_roles), ensure_ascii=False),
        seniority=seniority.strip()[:64] if seniority else None,
        business_visibility="private",
        status="uploaded",
    )
    try:
        db.add(doc)
        db.commit()
        db.refresh(doc)
    except Exception:
        db.rollback()
        if os.path.exists(save_path):
            os.remove(save_path)
        raise

    # 4. 推送 document_index 任务到 Redis/Celery。
    task_id = None
    try:
        task_id = enqueue_document_index_task(doc.id, user.id, kb_id)
    except Exception as e:
        doc.status = "failed"
        doc.error_message = f"索引任务入队失败: {e}"
        db.commit()
        raise AppError(
            code=ErrorCode.DOCUMENT_INDEX_QUEUE_UNAVAILABLE,
            message="文档已保存，但索引队列不可用",
            status_code=503,
            retryable=True,
            details={"document_id": doc.id},
        ) from e

    return {
        "id": doc.id,
        "file_name": safe_filename,
        "status": doc.status,
        "doc_type": doc.doc_type,
        "title": doc.title,
        "quality_score": doc.quality_score,
        "quality_level": doc.quality_level,
        "task_id": task_id,
        "kb_id": kb_id,
        "created_at": str(doc.created_at),
    }
