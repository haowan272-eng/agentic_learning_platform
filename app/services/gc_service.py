"""Global garbage collection helpers for uploaded files, stale docs, and Qdrant.

The collector is intentionally conservative:
- dry-run by default;
- only removes files under UPLOAD_DIR;
- deletes database documents only when explicitly requested by age/status rules.
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import UPLOAD_DIR
from app.models import Document
from app.rag.chunk_models import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass
class GcReport:
    dry_run: bool
    orphan_files_found: int = 0
    orphan_files_deleted: int = 0
    orphan_assets_found: int = 0
    orphan_assets_deleted: int = 0
    stale_documents_found: int = 0
    stale_documents_deleted: int = 0
    orphan_qdrant_points_found: int = 0
    orphan_qdrant_points_deleted: int = 0
    errors: list[str] = field(default_factory=list)
    details: dict[str, list[str]] = field(default_factory=lambda: {
        "orphan_files": [],
        "orphan_assets": [],
        "stale_documents": [],
        "orphan_qdrant_points": [],
    })

    def to_dict(self) -> dict[str, Any]:
        return {
            "dry_run": self.dry_run,
            "orphan_files_found": self.orphan_files_found,
            "orphan_files_deleted": self.orphan_files_deleted,
            "orphan_assets_found": self.orphan_assets_found,
            "orphan_assets_deleted": self.orphan_assets_deleted,
            "stale_documents_found": self.stale_documents_found,
            "stale_documents_deleted": self.stale_documents_deleted,
            "orphan_qdrant_points_found": self.orphan_qdrant_points_found,
            "orphan_qdrant_points_deleted": self.orphan_qdrant_points_deleted,
            "errors": self.errors,
            "details": self.details,
        }


def _upload_root() -> Path:
    return Path(UPLOAD_DIR).resolve()


def _is_under_upload_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(_upload_root())
        return True
    except ValueError:
        return False


def _document_paths(doc: Document) -> set[Path]:
    paths: set[Path] = set()
    for value in (doc.file_path, doc.storage_key):
        if value:
            path = Path(value)
            if _is_under_upload_dir(path):
                paths.add(path.resolve())
    return paths


def _related_asset_paths(source_path: Path) -> list[Path]:
    return [
        source_path.with_suffix(source_path.suffix + ".caption.json"),
        source_path.parent / "rag_assets" / source_path.stem,
    ]


def _delete_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def _known_storage_paths(db: Session) -> set[Path]:
    known: set[Path] = set()
    for doc in db.query(Document).all():
        known.update(_document_paths(doc))
    return known


def collect_orphan_files(db: Session, report: GcReport, execute: bool) -> None:
    root = _upload_root()
    if not root.exists():
        return

    known = _known_storage_paths(db)
    ignored_parts = {"rag_assets"}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in ignored_parts for part in path.relative_to(root).parts):
            continue
        resolved = path.resolve()
        if resolved in known:
            continue
        if path.name.endswith(".caption.json"):
            continue

        report.orphan_files_found += 1
        report.details["orphan_files"].append(str(resolved))
        if execute:
            try:
                _delete_path(resolved)
                report.orphan_files_deleted += 1
            except Exception as exc:  # pragma: no cover - defensive logging path
                report.errors.append(f"delete file failed: {resolved}: {exc}")


def collect_orphan_assets(db: Session, report: GcReport, execute: bool) -> None:
    root = _upload_root()
    if not root.exists():
        return

    live_assets: set[Path] = set()
    for source in _known_storage_paths(db):
        live_assets.update(path.resolve() for path in _related_asset_paths(source))

    candidates: list[Path] = []
    candidates.extend(root.glob("*.caption.json"))
    for assets_root in root.rglob("rag_assets"):
        if assets_root.is_dir():
            candidates.extend(child for child in assets_root.iterdir())

    for path in candidates:
        resolved = path.resolve()
        if resolved in live_assets:
            continue
        if not _is_under_upload_dir(resolved):
            continue

        report.orphan_assets_found += 1
        report.details["orphan_assets"].append(str(resolved))
        if execute:
            try:
                _delete_path(resolved)
                report.orphan_assets_deleted += 1
            except Exception as exc:  # pragma: no cover
                report.errors.append(f"delete asset failed: {resolved}: {exc}")


def _delete_document_assets(doc: Document, report: GcReport) -> None:
    seen: set[Path] = set()
    for source in _document_paths(doc):
        seen.add(source)
        seen.update(_related_asset_paths(source))

    for path in seen:
        if not path.exists() or not _is_under_upload_dir(path):
            continue
        try:
            _delete_path(path)
        except Exception as exc:  # pragma: no cover
            report.errors.append(f"delete document asset failed: {path}: {exc}")


def collect_stale_documents(
    db: Session,
    report: GcReport,
    execute: bool,
    failed_days: int,
    stuck_hours: int,
) -> None:
    now = datetime.now()
    failed_before = now - timedelta(days=failed_days)
    stuck_before = now - timedelta(hours=stuck_hours)
    stale_docs = (
        db.query(Document)
        .filter(
            ((Document.status == "failed") & (Document.updated_at < failed_before))
            | (Document.status.in_(("uploaded", "indexing")) & (Document.updated_at < stuck_before))
        )
        .all()
    )

    for doc in stale_docs:
        report.stale_documents_found += 1
        report.details["stale_documents"].append(
            f"id={doc.id} status={doc.status} file={doc.original_file_name or doc.file_name}"
        )
        if not execute:
            continue

        try:
            try:
                from app.rag.vectorstore import get_qdrant_store

                get_qdrant_store().delete_by_document_id(doc.id, strict=False)
            except Exception as exc:
                report.errors.append(f"qdrant delete for document {doc.id} failed: {exc}")

            _delete_document_assets(doc, report)
            db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).delete()
            db.delete(doc)
            db.commit()
            report.stale_documents_deleted += 1
        except Exception as exc:
            db.rollback()
            report.errors.append(f"delete stale document {doc.id} failed: {exc}")


def collect_orphan_qdrant_points(db: Session, report: GcReport, execute: bool) -> None:
    try:
        from app.rag.vectorstore import get_qdrant_store
    except ImportError:
        report.errors.append("qdrant-client is not installed")
        return

    store = get_qdrant_store()
    collection = store.get_active_collection()
    if not collection:
        return

    live_doc_ids = {row[0] for row in db.query(Document.id).all()}
    live_chunk_ids = {row[0] for row in db.query(DocumentChunk.id).all()}
    delete_ids: list[int] = []
    offset = None

    try:
        while True:
            points, offset = store.client.scroll(
                collection_name=collection,
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                document_id = payload.get("document_id")
                chunk_id = payload.get("chunk_id")
                if document_id not in live_doc_ids or chunk_id not in live_chunk_ids:
                    delete_ids.append(point.id)
                    report.details["orphan_qdrant_points"].append(str(point.id))
            if offset is None:
                break
    except Exception as exc:
        report.errors.append(f"scan qdrant failed: {exc}")
        return

    report.orphan_qdrant_points_found = len(delete_ids)
    if execute and delete_ids:
        try:
            deleted = store.delete_points_by_ids(delete_ids, collection_name=collection)
            report.orphan_qdrant_points_deleted = len(delete_ids) if deleted else 0
        except Exception as exc:
            report.errors.append(f"delete qdrant points failed: {exc}")


def run_global_gc(
    db: Session,
    *,
    execute: bool = False,
    failed_days: int = 7,
    stuck_hours: int = 24,
    skip_files: bool = False,
    skip_qdrant: bool = False,
    skip_stale_docs: bool = False,
) -> GcReport:
    report = GcReport(dry_run=not execute)

    if not skip_stale_docs:
        collect_stale_documents(db, report, execute, failed_days, stuck_hours)
    if not skip_files:
        collect_orphan_files(db, report, execute)
        collect_orphan_assets(db, report, execute)
    if not skip_qdrant:
        collect_orphan_qdrant_points(db, report, execute)

    logger.info("global gc report: %s", report.to_dict())
    return report

