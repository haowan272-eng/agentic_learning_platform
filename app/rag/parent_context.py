"""Runtime parent-context expansion for retrieved child chunks."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.rag.chunk_models import DocumentChunk
from app.rag.chunker import PARENT_TOKENS, approx_token_len


def _as_int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _joined_with_budget(center: int, chunks: list[DocumentChunk], max_tokens: int) -> str:
    by_index = {chunk.chunk_index: chunk for chunk in chunks}
    if center not in by_index:
        return ""

    selected = [center]
    used = approx_token_len(by_index[center].content)
    left = center - 1
    right = center + 1
    take_left = True

    while True:
        index = left if take_left else right
        take_left = not take_left
        chunk = by_index.get(index)
        if chunk is None:
            if left < min(by_index) and right > max(by_index):
                break
            if index == left:
                left -= 1
            else:
                right += 1
            continue
        count = approx_token_len(chunk.content)
        if used + count > max_tokens and selected:
            break
        selected.append(index)
        used += count
        if index == left:
            left -= 1
        else:
            right += 1

    selected.sort()
    return "\n\n".join(by_index[index].content for index in selected)


def attach_parent_contexts(
    db: Session,
    results: list[dict],
    radius: int = 1,
    max_tokens: int = PARENT_TOKENS,
) -> list[dict]:
    """Attach parent contexts by expanding around retrieved child chunks.

    Retrieval still happens on child chunks. This function performs one batched
    SQL read for neighboring chunks and fills ``parent_content`` at runtime, so
    storage does not need to duplicate parent windows in metadata or Qdrant.
    """
    if not results:
        return results

    hits_by_doc: dict[int, list[int]] = defaultdict(list)
    for result in results:
        document_id = _as_int(result.get("document_id"))
        chunk_index = _as_int(result.get("chunk_index"))
        if document_id is None or chunk_index is None:
            continue
        hits_by_doc[document_id].append(chunk_index)
    if not hits_by_doc:
        return results

    conditions = []
    for document_id, indexes in hits_by_doc.items():
        low = min(indexes) - max(0, radius)
        high = max(indexes) + max(0, radius)
        conditions.append(
            (DocumentChunk.document_id == document_id)
            & (DocumentChunk.chunk_index >= low)
            & (DocumentChunk.chunk_index <= high)
        )

    rows: Iterable[DocumentChunk] = (
        db.query(DocumentChunk)
        .filter(or_(*conditions))
        .order_by(DocumentChunk.document_id, DocumentChunk.chunk_index)
        .all()
    )
    chunks_by_doc: dict[int, list[DocumentChunk]] = defaultdict(list)
    for row in rows:
        chunks_by_doc[row.document_id].append(row)

    for result in results:
        document_id = _as_int(result.get("document_id"))
        chunk_index = _as_int(result.get("chunk_index"))
        if document_id is None or chunk_index is None:
            continue
        parent = _joined_with_budget(
            chunk_index,
            chunks_by_doc.get(document_id, []),
            max_tokens=max(1, max_tokens),
        )
        if parent:
            result["parent_content"] = parent
    return results


__all__ = ["attach_parent_contexts"]
