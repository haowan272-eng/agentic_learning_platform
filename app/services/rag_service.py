"""RAG service orchestration."""
from __future__ import annotations

import json
import contextvars
import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Document, KnowledgeBaseMember, RagConversation, RagMessage, User
from app.rag.answering import (
    build_evidence,
    extractive_fallback,
    get_rag_answerer,
    validate_answer_citations,
)
from app.rag.chain import Retriever
from app.rag.query_rewriter import get_query_rewriter
from app.rag.embeddings import get_embedder
from app.rag.parent_context import attach_parent_contexts
from app.core.config import RAG_QUERY_REWRITE_ENABLED
from app.schemas.rag import (
    AnswerRequest,
    AnswerResponse,
    CitationResult,
    MemoryResult,
    RetrievedSourceResult,
)
from app.services.conversation_context import build_conversation_context
from app.services.memory_service import (
    load_memory,
    memory_context,
    remember_short_term_window,
    retrieval_query,
)
from app.services.short_term_memory import append_short_term_message
from app.core.database import SessionLocal

logger = logging.getLogger(__name__)


def _rewrite_query_for_retrieval(
    question: str,
    history: str,
    memory: str,
    task_state: str,
    enabled: bool,
) -> str:
    if not enabled:
        return question
    try:
        return get_query_rewriter().rewrite(
            question=question,
            history=history,
            memory=memory,
            task_state=task_state,
        )
    except Exception:
        logger.exception("query rewrite failed; falling back to original query")
        return question


def _get_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


def _private_conversation(
    db: Session,
    conversation_id: int,
    user_id: int,
) -> RagConversation:
    row = db.query(RagConversation).filter(
        RagConversation.id == conversation_id,
        RagConversation.user_id == user_id,
    ).first()
    if not row:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return row


def _effective_kb(
    body: AnswerRequest,
    conversation: Optional[RagConversation],
) -> Optional[int]:
    effective = body.kb_id
    if conversation and conversation.kb_id is not None:
        if effective is not None and effective != conversation.kb_id:
            raise HTTPException(status_code=400, detail="Request knowledge base does not match conversation knowledge base")
        effective = conversation.kb_id
    return effective


def _ensure_kb_viewer(db: Session, user_id: int, kb_id: int) -> None:
    membership = db.query(KnowledgeBaseMember).filter(
        KnowledgeBaseMember.kb_id == kb_id,
        KnowledgeBaseMember.user_id == user_id,
    ).first()
    if not membership:
        raise HTTPException(status_code=403, detail="User is not a member of this knowledge base")


def _validate_document(
    db: Session,
    document_id: Optional[int],
    kb_id: Optional[int],
    user: User,
) -> None:
    if document_id is None:
        return
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    if document.kb_id is None:
        if kb_id is not None:
            raise HTTPException(status_code=400, detail="Personal document does not belong to a knowledge base")
        if document.user_id != user.id:
            raise HTTPException(status_code=403, detail="No permission to access this personal document")
        return
    if kb_id is not None and document.kb_id != kb_id:
        raise HTTPException(status_code=400, detail="Document does not belong to the specified knowledge base")
    _ensure_kb_viewer(db, user.id, document.kb_id)

def run_rag_answer(
    db: Session,
    username: str,
    body: AnswerRequest,
    on_token: Optional[Callable[[str], None]] = None,
) -> AnswerResponse:
    request_started = time.perf_counter()
    timings_ms: dict[str, float] = {}
    last_mark = request_started

    def mark(name: str) -> None:
        nonlocal last_mark
        now = time.perf_counter()
        timings_ms[name] = round((now - last_mark) * 1000, 2)
        last_mark = now

    user = _get_user(db, username)
    had_conversation = body.conversation_id is not None
    conversation = (
        _private_conversation(db, body.conversation_id, user.id)
        if body.conversation_id is not None
        else None
    )
    effective_kb_id = _effective_kb(body, conversation)
    if effective_kb_id is not None:
        _ensure_kb_viewer(db, user.id, effective_kb_id)
    _validate_document(db, body.document_id, effective_kb_id, user)
    context_state = build_conversation_context(db, conversation)

    if conversation is None:
        conversation = RagConversation(
            user_id=user.id,
            kb_id=effective_kb_id,
            title=body.query[:80],
            task_state_json=json.dumps(
                {"goal": body.query[:500], "recent_requests": [body.query[:500]]},
                ensure_ascii=False,
            ),
        )
        db.add(conversation)
        db.flush()
    user_message = RagMessage(
        conversation_id=conversation.id,
        role="user",
        content=body.query,
        memory_extracted=not body.use_memory,
    )
    db.add(user_message)
    conversation.updated_at = func.now()
    db.commit()
    db.refresh(user_message)
    append_short_term_message(
        user.id,
        conversation.id,
        user_message.id,
        user_message.role,
        user_message.content,
        user_message.created_at,
    )
    if body.use_memory:
        remember_short_term_window(db, user.id, conversation.id)
        db.commit()

    mark("setup_ms")
    memories = load_memory(db, user.id) if body.use_memory else []
    memory_text = memory_context(memories)
    mark("memory_ms")
    rewritten_query = _rewrite_query_for_retrieval(
        question=body.query,
        history=context_state.history,
        memory=memory_text,
        task_state=context_state.task_state,
        enabled=had_conversation and body.rewrite_query and RAG_QUERY_REWRITE_ENABLED,
    )
    mark("rewrite_ms")
    search_query = retrieval_query(rewritten_query, memories)
    personal_space_only = effective_kb_id is None and body.document_id is None
    retrieval_user_id = user.id if personal_space_only else None
    try:
        embedder = get_embedder()
        bm25_weight = body.bm25_weight
        if bm25_weight > 0:
            try:
                embedder.ensure_bm25(
                    db,
                    user_id=retrieval_user_id,
                    document_id=body.document_id,
                    kb_id=effective_kb_id,
                    personal_space_only=personal_space_only,
                )
            except Exception:
                bm25_weight = 0.0
        mark("bm25_ms")
        results = Retriever(embedder).retrieve(
            query=search_query,
            top_k=body.top_k,
            document_id=body.document_id,
            user_id=retrieval_user_id,
            kb_id=effective_kb_id,
            personal_space_only=personal_space_only,
            bm25_weight=bm25_weight,
        )
        results = attach_parent_contexts(db, results)
        mark("retrieve_ms")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Knowledge retrieval is temporarily unavailable: {exc}") from exc

    evidence, records = build_evidence(results)
    mark("evidence_ms")
    degraded = False
    if not records:
        answer = "No sufficient information was found in the knowledge base to answer this question."
        response_records = []
    else:
        try:
            answerer = get_rag_answerer()
            generation_args = {
                "question": body.query,
                "context": evidence,
                "history": context_state.history,
                "memory": memory_text,
                "task_state": context_state.task_state,
            }
            if on_token is None:
                answer = answerer.answer(**generation_args)
            else:
                chunks = []
                for token in answerer.stream(**generation_args):
                    chunks.append(token)
                    on_token(token)
                answer = "".join(chunks).strip()
            answer, response_records = validate_answer_citations(answer, records)
            if not response_records:
                raise ValueError("Answer model did not return valid citations")
        except Exception:
            answer, response_records = extractive_fallback(records)
            degraded = True

    mark("answer_ms")
    citations = [CitationResult(**record.citation_dict()) for record in response_records]
    retrieved_sources = [RetrievedSourceResult(**record.as_dict()) for record in records]
    retrieved_contexts = [record.context for record in records]
    memory_payload = [item.as_dict() for item in memories]
    assistant_message = RagMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        citations_json=json.dumps(
            [citation.model_dump() for citation in citations],
            ensure_ascii=False,
        ),
        memory_json=json.dumps(memory_payload, ensure_ascii=False),
        memory_extracted=True,
        degraded=degraded,
    )
    db.add(assistant_message)
    conversation.updated_at = func.now()
    db.commit()
    db.refresh(assistant_message)
    append_short_term_message(
        user.id,
        conversation.id,
        assistant_message.id,
        assistant_message.role,
        assistant_message.content,
        assistant_message.created_at,
    )

    mark("persist_ms")
    timings_ms["total_ms"] = round((time.perf_counter() - request_started) * 1000, 2)
    logger.info("rag timings query=%r timings_ms=%s", body.query, timings_ms)

    return AnswerResponse(
        query=body.query,
        rewritten_query=rewritten_query if rewritten_query != body.query else None,
        answer=answer,
        conversation_id=conversation.id,
        citations=citations,
        retrieved_contexts=retrieved_contexts,
        retrieved_sources=retrieved_sources,
        retrieved_count=len(results),
        memory_used=[MemoryResult(**item) for item in memory_payload],
        degraded=degraded,
        context_compacted=context_state.compacted,
        timings_ms=timings_ms,
    )


class _ClientDisconnected(RuntimeError):
    pass


def stream_rag_answer(username: str, body: AnswerRequest) -> Iterator[dict]:
    """Run the existing RAG transaction in a worker thread and emit bounded SSE events."""
    events: queue.Queue[dict] = queue.Queue(maxsize=100)
    stopped = threading.Event()
    request_context = contextvars.copy_context()

    def emit(event: str, data: dict) -> None:
        while not stopped.is_set():
            try:
                events.put({"event": event, "data": data}, timeout=0.5)
                return
            except queue.Full:
                continue
        raise _ClientDisconnected()

    def execute() -> None:
        db = SessionLocal()
        try:
            result = run_rag_answer(
                db,
                username,
                body,
                on_token=lambda token: emit("token", {"delta": token}),
            )
            emit("final", result.model_dump(mode="json"))
        except _ClientDisconnected:
            logger.info("SSE client disconnected before completion")
        except HTTPException as exc:
            try:
                emit("error", {"status_code": exc.status_code, "detail": exc.detail})
            except _ClientDisconnected:
                pass
        except Exception:
            logger.exception("streaming RAG answer failed")
            try:
                emit("error", {"status_code": 500, "detail": "Streaming RAG answer failed"})
            except _ClientDisconnected:
                pass
        finally:
            db.close()
            try:
                emit("done", {})
            except _ClientDisconnected:
                pass

    thread = threading.Thread(
        target=lambda: request_context.run(execute),
        name="rag-sse-answer",
        daemon=True,
    )
    thread.start()
    try:
        while True:
            item = events.get()
            yield item
            if item["event"] == "done":
                return
    finally:
        stopped.set()
