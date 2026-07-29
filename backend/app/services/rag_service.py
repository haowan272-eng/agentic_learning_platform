"""RAG service entrypoints for HTTP, tools, and SSE streaming."""
from __future__ import annotations

import contextvars
import json
import logging
import queue
import threading
import time
from typing import Callable, Iterator, Optional

from fastapi import HTTPException, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import RAG_QUERY_REWRITE_ENABLED
from app.core.database import SessionLocal
from app.core.errors import AppError, ErrorCode, warning_payload
from app.models import Document, KnowledgeBase, KnowledgeBaseMember, RagConversation, RagMessage, User
from app.rag.answering import (
    build_evidence,
    extractive_fallback,
    get_rag_answerer,
    validate_answer_citations,
)
from app.rag.chain import Retriever
from app.rag.embeddings import get_embedder
from app.rag.parent_context import attach_parent_contexts
from app.rag.query_rewriter import get_query_rewriter
from app.schemas.rag import AnswerRequest, AnswerResponse, CitationResult, RetrievedSourceResult
from app.services.conversation_context import ConversationContext, build_conversation_context
from app.services.learning_service import (
    build_interview_capability_query,
    is_interview_capability_request,
    record_rag_interview_diagnostic,
)

logger = logging.getLogger(__name__)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


def _get_user(db: Session, username: str) -> User:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise AppError(
            code=ErrorCode.AUTH_USER_NOT_FOUND,
            message="User not found",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )
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
        raise AppError(
            code=ErrorCode.RAG_CONVERSATION_NOT_FOUND,
            message="Conversation not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    return row


def _effective_kb(
    body: AnswerRequest,
    conversation: Optional[RagConversation],
) -> Optional[int]:
    effective = body.kb_id
    if conversation and conversation.kb_id is not None:
        if effective is not None and effective != conversation.kb_id:
            raise AppError(
                code=ErrorCode.RAG_SCOPE_MISMATCH,
                message="Request knowledge base does not match conversation knowledge base",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        effective = conversation.kb_id
    return effective


def _ensure_kb_viewer(db: Session, user_id: int, kb_id: int) -> None:
    kb = db.query(KnowledgeBase).filter(KnowledgeBase.id == kb_id).first()
    if not kb:
        raise AppError(
            code=ErrorCode.KB_NOT_FOUND,
            message="Knowledge base not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    if kb.visibility == "shared":
        return
    membership = db.query(KnowledgeBaseMember).filter(
        KnowledgeBaseMember.kb_id == kb_id,
        KnowledgeBaseMember.user_id == user_id,
    ).first()
    if not membership:
        raise AppError(
            code=ErrorCode.KB_FORBIDDEN,
            message="User is not a member of this knowledge base",
            status_code=status.HTTP_403_FORBIDDEN,
        )


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
        raise AppError(
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message="Document not found",
            status_code=status.HTTP_404_NOT_FOUND,
        )
    if document.kb_id is None:
        if kb_id is not None:
            raise AppError(
                code=ErrorCode.RAG_SCOPE_MISMATCH,
                message="Personal document does not belong to a knowledge base",
                status_code=status.HTTP_400_BAD_REQUEST,
            )
        if document.user_id != user.id:
            raise AppError(
                code=ErrorCode.DOCUMENT_FORBIDDEN,
                message="No permission to access this personal document",
                status_code=status.HTTP_403_FORBIDDEN,
            )
        return
    if kb_id is not None and document.kb_id != kb_id:
        raise AppError(
            code=ErrorCode.RAG_SCOPE_MISMATCH,
            message="Document does not belong to the specified knowledge base",
            status_code=status.HTTP_400_BAD_REQUEST,
        )
    _ensure_kb_viewer(db, user.id, document.kb_id)


def _ensure_conversation(
    db: Session,
    user_id: int,
    kb_id: Optional[int],
    query: str,
    conversation: Optional[RagConversation],
) -> RagConversation:
    if conversation is not None:
        return conversation
    conversation = RagConversation(
        user_id=user_id,
        kb_id=kb_id,
        title=query[:80],
        task_state_json=json.dumps(
            {"goal": query[:500], "recent_requests": [query[:500]]},
            ensure_ascii=False,
        ),
    )
    db.add(conversation)
    db.flush()
    return conversation


def _persist_user_message(
    db: Session,
    user_id: int,
    conversation: RagConversation,
    body: AnswerRequest,
) -> None:
    user_message = RagMessage(
        conversation_id=conversation.id,
        role="user",
        content=body.query,
    )
    db.add(user_message)
    conversation.updated_at = func.now()
    db.commit()
    db.refresh(user_message)


def _persist_assistant_message(
    db: Session,
    user_id: int,
    conversation: RagConversation,
    answer: str,
    citations: list[CitationResult],
    degraded: bool,
) -> None:
    assistant_message = RagMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=answer,
        citations_json=json.dumps(
            [citation.model_dump() for citation in citations],
            ensure_ascii=False,
        ),
        degraded=degraded,
    )
    db.add(assistant_message)
    conversation.updated_at = func.now()
    db.commit()
    db.refresh(assistant_message)


def _record_interview_diagnostic(
    db: Session,
    user_id: int,
    body: AnswerRequest,
    conversation: RagConversation,
    effective_kb_id: Optional[int],
    answer: str,
    citations: list[CitationResult],
) -> None:
    try:
        record_rag_interview_diagnostic(
            db,
            user_id=int(user_id),
            query=body.query,
            answer=answer,
            citations=citations,
            kb_id=effective_kb_id,
            document_id=body.document_id,
            conversation_id=conversation.id,
        )
    except Exception:
        logger.exception("failed to persist RAG interview diagnostic learning outputs")


class _StepTimer:
    def __init__(self) -> None:
        self.timings_ms: dict[str, float] = {}
        self._last_mark = time.perf_counter()

    def mark(self, name: str) -> None:
        now = time.perf_counter()
        self.timings_ms[name] = round((now - self._last_mark) * 1000, 2)
        self._last_mark = now


def _rewrite_query_for_retrieval(
    question: str,
    history: str,
    memory: str,
    task_state: str,
    enabled: bool,
) -> tuple[str, list[dict]]:
    if not enabled:
        return question, []
    try:
        return get_query_rewriter().rewrite(
            question=question,
            history=history,
            memory=memory,
            task_state=task_state,
        ), []
    except Exception:
        logger.exception("query rewrite failed; falling back to original query")
        return question, [
            warning_payload(
                ErrorCode.RAG_QUERY_REWRITE_DEGRADED,
                "Query rewrite failed; using the original question for retrieval.",
                retryable=True,
            )
        ]


def _retrieve_records(
    db: Session,
    search_query: str,
    *,
    top_k: int,
    document_id: Optional[int],
    kb_id: Optional[int],
    user_id: int,
    bm25_weight: float,
    warnings: list[dict],
    timer: _StepTimer,
) -> list[dict]:
    personal_space_only = kb_id is None and document_id is None
    retrieval_user_id = user_id if personal_space_only else None
    try:
        embedder = get_embedder()
        if bm25_weight > 0:
            try:
                embedder.ensure_bm25(
                    db,
                    user_id=retrieval_user_id,
                    document_id=document_id,
                    kb_id=kb_id,
                    personal_space_only=personal_space_only,
                )
            except Exception:
                bm25_weight = 0.0
                warnings.append(
                    warning_payload(
                        ErrorCode.RAG_BM25_DEGRADED,
                        "BM25 index is unavailable; vector retrieval was used.",
                        retryable=True,
                    )
                )
        timer.mark("bm25_ms")
        results = Retriever(embedder).retrieve(
            query=search_query,
            top_k=top_k,
            document_id=document_id,
            user_id=retrieval_user_id,
            kb_id=kb_id,
            personal_space_only=personal_space_only,
            bm25_weight=bm25_weight,
        )
        results = attach_parent_contexts(db, results)
        timer.mark("retrieve_ms")
        return results
    except Exception as exc:
        raise AppError(
            code=ErrorCode.RAG_RETRIEVAL_UNAVAILABLE,
            message="Knowledge retrieval is temporarily unavailable",
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            retryable=True,
            details={"error": str(exc)},
        ) from exc


def _generate_answer(
    query: str,
    context_state: ConversationContext,
    interview_diagnostic: bool,
    evidence: str,
    records,
    warnings: list[dict],
    on_token: Optional[Callable[[str], None]],
):
    if not records:
        return "No sufficient information was found in the knowledge base to answer this question.", [], False
    try:
        answerer = get_rag_answerer()
        answer_question = (
            build_interview_capability_query(query)
            if interview_diagnostic
            else query
        )
        generation_args = {
            "question": answer_question,
            "context": evidence,
            "history": context_state.history,
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
        return answer, response_records, False
    except Exception:
        warnings.append(
            warning_payload(
                ErrorCode.RAG_GENERATION_DEGRADED,
                "Answer generation failed; extractive fallback was used.",
                retryable=True,
            )
        )
        answer, response_records = extractive_fallback(records)
        return answer, response_records, True

def run_rag_answer(
    db: Session,
    username: str,
    body: AnswerRequest,
    on_token: Optional[Callable[[str], None]] = None,
) -> AnswerResponse:
    request_started = time.perf_counter()

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

    interview_diagnostic = is_interview_capability_request(
        body.query,
        has_material_scope=bool(effective_kb_id or body.document_id),
    )
    context_state = build_conversation_context(db, conversation)
    conversation = _ensure_conversation(db, user.id, effective_kb_id, body.query, conversation)
    _persist_user_message(db, user.id, conversation, body)

    timings_ms = {"setup_ms": _ms_since(request_started)}
    timer = _StepTimer()
    warnings: list[dict] = []

    rewritten_query, rewrite_warnings = _rewrite_query_for_retrieval(
        question=body.query,
        history=context_state.history,
        memory="",
        task_state=context_state.task_state,
        enabled=had_conversation and body.rewrite_query and RAG_QUERY_REWRITE_ENABLED,
    )
    warnings.extend(rewrite_warnings)
    timer.mark("rewrite_ms")

    retrieval_input = rewritten_query
    if interview_diagnostic:
        retrieval_input = (
            f"{rewritten_query} resume JD role fit project communication knowledge depth "
            "follow-up system design architecture performance reliability"
        )
    results = _retrieve_records(
        db,
        retrieval_input,
        top_k=body.top_k,
        document_id=body.document_id,
        kb_id=effective_kb_id,
        user_id=user.id,
        bm25_weight=body.bm25_weight,
        warnings=warnings,
        timer=timer,
    )
    evidence, records = build_evidence(results)
    timer.mark("evidence_ms")

    answer, response_records, degraded = _generate_answer(
        body.query,
        context_state,
        interview_diagnostic,
        evidence,
        records,
        warnings,
        on_token,
    )
    timer.mark("answer_ms")

    citations = [CitationResult(**record.citation_dict()) for record in response_records]
    retrieved_sources = [RetrievedSourceResult(**record.as_dict()) for record in records]
    timings_ms.update(timer.timings_ms)

    persist_started = time.perf_counter()
    _persist_assistant_message(db, user.id, conversation, answer, citations, degraded)
    if interview_diagnostic:
        _record_interview_diagnostic(db, user.id, body, conversation, effective_kb_id, answer, citations)
    timings_ms["persist_ms"] = _ms_since(persist_started)
    timings_ms["total_ms"] = _ms_since(request_started)
    logger.info("rag timings query=%r timings_ms=%s", body.query, timings_ms)

    return AnswerResponse(
        query=body.query,
        rewritten_query=rewritten_query if rewritten_query != body.query else None,
        answer=answer,
        conversation_id=conversation.id,
        citations=citations,
        retrieved_contexts=[record.context for record in records],
        retrieved_sources=retrieved_sources,
        retrieved_count=len(results),
        degraded=degraded,
        warnings=warnings,
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
        except AppError as exc:
            try:
                emit("error", {"status_code": exc.status_code, "detail": exc.to_detail().model_dump(mode="json")})
            except _ClientDisconnected:
                pass
        except HTTPException as exc:
            try:
                emit("error", {"status_code": exc.status_code, "detail": exc.detail})
            except _ClientDisconnected:
                pass
        except Exception:
            logger.exception("streaming RAG answer failed")
            try:
                emit("error", {
                    "status_code": 500,
                    "detail": {
                        "code": ErrorCode.RAG_STREAM_FAILED,
                        "message": "Streaming RAG answer failed",
                        "request_id": "-",
                        "retryable": True,
                        "details": {},
                    },
                })
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
