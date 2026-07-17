from __future__ import annotations

import concurrent.futures
import logging
import time
from dataclasses import asdict
from typing import Any, Callable

from app.core.config import AGENT_TOOL_TIMEOUT_SECONDS
from app.core.database import SessionLocal
from app.observability import trace_span

logger = logging.getLogger(__name__)

from app.memory.service import (
    build_context_for_state,
    consolidate_task_memory,
    read_profile_for_user,
    record_memory_event,
    summarize_task_session,
)
from app.schemas.rag import AnswerRequest
from app.services.rag_service import run_rag_answer

from .schemas import ToolDescriptor, ToolResult
from .tool_permissions import ToolPermissionError, assert_tool_allowed


ToolHandler = Callable[[dict[str, Any]], ToolResult]


def _call_handler(name: str, args: dict[str, Any], handler: ToolHandler) -> ToolResult:
    _validate_tool_args(name, args)
    return handler(args)


def _validate_tool_args(name: str, args: dict[str, Any]) -> None:
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object.")
    if name in {"knowledge.answer", "knowledge.repair_retrieval"}:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Knowledge tools require a non-empty query.")
        if len(query) > 2000:
            raise ValueError("Knowledge query exceeds 2000 characters.")
        top_k = int(args.get("top_k") or 5)
        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20.")
    if name.startswith("memory.") and args.get("user_id") is not None:
        int(args["user_id"])


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, ToolHandler] = {}
        self._descriptors: dict[str, ToolDescriptor] = {}

    def register(self, descriptor: ToolDescriptor, handler: ToolHandler) -> None:
        self._descriptors[descriptor.name] = descriptor
        self._handlers[descriptor.name] = handler

    def describe(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._descriptors.values()]

    def call(self, name: str, args: dict[str, Any], *, agent: str) -> ToolResult:
        try:
            assert_tool_allowed(agent, name)
        except ToolPermissionError as exc:
            return _tool_error(name, "tool_permission_denied", str(exc), retryable=False)

        descriptor = self._descriptors.get(name)
        handler = self._handlers.get(name)
        if handler is None:
            return _tool_error(name, "tool_missing", f"Tool {name} is not registered.", retryable=False)

        max_retries = 1 if (descriptor and descriptor.retryable) else 0
        timeout = (descriptor.timeout_seconds if descriptor else AGENT_TOOL_TIMEOUT_SECONDS)

        last_error: str | None = None
        for attempt in range(max_retries + 1):
            result = self._execute_with_timeout(name, args, handler, timeout)
            if result["ok"]:
                return result
            # Only retry when the error is explicitly marked retryable.
            error_info = result.get("error") or {}
            if not error_info.get("retryable", True):
                return result
            last_error = error_info.get("message") or error_info.get("type", "unknown")
            if attempt < max_retries:
                logger.debug("tool %s retry %d/%d: %s", name, attempt + 1, max_retries, last_error)

        result = _tool_error(name, "tool_exhausted_retries", last_error or "Tool failed after retries.", retryable=False)
        result["latency_ms"] = 0
        return result

    @staticmethod
    def _execute_with_timeout(
        name: str, args: dict[str, Any], handler: ToolHandler, timeout: float,
    ) -> ToolResult:
        started = time.perf_counter()
        with trace_span(f"tool.{name}", kind="CLIENT", attrs={"tool.name": name}) as span:
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(_call_handler, name, args, handler)
                    result = future.result(timeout=max(1.0, timeout))
            except concurrent.futures.TimeoutError:
                elapsed = round((time.perf_counter() - started) * 1000)
                if span:
                    span.set_attribute("error", "timeout")
                return {**_tool_error(name, "tool_timeout", f"Tool exceeded {timeout:.0f}s timeout.", retryable=True), "latency_ms": elapsed}
            except Exception as exc:  # noqa: BLE001
                elapsed = round((time.perf_counter() - started) * 1000)
                if span:
                    span.set_attribute("error", "true")
                    span.set_attribute("error.message", str(exc)[:256])
                return {**_tool_error(name, "tool_exception", str(exc), retryable=True), "latency_ms": elapsed}
            result["latency_ms"] = round((time.perf_counter() - started) * 1000)
            if span:
                span.set_attribute("ok", str(result["ok"]))
            return result


def _tool_error(name: str, error_type: str, message: str, *, retryable: bool) -> ToolResult:
    return {
        "ok": False,
        "tool_name": name,
        "data": {},
        "confidence": 0.0,
        "citations": [],
        "grounding": {"mode": error_type, "rag_used": False},
        "trace": [],
        "error": {"type": error_type, "message": message, "retryable": retryable},
    }


def memory_read_profile(args: dict[str, Any]) -> ToolResult:
    profile = read_profile_for_user(args.get("user_id"))
    return {
        "ok": True,
        "tool_name": "memory.read_profile",
        "data": profile,
        "confidence": 1.0,
        "citations": [],
        "grounding": {"mode": "memory", "rag_used": False},
        "trace": [],
        "error": None,
    }


def memory_read_context(args: dict[str, Any]) -> ToolResult:
    # A model must never be able to smuggle another user's nested `state` into
    # the memory gateway.  The runtime injects these trusted scope fields.
    context = build_context_for_state({
        key: args.get(key)
        for key in ("user_id", "session_id", "task_id", "run_id")
    })
    return {
        "ok": True,
        "tool_name": "memory.read_context",
        "data": context,
        "confidence": 1.0,
        "citations": [],
        "grounding": {"mode": "memory_context", "rag_used": False, "memory_used": True},
        "trace": [{"step": "build_agent_context", "profile_items": context.get("profile_item_count", 0)}],
        "error": None,
    }


def memory_write_event(args: dict[str, Any]) -> ToolResult:
    row = record_memory_event(
        user_id=args.get("user_id"), session_id=args.get("session_id"), task_id=args.get("task_id"),
        event_type=str(args.get("event_type") or "agent_observation"), category=args.get("category"),
        content=str(args.get("content") or ""), source=str(args.get("source") or "agent"),
        metadata=args.get("metadata") or {},
    )
    return {
        "ok": row is not None, "tool_name": "memory.write_event", "data": row or {},
        "confidence": 1.0 if row else 0.0, "citations": [],
        "grounding": {"mode": "memory_write", "rag_used": False}, "trace": [],
        "error": None if row else {"type": "memory_write_skipped", "retryable": False, "message": "No user memory event was persisted."},
    }


def memory_consolidate(args: dict[str, Any]) -> ToolResult:
    records = consolidate_task_memory(args.get("state") or args)
    return {
        "ok": True, "tool_name": "memory.consolidate", "data": {"records": records, "count": len(records)},
        "confidence": 1.0, "citations": [], "grounding": {"mode": "memory_consolidation", "rag_used": False},
        "trace": [], "error": None,
    }


def memory_summarize_session(args: dict[str, Any]) -> ToolResult:
    summary = summarize_task_session(args.get("state") or args)
    return {
        "ok": summary is not None, "tool_name": "memory.summarize_session", "data": summary or {},
        "confidence": 1.0 if summary else 0.0, "citations": [],
        "grounding": {"mode": "session_summary", "rag_used": False}, "trace": [],
        "error": None if summary else {"type": "session_summary_skipped", "retryable": False, "message": "No session data was available to summarize."},
    }


def knowledge_answer(args: dict[str, Any]) -> ToolResult:
    db = SessionLocal()
    try:
        request = AnswerRequest(
            query=str(args.get("query") or ""),
            top_k=int(args.get("top_k") or 5),
            document_id=args.get("document_id"),
            kb_id=args.get("kb_id"),
            conversation_id=args.get("conversation_id"),
            use_memory=bool(args.get("use_memory", True)),
            rewrite_query=bool(args.get("rewrite_query", True)),
        )
        response = run_rag_answer(db, str(args.get("username") or "admin"), request)
        citations = [item.model_dump() for item in response.citations]
        estimated_tokens = max(1, len(request.query + response.answer + "".join(response.retrieved_contexts)) // 4)
        retrieved_count = response.retrieved_count
        source_count = len({str(item.get("document_id") or item.get("source") or index) for index, item in enumerate(citations)})
        retrieval_status = "grounded" if citations else ("retrieved_without_citation" if retrieved_count else "empty")
        return {
            "ok": retrieved_count > 0,
            "tool_name": "knowledge.answer",
            "data": response.model_dump(),
            "confidence": 0.85 if citations else (0.55 if retrieved_count else 0.2),
            "citations": citations,
            "grounding": {
                "mode": "rag_grounded" if citations else "insufficient_evidence",
                "rag_used": True,
                "retrieved_count": retrieved_count,
                "source_count": source_count,
                "citation_count": len(citations),
                "retrieval_status": retrieval_status,
                "query": request.query,
                "rewritten_query": getattr(response, "rewritten_query", None),
                "context_compacted": bool(getattr(response, "context_compacted", False)),
            },
            "trace": [{"step": "run_rag_answer", "timings_ms": response.timings_ms, "retrieval_status": retrieval_status}],
            "usage": {"total_tokens": estimated_tokens, "estimated": True},
            "error": None if retrieved_count > 0 else {
                "type": "retrieval_empty",
                "retryable": True,
                "message": "RAG returned no retrieved context.",
            },
        }
    finally:
        db.close()


REPAIR_STRATEGIES: dict[str, dict[str, Any]] = {
    "rewrite_query": {
        "description": "重写查询增加关键词覆盖",
        "query_suffix": "证据 引用 项目文档 实现细节",
        "top_k_multiplier": 1.5,
    },
    "expand_top_k": {
        "description": "扩大 top_k 获取更多候选",
        "query_suffix": "",
        "top_k_multiplier": 2.5,
    },
    "dense_only": {
        "description": "仅使用稠密向量检索",
        "query_suffix": "",
        "top_k_multiplier": 1.0,
        "bm25_weight": 0.0,
    },
    "sparse_only": {
        "description": "仅使用 BM25 关键词检索",
        "query_suffix": "",
        "top_k_multiplier": 1.0,
        "bm25_weight": 1.0,
    },
    "hybrid_boost": {
        "description": "同时提升稠密和稀疏权重",
        "query_suffix": "",
        "top_k_multiplier": 2.0,
        "bm25_weight": 0.5,
    },
}


def knowledge_repair_retrieval(args: dict[str, Any]) -> ToolResult:
    """Multi-strategy retrieval repair.

    Supports strategy selection via ``repair_strategy`` arg:
    - ``rewrite_query`` (default): append keywords to query + moderate top_k bump
    - ``expand_top_k``: aggressive top_k expansion
    - ``dense_only`` / ``sparse_only`` / ``hybrid_boost``: switch retrieval mode

    When no strategy is specified, falls back to the original rewrite approach.
    """
    strategy_name = str(args.get("repair_strategy") or "rewrite_query")
    strategy = REPAIR_STRATEGIES.get(strategy_name, REPAIR_STRATEGIES["rewrite_query"])

    query = str(args.get("query") or "")
    original_top_k = int(args.get("top_k") or 5)
    suffix = str(strategy.get("query_suffix") or "")
    if suffix and suffix not in query:
        query = f"{query} {suffix}"

    top_k = min(20, max(3, int(original_top_k * float(strategy.get("top_k_multiplier", 1.5)))))

    repaired: dict[str, Any] = {**args, "query": query, "top_k": top_k}

    # Apply retrieval-mode overrides.
    if "bm25_weight" in strategy:
        repaired["bm25_weight"] = float(strategy["bm25_weight"])

    result = knowledge_answer(repaired)
    result["tool_name"] = "knowledge.repair_retrieval"
    result.setdefault("grounding", {})["repair_strategy"] = strategy_name
    result.setdefault("grounding", {})["repair_reason"] = args.get("repair_reason") or "retrieval_insufficient"
    result.setdefault("trace", []).append({
        "step": "retrieval_repair",
        "strategy": strategy_name,
        "top_k": top_k,
        "original_top_k": original_top_k,
    })
    return result


def knowledge_verify_claim(args: dict[str, Any]) -> ToolResult:
    knowledge = args.get("s2") or {}
    citations = knowledge.get("citations") or knowledge.get("data", {}).get("citations") or []
    grounding = knowledge.get("grounding") or {}
    issues: list[dict[str, Any]] = []
    if not citations:
        issues.append({"type": "citation_missing", "message": "Key claims have no RAG citations.", "retryable": True})
    if int(grounding.get("retrieved_count") or 0) == 0:
        issues.append({"type": "retrieval_empty", "message": "No retrievable context supports this claim.", "retryable": True})
    elif int(grounding.get("source_count") or 0) < 1:
        issues.append({"type": "evidence_weak", "message": "Retrieved context has no traceable source.", "retryable": True})
    ok = not issues
    return {
        "ok": ok,
        "tool_name": "knowledge.verify_claim",
        "data": {
            "status": "passed" if ok else "insufficient_evidence",
            "issues": issues,
        },
        "confidence": 0.84 if ok else 0.35,
        "citations": citations,
        "grounding": {"mode": "rag_grounded" if ok else "insufficient_evidence", "rag_used": ok},
        "trace": [],
        "error": None if ok else {"type": issues[0]["type"], "retryable": True, "message": issues[0]["message"]},
    }


registry = ToolRegistry()
registry.register(
    ToolDescriptor("memory.read_context", "memory", "Build long-term, session, and task memory context for planning.", {"type": "object"}, retryable=False, timeout_seconds=10.0),
    memory_read_context,
)
registry.register(
    ToolDescriptor("memory.write_event", "memory", "Persist a memory event for later profile consolidation.", {"type": "object"}, retryable=False, timeout_seconds=5.0),
    memory_write_event,
)
registry.register(
    ToolDescriptor("memory.consolidate", "memory", "Consolidate eligible events into durable user memory.", {"type": "object"}, retryable=False, timeout_seconds=15.0),
    memory_consolidate,
)
registry.register(
    ToolDescriptor("memory.summarize_session", "memory", "Summarize this task session for the next plan.", {"type": "object"}, retryable=False, timeout_seconds=15.0),
    memory_summarize_session,
)
registry.register(
    ToolDescriptor("memory.read_profile", "memory", "读取用户目标和薄弱点。", {"type": "object"}, retryable=False, timeout_seconds=5.0),
    memory_read_profile,
)
registry.register(
    ToolDescriptor("knowledge.answer", "rag", "调用原 RAG run_rag_answer 获取带引用答案。", {"type": "object"}, retryable=True, timeout_seconds=75.0),
    knowledge_answer,
)
registry.register(
    ToolDescriptor("knowledge.repair_retrieval", "rag", "检索为空或引用不足时重写 query 并扩大 top_k。", {"type": "object"}, retryable=True, timeout_seconds=75.0),
    knowledge_repair_retrieval,
)
registry.register(
    ToolDescriptor("knowledge.verify_claim", "verifier", "检查输出是否有 RAG citations 支撑。", {"type": "object"}, retryable=False, timeout_seconds=10.0),
    knowledge_verify_claim,
)


def call_tool(name: str, args: dict[str, Any], *, agent: str) -> ToolResult:
    return registry.call(name, args, agent=agent)


def list_tools() -> list[dict[str, Any]]:
    return registry.describe()
