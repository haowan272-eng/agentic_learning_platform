from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import time
from dataclasses import asdict
from typing import Any, Callable
from uuid import uuid4

import httpx
from ddgs import DDGS

from app.core.config import AGENT_TOOL_TIMEOUT_SECONDS, GITHUB_CACHE_TTL_SECONDS, GITHUB_RETRY_ATTEMPTS, GITHUB_TOKEN
from app.core.database import SessionLocal
from app.core.redis import get_redis
from app.observability import trace_span

logger = logging.getLogger(__name__)

from app.schemas.rag import AnswerRequest
from app.services.rag_service import run_rag_answer

from .scenarios import get_scenario, list_scenarios
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
    if name == "github.search_repositories":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("GitHub search requires a non-empty query.")
        if len(query) > 512:
            raise ValueError("GitHub search query exceeds 512 characters.")
        limit = int(args.get("limit") or 3)
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10.")
    if name == "github.read_readme":
        repo = args.get("repo")
        if not isinstance(repo, str) or not repo.strip() or "/" not in repo:
            raise ValueError("github.read_readme requires repo in owner/name format.")
        if len(repo) > 255:
            raise ValueError("repo exceeds 255 characters.")
    if name == "web.search_duckduckgo":
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("DuckDuckGo search requires a non-empty query.")
        if len(query) > 512:
            raise ValueError("DuckDuckGo search query exceeds 512 characters.")
        limit = int(args.get("limit") or 5)
        if not 1 <= limit <= 10:
            raise ValueError("limit must be between 1 and 10.")
    if name == "learning.scenario_blueprint":
        key = args.get("scenario_key")
        if key is not None and (not isinstance(key, str) or len(key) > 128):
            raise ValueError("scenario_key must be a string under 128 characters.")
    if name == "architecture.generate_proposal":
        artifacts = args.get("artifacts")
        if artifacts is not None and not isinstance(artifacts, list):
            raise ValueError("architecture.generate_proposal requires artifacts to be a list.")
    if name == "verification.verify_proposal":
        proposal = args.get("proposal")
        artifacts = args.get("artifacts")
        if not isinstance(proposal, dict):
            raise ValueError("verification.verify_proposal requires a proposal object.")
        if artifacts is not None and not isinstance(artifacts, list):
            raise ValueError("verification.verify_proposal requires artifacts to be a list.")
    if name == "planning.repair_research_tasks":
        verification = args.get("verification")
        if not isinstance(verification, dict):
            raise ValueError("planning.repair_research_tasks requires a verification object.")


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
        "description": "Rewrite query with evidence-oriented keywords.",
        "query_suffix": "evidence citations project docs implementation details",
        "top_k_multiplier": 1.5,
    },
    "expand_top_k": {
        "description": "Increase top_k to retrieve more candidates.",
        "query_suffix": "",
        "top_k_multiplier": 2.5,
    },
    "dense_only": {
        "description": "Use dense vector retrieval only.",
        "query_suffix": "",
        "top_k_multiplier": 1.0,
        "bm25_weight": 0.0,
    },
    "sparse_only": {
        "description": "Use sparse BM25 retrieval only.",
        "query_suffix": "",
        "top_k_multiplier": 1.0,
        "bm25_weight": 1.0,
    },
    "hybrid_boost": {
        "description": "Boost both dense and sparse retrieval signals.",
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


def architecture_generate_proposal(args: dict[str, Any]) -> ToolResult:
    from .planner import generate_proposal

    state = {
        "session_id": args.get("session_id") or "",
        "task_id": args.get("task_id") or "",
        "run_id": args.get("run_id") or "",
        "username": args.get("username") or "",
        "user_id": args.get("user_id"),
        "user_input": str(args.get("user_input") or ""),
        "task_type": str(args.get("task_type") or "project_upgrade"),
        "scenario": args.get("scenario") or {},
        "memory_context": args.get("memory_context") or {},
        "budget": args.get("budget") or {},
        "artifacts": args.get("artifacts") or [],
    }
    proposal, error, source = generate_proposal(state)
    payload = proposal.model_dump(mode="json")
    citations = [citation for item in state["artifacts"] for citation in item.get("citations", [])]
    return {
        "ok": error is None,
        "tool_name": "architecture.generate_proposal",
        "data": payload,
        "confidence": 0.8 if error is None else 0.45,
        "citations": citations,
        "grounding": {"mode": "proposal_generation", "rag_used": bool(citations), "model_source": source},
        "trace": [{"step": "generate_proposal", "source": source, "artifact_count": len(state["artifacts"])}],
        "error": None if error is None else {"type": "proposal_degraded", "message": error, "retryable": True},
    }


def verification_verify_proposal(args: dict[str, Any]) -> ToolResult:
    from .planner import verify_proposal

    state = {
        "session_id": args.get("session_id") or "",
        "task_id": args.get("task_id") or "",
        "run_id": args.get("run_id") or "",
        "username": args.get("username") or "",
        "user_id": args.get("user_id"),
        "kb_id": args.get("kb_id"),
        "user_input": str(args.get("user_input") or ""),
        "proposal": args.get("proposal") or {},
        "artifacts": args.get("artifacts") or [],
        "citations": args.get("citations") or [],
        "grounding": args.get("grounding") or {},
        "budget": args.get("budget") or {},
    }
    decision, error, source = verify_proposal(state)
    payload = decision.model_dump(mode="json")
    payload["error"] = error
    citations = [citation for item in state["artifacts"] for citation in item.get("citations", [])]
    ok = decision.status == "passed"
    return {
        "ok": ok,
        "tool_name": "verification.verify_proposal",
        "data": payload,
        "confidence": float(decision.score),
        "citations": citations,
        "grounding": {"mode": "verification", "rag_used": bool(citations), "model_source": source},
        "trace": [{"step": "verify_proposal", "source": source, "status": decision.status}],
        "error": None if ok else {
            "type": str((decision.issues[0] if decision.issues else {}).get("type", decision.status)),
            "message": str((decision.issues[0] if decision.issues else {}).get("message", "Proposal verification did not pass.")),
            "retryable": decision.status in {"repair", "needs_approval"},
        },
    }


def planning_repair_research_tasks(args: dict[str, Any]) -> ToolResult:
    from .planner import ResearchTask

    verification = args.get("verification") or {}
    queries = verification.get("repair_queries") or [args.get("user_input", "")]
    tasks = [
        ResearchTask(
            task_id=f"repair-{index + 1}-{uuid4().hex[:6]}",
            query=str(query),
            objective="Supplement evidence requested by verification.",
            top_k=8,
        ).model_dump(mode="json")
        for index, query in enumerate(queries[:3])
        if str(query).strip()
    ]
    return {
        "ok": bool(tasks),
        "tool_name": "planning.repair_research_tasks",
        "data": {"research_tasks": tasks, "repair_count": int(args.get("repair_count") or 0) + 1},
        "confidence": 0.7 if tasks else 0.0,
        "citations": [],
        "grounding": {"mode": "repair_planning", "rag_used": False},
        "trace": [{"step": "repair_research_tasks", "query_count": len(tasks)}],
        "error": None if tasks else {"type": "repair_query_missing", "message": "No repair query was available.", "retryable": False},
    }


def learning_scenario_blueprint(args: dict[str, Any]) -> ToolResult:
    scenario_key = str(args.get("scenario_key") or "").strip()
    if scenario_key:
        scenario = get_scenario(scenario_key)
        if not scenario:
            return _tool_error(
                "learning.scenario_blueprint",
                "scenario_missing",
                f"Scenario {scenario_key} is not registered.",
                retryable=False,
            )
        payload: dict[str, Any] = {"scenario_key": scenario_key, "scenario": scenario}
    else:
        payload = {"scenarios": list_scenarios()}
    return {
        "ok": True,
        "tool_name": "learning.scenario_blueprint",
        "data": payload,
        "confidence": 1.0,
        "citations": [],
        "grounding": {"mode": "business_scenario_catalog", "rag_used": False},
        "trace": [{"step": "load_scenario_blueprint", "scenario_key": scenario_key or "*"}],
        "error": None,
    }


def web_search_duckduckgo(args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    limit = int(args.get("limit") or 5)
    try:
        rows = list(DDGS().text(query, max_results=limit))
    except Exception as exc:  # noqa: BLE001
        logger.warning("DuckDuckGo search failed: %s", exc)
        return {
            **_tool_error(
                "web.search_duckduckgo",
                "duckduckgo_search_failed",
                str(exc),
                retryable=True,
            ),
            "data": {"query": query, "results": []},
        }

    results: list[dict[str, str]] = []
    seen_urls: set[str] = set()
    for item in rows:
        url = str(item.get("href") or item.get("url") or "").strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        results.append({
            "title": str(item.get("title") or "").strip()[:300],
            "url": url,
            "snippet": str(item.get("body") or item.get("snippet") or "").strip()[:1200],
        })
        if len(results) >= limit:
            break

    citations = [
        {
            "source_id": index,
            "title": item["title"],
            "url": item["url"],
            "snippet": item["snippet"],
            "source": "duckduckgo",
        }
        for index, item in enumerate(results, start=1)
    ]
    return {
        "ok": True,
        "tool_name": "web.search_duckduckgo",
        "data": {"query": query, "results": results},
        "confidence": 0.7 if results else 0.0,
        "citations": citations,
        "grounding": {
            "mode": "web_search",
            "rag_used": False,
            "external_source": "duckduckgo",
            "result_count": len(results),
        },
        "trace": [{"step": "duckduckgo.text", "query": query, "result_count": len(results)}],
        "error": None if results else {
            "type": "duckduckgo_empty",
            "message": "DuckDuckGo returned no results.",
            "retryable": False,
        },
    }


GITHUB_API_HEADERS = {
    "Accept": "application/vnd.github+json",
    "X-GitHub-Api-Version": "2022-11-28",
}
GITHUB_RAW_HEADERS = {
    "Accept": "application/vnd.github.raw+json",
    "X-GitHub-Api-Version": "2022-11-28",
}


def _github_headers(*, raw: bool = False, token: str | None = None) -> dict[str, str]:
    headers = dict(GITHUB_RAW_HEADERS if raw else GITHUB_API_HEADERS)
    resolved_token = token if token is not None else GITHUB_TOKEN
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"
    return headers


def _github_cache_key(tool: str, payload: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()
    return f"agent_tool:{tool}:{digest}"


def _github_cache_get(key: str) -> dict[str, Any] | None:
    client = get_redis()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        return json.loads(raw)
    except Exception as exc:  # noqa: BLE001
        logger.debug("github cache read skipped: %s", exc)
        return None


def _github_cache_set(key: str, value: dict[str, Any]) -> None:
    client = get_redis()
    if client is None:
        return
    try:
        client.setex(key, GITHUB_CACHE_TTL_SECONDS, json.dumps(value, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        logger.debug("github cache write skipped: %s", exc)


def _github_get(url: str, *, params: dict[str, Any] | None = None, raw: bool = False, token: str | None = None) -> httpx.Response:
    last_exc: Exception | None = None
    for attempt in range(GITHUB_RETRY_ATTEMPTS):
        try:
            response = httpx.get(url, headers=_github_headers(raw=raw, token=token), params=params, timeout=15.0)
            if response.status_code not in {429, 500, 502, 503, 504}:
                return response
        except httpx.HTTPError as exc:
            last_exc = exc
        if attempt < GITHUB_RETRY_ATTEMPTS - 1:
            time.sleep(min(2.0, 0.25 * (2**attempt)))
    if last_exc:
        raise last_exc
    return response


def _github_repo_payload(item: dict[str, Any], *, query: str) -> dict[str, Any]:
    owner = item.get("owner") or {}
    return {
        "full_name": item.get("full_name"),
        "name": item.get("name"),
        "owner": owner.get("login"),
        "description": item.get("description"),
        "url": item.get("html_url"),
        "html_url": item.get("html_url"),
        "stars": item.get("stargazers_count") or 0,
        "stargazers_count": item.get("stargazers_count") or 0,
        "forks": item.get("forks_count") or 0,
        "language": item.get("language"),
        "topics": item.get("topics") or [],
        "updated_at": item.get("updated_at"),
        "license": (item.get("license") or {}).get("spdx_id"),
        "source": "github",
        "query": query,
    }


def _github_search_result(
    *, query: str, repositories: list[dict[str, Any]], auth_mode: str, cache_hit: bool, error: dict[str, Any] | None = None,
) -> ToolResult:
    return {
        "ok": bool(repositories) and error is None,
        "tool_name": "github.search_repositories",
        "data": {"query": query, "repositories": repositories, "auth_mode": auth_mode, "cache_hit": cache_hit},
        "confidence": 0.82 if repositories else 0.2,
        "citations": [
            {"source_id": index + 1, "repo": item.get("full_name"), "url": item.get("url")}
            for index, item in enumerate(repositories)
        ],
        "grounding": {
            "mode": "github_search",
            "rag_used": False,
            "external_source": "github",
            "result_count": len(repositories),
            "auth_mode": auth_mode,
            "cache_hit": cache_hit,
        },
        "trace": [{"step": "github.search_repositories", "query": query, "result_count": len(repositories)}],
        "error": error,
    }


def github_search_repositories(args: dict[str, Any]) -> ToolResult:
    query = str(args.get("query") or "").strip()
    limit = min(10, max(1, int(args.get("limit") or 3)))
    sort = str(args.get("sort") or "stars").strip() or "stars"
    cache_key = _github_cache_key("search_repositories", {"query": query, "limit": limit, "sort": sort})
    cached = _github_cache_get(cache_key)
    if cached:
        return _github_search_result(query=query, repositories=cached.get("repositories") or [], auth_mode=cached.get("auth_mode") or "cache", cache_hit=True)

    auth_mode = "token" if GITHUB_TOKEN else "public"
    try:
        params = {"q": query, "sort": sort, "order": "desc", "per_page": limit}
        response = _github_get("https://api.github.com/search/repositories", params=params, token=GITHUB_TOKEN or None)
        if response.status_code in {401, 403} and GITHUB_TOKEN:
            logger.warning("GitHub token search failed, retrying public API")
            response = _github_get("https://api.github.com/search/repositories", params=params, token="")
            auth_mode = "public"
        response.raise_for_status()

        payload = response.json()
        repositories = [_github_repo_payload(item, query=query) for item in (payload.get("items") or [])[:limit]]
        _github_cache_set(cache_key, {"repositories": repositories, "auth_mode": auth_mode})
        error = None if repositories else {"type": "github_empty", "retryable": False, "message": "GitHub returned no repositories for this query."}
        return _github_search_result(query=query, repositories=repositories, auth_mode=auth_mode, cache_hit=False, error=error)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GitHub search failed: %s", exc)
        return _github_search_result(
            query=query,
            repositories=[],
            auth_mode=auth_mode,
            cache_hit=False,
            error={"type": "github_search_failed", "retryable": False, "message": str(exc)},
        )


def _github_readme_result(
    *, repo: str, readme: str, auth_mode: str, cache_hit: bool, error: dict[str, Any] | None = None,
) -> ToolResult:
    return {
        "ok": bool(readme) and error is None,
        "tool_name": "github.read_readme",
        "data": {"repo": repo, "url": f"https://github.com/{repo}", "readme": readme, "auth_mode": auth_mode, "cache_hit": cache_hit},
        "confidence": 0.86 if readme else 0.2,
        "citations": [{"source_id": 1, "repo": repo, "url": f"https://github.com/{repo}"}] if readme else [],
        "grounding": {
            "mode": "github_readme",
            "rag_used": False,
            "external_source": "github",
            "repo": repo,
            "auth_mode": auth_mode,
            "cache_hit": cache_hit,
        },
        "trace": [{"step": "github.read_readme", "repo": repo, "readme_chars": len(readme)}],
        "error": error,
    }


def github_read_readme(args: dict[str, Any]) -> ToolResult:
    repo = str(args.get("repo") or "").strip()
    max_chars = min(30000, max(1000, int(args.get("max_chars") or 12000)))
    cache_key = _github_cache_key("read_readme", {"repo": repo, "max_chars": max_chars})
    cached = _github_cache_get(cache_key)
    if cached:
        return _github_readme_result(repo=repo, readme=str(cached.get("readme") or ""), auth_mode=cached.get("auth_mode") or "cache", cache_hit=True)

    auth_mode = "token" if GITHUB_TOKEN else "public"
    try:
        url = f"https://api.github.com/repos/{repo}/readme"
        response = _github_get(url, raw=True, token=GITHUB_TOKEN or None)
        if response.status_code in {401, 403} and GITHUB_TOKEN:
            logger.warning("GitHub token README access failed, retrying public API")
            response = _github_get(url, raw=True, token="")
            auth_mode = "public"
        response.raise_for_status()

        readme = response.text[:max_chars]
        _github_cache_set(cache_key, {"readme": readme, "auth_mode": auth_mode})
        error = None if readme else {"type": "github_readme_empty", "retryable": False, "message": "GitHub README was empty."}
        return _github_readme_result(repo=repo, readme=readme, auth_mode=auth_mode, cache_hit=False, error=error)
    except Exception as exc:  # noqa: BLE001
        logger.warning("GitHub README read failed: %s", exc)
        return _github_readme_result(
            repo=repo,
            readme="",
            auth_mode=auth_mode,
            cache_hit=False,
            error={"type": "github_readme_failed", "retryable": False, "message": str(exc)},
        )


registry = ToolRegistry()
registry.register(
    ToolDescriptor(
        "learning.scenario_blueprint",
        "learning",
        "Read a scenario blueprint including agent flow, required inputs, tools, deliverables, and research task templates.",
        {
            "type": "object",
            "properties": {"scenario_key": {"type": "string"}},
        },
        retryable=False,
        timeout_seconds=5.0,
    ),
    learning_scenario_blueprint,
)
registry.register(
    ToolDescriptor(
        "web.search_duckduckgo",
        "web",
        "Search public web pages with DuckDuckGo and return source URLs and snippets.",
        {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
            },
        },
        retryable=True,
        timeout_seconds=20.0,
    ),
    web_search_duckduckgo,
)
registry.register(
    ToolDescriptor(
        "github.search_repositories",
        "github",
        "Search public GitHub repositories by query.",
        {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                "sort": {"type": "string"},
            },
        },
        retryable=False,
        timeout_seconds=20.0,
    ),
    github_search_repositories,
)
registry.register(
    ToolDescriptor(
        "github.read_readme",
        "github",
        "Read a repository README by owner/name.",
        {
            "type": "object",
            "required": ["repo"],
            "properties": {
                "repo": {"type": "string"},
                "max_chars": {"type": "integer", "minimum": 1000, "maximum": 30000},
            },
        },
        retryable=False,
        timeout_seconds=20.0,
    ),
    github_read_readme,
)
registry.register(
    ToolDescriptor("knowledge.answer", "rag", "Call RAG run_rag_answer and return an answer with citations.", {"type": "object"}, retryable=True, timeout_seconds=75.0),
    knowledge_answer,
)
registry.register(
    ToolDescriptor("knowledge.repair_retrieval", "rag", "Repair empty or weak retrieval by rewriting the query and increasing top_k.", {"type": "object"}, retryable=True, timeout_seconds=75.0),
    knowledge_repair_retrieval,
)
registry.register(
    ToolDescriptor("knowledge.verify_claim", "verification", "Check whether output is supported by RAG citations.", {"type": "object"}, retryable=False, timeout_seconds=10.0),
    knowledge_verify_claim,
)


registry.register(
    ToolDescriptor(
        "architecture.generate_proposal",
        "architecture",
        "Generate a structured proposal from research artifacts.",
        {
            "type": "object",
            "required": ["user_input", "artifacts"],
            "properties": {
                "user_input": {"type": "string"},
                "artifacts": {"type": "array"},
                "memory_context": {"type": "object"},
                "scenario": {"type": "object"},
            },
        },
        retryable=True,
        timeout_seconds=45.0,
    ),
    architecture_generate_proposal,
)
registry.register(
    ToolDescriptor(
        "verification.verify_proposal",
        "verification",
        "Verify whether a proposal is sufficiently supported by research evidence.",
        {
            "type": "object",
            "required": ["proposal", "artifacts"],
            "properties": {
                "proposal": {"type": "object"},
                "artifacts": {"type": "array"},
                "citations": {"type": "array"},
                "grounding": {"type": "object"},
            },
        },
        retryable=False,
        timeout_seconds=20.0,
    ),
    verification_verify_proposal,
)
registry.register(
    ToolDescriptor(
        "planning.repair_research_tasks",
        "planning",
        "Create focused follow-up research tasks from verification failures.",
        {
            "type": "object",
            "required": ["user_input", "verification"],
            "properties": {
                "user_input": {"type": "string"},
                "verification": {"type": "object"},
                "repair_count": {"type": "integer"},
            },
        },
        retryable=False,
        timeout_seconds=5.0,
    ),
    planning_repair_research_tasks,
)


def call_tool(name: str, args: dict[str, Any], *, agent: str) -> ToolResult:
    return registry.call(name, args, agent=agent)


def list_tools() -> list[dict[str, Any]]:
    return registry.describe()
