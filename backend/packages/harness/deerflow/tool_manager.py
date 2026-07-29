from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.observability import increment, observe_ms

from .schemas import Artifact, RuntimeStore, ToolResult
from .tools import call_tool


@dataclass(frozen=True)
class ManagedToolExecution:
    call_id: str
    tool_name: str
    agent_name: str
    skill_name: str
    step_id: str
    result: ToolResult
    artifact: Artifact


def _artifact_kind(tool_name: str) -> str:
    if tool_name.startswith("architecture."):
        return "proposal"
    if tool_name.startswith("knowledge.verify"):
        return "verification"
    if tool_name.startswith("verification."):
        return "verification"
    if tool_name.startswith("planning."):
        return "plan"
    if tool_name.startswith(("knowledge.", "web.")):
        return "research"
    if tool_name.startswith("learning."):
        return "learning_blueprint"
    return "tool_result"


def _trusted_args(state: dict[str, Any], tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    args = dict(arguments)
    if tool_name.startswith("knowledge."):
        args["username"] = state["username"]
        args["user_id"] = state.get("user_id")
        args["kb_id"] = state.get("kb_id")
        args["document_id"] = state.get("document_id")
        args["conversation_id"] = state.get("conversation_id")
        args.setdefault("top_k", 5)
        args.setdefault("rewrite_query", True)
    if tool_name.startswith(("architecture.", "verification.", "planning.")):
        args["user_input"] = state.get("user_input", "")
        args["task_id"] = state.get("task_id", "")
        args["run_id"] = state.get("run_id", "")
        args["username"] = state.get("username", "")
        args["user_id"] = state.get("user_id")
        args["kb_id"] = state.get("kb_id")
        args["memory_context"] = state.get("memory_context") or {}
        args["scenario"] = state.get("scenario") or {}
        args["budget"] = state.get("budget") or {}
    if tool_name.startswith("architecture."):
        args["artifacts"] = state.get("artifacts") or []
    if tool_name.startswith("verification."):
        args["proposal"] = state.get("proposal") or {}
        args["artifacts"] = state.get("artifacts") or []
        args["citations"] = state.get("citations") or []
        args["grounding"] = state.get("grounding") or {}
    if tool_name.startswith("planning."):
        args["verification"] = state.get("verification") or {}
        args["repair_count"] = state.get("repair_count") or 0
    return args


def execute_managed_tool(
    state: dict[str, Any],
    store: RuntimeStore,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    agent_name: str,
    skill_name: str,
    step_id: str | None = None,
    call_id: str | None = None,
) -> ManagedToolExecution:
    call_id = call_id or f"tool-{uuid4().hex[:12]}"
    step_id = step_id or call_id
    trusted_args = _trusted_args(state, tool_name, arguments)
    execution_key = f"{state['task_id']}:{call_id}"
    if len(execution_key) > 128:
        execution_key = f"tool-{hashlib.sha256(execution_key.encode('utf-8')).hexdigest()}"
    completed_call = getattr(store, "get_completed_tool_call", lambda _key: None)(execution_key)
    if completed_call:
        result = completed_call.get("result") or completed_call.get("output") or {}
    else:
        # Write-capable tools must forward this stable key to their downstream
        # provider. A worker retry can then safely reissue an ambiguous call.
        trusted_args["idempotency_key"] = execution_key
        result = call_tool(tool_name, trusted_args, agent=agent_name)
        latency_ms = float(result.get("latency_ms") or 0)
        observe_ms("agent_tool_latency_ms", latency_ms, {"tool": tool_name, "ok": bool(result.get("ok"))})
        increment("agent_tool_calls_total", {"tool": tool_name, "ok": bool(result.get("ok"))})

    artifact: Artifact = {
        "artifact_id": f"tool-{call_id}",
        "kind": _artifact_kind(tool_name),
        "producer": agent_name,
        "correlation_id": step_id,
        "data": result.get("data") or {},
        "citations": result.get("citations") or [],
        "confidence": float(result.get("confidence") or 0),
        "error": result.get("error"),
        "grounding": result.get("grounding") or {},
    }
    if not completed_call:
        store.save_tool_call({
            "task_id": state["task_id"],
            "run_id": state["run_id"],
            "step_id": step_id,
            "agent_name": agent_name,
            "skill_name": skill_name,
            "tool_name": tool_name,
            "idempotency_key": execution_key,
            "input": trusted_args,
            "output": result,
            "result": result,
            "ok": bool(result.get("ok")),
            "error_type": (result.get("error") or {}).get("type"),
            "error_message": (result.get("error") or {}).get("message"),
            "retry_count": 0,
            "latency_ms": result.get("latency_ms"),
        })
    return ManagedToolExecution(
        call_id=call_id,
        tool_name=tool_name,
        agent_name=agent_name,
        skill_name=skill_name,
        step_id=step_id,
        result=result,
        artifact=artifact,
    )
