from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal
from uuid import uuid4

from deerflow.event_bus import publish_task_event
from app.core.config import DATABASE_URL, LANGGRAPH_CHECKPOINT_SETUP, SKILL_ONLINE_EVOLUTION_ENABLED
from app.core.database import SessionLocal
from app.memory.service import build_context_for_state, consolidate_task_memory, record_memory_event, summarize_task_session
from app.memory.short_term import append_recent_event
from app.observability import increment
from app.services.learning_service import record_agent_learning_outputs

from .llm_gateway import llm_gateway
from .planner import generate_answer_agent_decision, generate_plan, generate_research_source_decision, generate_supervisor_decision
from .schemas import (
    AgentEvent,
    AgentMessage,
    AgentTaskState,
    Artifact,
    RuntimeStore,
    validate_agent_event,
    validate_agent_state,
    validate_node_update,
    validate_plan_route,
    validate_supervisor_route,
)
from .tool_manager import execute_managed_tool
from .tools import registry as tool_registry
from .tool_permissions import ToolPermissionError, assert_tool_allowed
from .feedback import FailureSignal, get_feedback_summary_for_prompt, record_verification_failure
from .source_policy import resolve_source_policy

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, interrupt
except Exception as exc:  # noqa: BLE001
    END = "__end__"
    START = "__start__"
    StateGraph = None
    Command = None
    interrupt = None
    LANGGRAPH_IMPORT_ERROR = exc
else:
    LANGGRAPH_IMPORT_ERROR = None


MAX_REPAIR_COUNT = 2
RESEARCH_AGENT_TOOLS = {
    "web.search_duckduckgo",
    "github.search_repositories",
    "github.read_readme",
    "knowledge.answer",
    "knowledge.repair_retrieval",
    "architecture.generate_proposal",
    "verification.verify_proposal",
    "planning.repair_research_tasks",
}


def _format_scorecard(scorecard: dict[str, Any] | list[dict[str, Any]] | None) -> list[str]:
    if not scorecard:
        return []
    items = scorecard if isinstance(scorecard, list) else scorecard.get("items") or []
    if not items:
        return []
    lines = ["", "## 鐠囧嫬鍨庨崺铏瑰殠", "| 缂佹潙瀹?| 閸掑棙鏆?| 閸樼喎娲?| 娑撳绔村?|", "| --- | ---: | --- | --- |"]
    for item in items[:6]:
        title = item.get("title") or item.get("key") or "score_item"
        score = item.get("score", "pending")
        reason = item.get("reason") or item.get("criterion") or "needs evidence"
        improvement = item.get("improvement") or "add evidence and retry"
        lines.append(f"| {title} | {score}/5 | {reason} | {improvement} |")
    if isinstance(scorecard, dict) and scorecard.get("total_score") is not None:
        lines.extend(["", f"total_score: {scorecard.get('total_score')}/100"])
    return lines


class AgentTaskCancelled(RuntimeError):
    pass


class AgentBudgetExceeded(RuntimeError):
    pass


def _event(state: dict[str, Any], store: RuntimeStore, event_type: str, message: str, *, agent_name: str, payload: dict[str, Any] | None = None, tool_name: str | None = None, step_id: str | None = None) -> AgentEvent:
    event: AgentEvent = validate_agent_event({
        "session_id": state["session_id"],
        "task_id": state["task_id"],
        "run_id": state.get("run_id"),
        "event_type": event_type,
        "agent_name": agent_name,
        "tool_name": tool_name,
        "step_id": step_id,
        "message": message,
        "payload": payload or {},
        "created_at": datetime.now(timezone.utc),
    })
    event_index = store.append_event(event)
    if event_index is not None:
        event["event_index"] = int(event_index)
        event = validate_agent_event(event)
    append_recent_event(state.get("user_id"), state.get("session_id"), event)
    publish_task_event(event)
    increment("agent_events_total", {"event_type": event_type, "agent": agent_name})
    return event


def _llm_token_callback(state: dict[str, Any], agent_name: str):
    """Publish transient token chunks without persisting every token to SQL."""
    def emit(token: str, metadata: dict[str, str]) -> None:
        if not token:
            return
        publish_task_event({
            "session_id": state["session_id"],
            "task_id": state["task_id"],
            "run_id": state.get("run_id"),
            "event_type": "llm.token",
            "agent_name": agent_name,
            "message": "LLM streaming output.",
            "payload": {"text": token, **metadata},
            "created_at": datetime.now(timezone.utc),
        })
    return emit


def _record_memory_event(state: dict[str, Any], *, event_type: str, category: str | None, content: str, metadata: dict[str, Any] | None = None) -> None:
    try:
        record_memory_event(
            user_id=state.get("user_id"), session_id=state.get("session_id"), task_id=state.get("task_id"),
            event_type=event_type, category=category, content=content, metadata=metadata or {},
        )
    except Exception:  # noqa: BLE001
        return


def _enforce_policy(state: dict[str, Any], store: RuntimeStore) -> None:
    if hasattr(store, "is_cancel_requested") and store.is_cancel_requested(state["task_id"]):
        raise AgentTaskCancelled("Task cancellation was requested.")
    budget = state.get("budget") or {}
    if len(state.get("artifacts") or []) >= int(budget.get("max_tool_calls") or 12):
        raise AgentBudgetExceeded("Task reached max_tool_calls.")
    started_at = budget.get("started_at")
    if started_at:
        try:
            elapsed = (datetime.now(timezone.utc) - datetime.fromisoformat(str(started_at).replace("Z", "+00:00"))).total_seconds()
            if elapsed > float(budget.get("deadline_seconds") or 900):
                raise AgentBudgetExceeded("Task deadline exceeded.")
        except ValueError:
            pass


def _supervisor_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    decision, source, error = generate_supervisor_decision(
        state,
        on_token=_llm_token_callback(state, "supervisor_agent"),
    )
    supervisor_payload = decision.model_dump(mode="json")
    route_payload = {
        "target_node": decision.route,
        "intent": decision.intent,
        "reason": decision.reason,
        "confidence": decision.confidence,
        "needs_rag": decision.needs_rag,
        "needs_tools": decision.needs_tools,
        "needs_verification": decision.needs_verification,
        "stop_after_node": decision.stop_after_children,
        "response_mode": decision.response_mode,
        "query": decision.query,
        "child_agents": supervisor_payload.get("child_agents") or [],
    }
    event = _event(
        state,
        store,
        "supervisor.delegated",
        f"Supervisor delegated to {', '.join(supervisor_payload.get('child_agents') or [])}.",
        agent_name="supervisor_agent",
        payload={"source": source, "decision": supervisor_payload, "route_decision": route_payload, "error": error},
    )
    return validate_node_update({
        "intent": decision.intent,
        "supervisor_decision": supervisor_payload,
        "supervisor_source": source,
        "supervisor_error": error,
        "route_decision": route_payload,
        "route_source": source,
        "route_error": error,
        "status": "running",
        "emitted_events": [event],
    })


def _supervisor_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    decision = state.get("supervisor_decision") or {}
    return validate_supervisor_route(str(decision.get("route") or "research"))


def _tools_for_agent(agent_name: str) -> list[dict[str, Any]]:
    available: list[dict[str, Any]] = []
    for item in tool_registry.describe():
        name = str(item.get("name") or "")
        try:
            assert_tool_allowed(agent_name, name)
        except ToolPermissionError:
            continue
        available.append(item)
    return available


def _answer_from_tool_artifacts(artifacts: list[Artifact]) -> str:
    for artifact in reversed(artifacts):
        data = artifact.get("data") or {}
        if data.get("answer"):
            return str(data["answer"])
        repositories = data.get("repositories")
        if isinstance(repositories, list) and repositories:
            lines = ["Registered GitHub search returned:"]
            for item in repositories[:5]:
                lines.append(f"- {item.get('full_name')}: {item.get('description') or item.get('url')}")
            return "\n".join(lines)
        if data.get("readme"):
            return str(data["readme"])[:4000]
    return "Registered tools ran, but no clear answer was produced. Please add more source material or ask a more specific question."


def _answer_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    available_tools = _tools_for_agent("answer_agent")
    decision, source, error = generate_answer_agent_decision(
        state,
        tool_package="deerflow.tools",
        available_tools=available_tools,
        on_token=_llm_token_callback(state, "answer_agent"),
    )
    decision_payload = decision.model_dump(mode="json")
    source_policy = resolve_source_policy(state)
    calls = list(decision_payload.get("calls") or [])
    if source_policy == "local_only":
        calls = [call for call in calls if str(call.get("tool_name") or "").startswith("knowledge.")]
        if not calls:
            calls = [{
                "call_id": f"tool-{uuid4().hex[:8]}",
                "tool_name": "knowledge.answer",
                "arguments": {"query": state.get("user_input") or "", "top_k": 5, "rewrite_query": True},
                "reason": "The task is restricted to the local knowledge base.",
            }]
    decision_payload["calls"] = calls
    artifacts: list[Artifact] = []
    errors: list[dict[str, Any]] = []
    result_summaries: list[dict[str, Any]] = []
    messages: list[AgentMessage] = []
    emitted_events: list[AgentEvent] = [
        _event(
            state,
            store,
            "answer.agent_decided",
            "Answer Agent selected registered tools.",
            agent_name="answer_agent",
            payload={
                "source": source,
                "decision": decision_payload,
                "tool_package": "deerflow.tools",
                "available_tools": available_tools,
                "error": error,
            },
        )
    ]

    if not calls:
        try:
            response = llm_gateway.invoke(
                role="planner",
                prompt=(
                    "Answer the user clearly and directly. Do not claim retrieval, web access, "
                    "or citations that were not provided.\n\n"
                    f"User question: {state.get('user_input', '')}"
                ),
                on_token=_llm_token_callback(state, "answer_agent"),
            )
            answer_draft = response.content.strip()
            model_source = f"{response.provider}:{response.model}"
        except Exception:  # noqa: BLE001
            answer_draft = f"I can help with this question: {state.get('user_input', '')}"
            model_source = "fallback"
        feedback = {"success_count": 0, "failure_count": 0, "next_action": "complete", "results": [], "source": source, "error": error}
        emitted_events.append(_event(
            state,
            store,
            "agent.completed",
            "Answer Agent prepared a direct response draft.",
            agent_name="answer_agent",
            payload={"model_source": model_source},
        ))
        return validate_node_update({
            "status": "running",
            "answer_draft": answer_draft,
            "tool_feedback": feedback,
            "grounding": {"mode": "answer_agent", "rag_used": False},
            "emitted_events": emitted_events,
        })

    for index, call in enumerate(decision_payload.get("calls") or []):
        tool_name = str(call.get("tool_name") or "")
        arguments = dict(call.get("arguments") or {})
        call_id = str(call.get("call_id") or f"tool-{index + 1}")
        messages.append({
            "message_id": str(uuid4()),
            "from_agent": "answer_agent",
            "to_agent": tool_name,
            "kind": "tool_request",
            "correlation_id": call_id,
            "payload": {"tool_name": tool_name, "arguments": arguments, "reason": call.get("reason")},
        })
        emitted_events.append(_event(
            state,
            store,
            "tool.started",
            f"Answer Agent invoked {tool_name}.",
            agent_name="answer_agent",
            tool_name=tool_name,
            step_id=call_id,
            payload={"reason": call.get("reason")},
        ))
        execution = execute_managed_tool(
            state,
            store,
            tool_name=tool_name,
            arguments=arguments,
            agent_name="answer_agent",
            skill_name="registered_tool_use",
            step_id=call_id,
            call_id=call_id,
        )
        artifacts.append(execution.artifact)
        result = execution.result
        error_info = result.get("error") or {}
        feedback_item = {
            "requester": "answer_agent",
            "executor": "answer_agent",
            "call_id": call_id,
            "tool_name": tool_name,
            "ok": bool(result.get("ok")),
            "artifact_id": execution.artifact["artifact_id"],
            "confidence": float(result.get("confidence") or 0),
            "grounding": result.get("grounding") or {},
            "error": error_info or None,
        }
        result_summaries.append(feedback_item)
        messages.append({
            "message_id": str(uuid4()),
            "from_agent": tool_name,
            "to_agent": "answer_agent",
            "kind": "tool_result",
            "correlation_id": call_id,
            "payload": feedback_item,
        })
        emitted_events.append(_event(
            state,
            store,
            "tool.completed" if result.get("ok") else "tool.failed",
            f"Registered tool {tool_name} completed.",
            agent_name="answer_agent",
            tool_name=tool_name,
            step_id=call_id,
            payload=feedback_item,
        ))
        if error_info:
            errors.append({
                "source": tool_name,
                "error_type": error_info.get("type", "tool_failed"),
                "message": error_info.get("message", "Registered tool failed."),
                "retryable": bool(error_info.get("retryable", True)),
                "correlation_id": call_id,
            })

    success_count = len([item for item in result_summaries if item.get("ok")])
    failure_count = max(0, len(result_summaries) - success_count)
    next_action = str(decision_payload.get("next_action") or "complete")
    if success_count == 0 and result_summaries:
        next_action = "fallback"
    feedback = {
        "success_count": success_count,
        "failure_count": failure_count,
        "next_action": next_action,
        "results": result_summaries,
        "reason": decision_payload.get("reason"),
        "source": source,
        "error": error,
    }
    citations = [citation for artifact in artifacts for citation in artifact.get("citations", [])]
    grounding = next((artifact.get("grounding") for artifact in reversed(artifacts) if artifact.get("grounding")), None)
    emitted_events.append(_event(state, store, "answer.feedback_ready", "Answer Agent produced structured feedback.", agent_name="answer_agent", payload=feedback))
    update: dict[str, Any] = {
        "tool_feedback": feedback,
        "artifacts": artifacts,
        "messages": messages,
        "errors": errors,
        "citations": citations,
        "grounding": grounding or {},
        "emitted_events": emitted_events,
    }
    if next_action == "complete":
        update.update({
            "status": "running",
            "answer_draft": _answer_from_tool_artifacts(artifacts),
            "grounding": grounding or {"mode": "answer_agent", "rag_used": bool(citations)},
        })
        emitted_events.append(_event(state, store, "agent.completed", "Answer Agent prepared a response draft.", agent_name="answer_agent", payload={"tool_feedback": feedback}))
    else:
        update["status"] = "running"
    return validate_node_update(update)


def _answer_agent_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    feedback = state.get("tool_feedback") or {}
    next_action = str(feedback.get("next_action") or "complete")
    if next_action == "fallback":
        return "fallback_response"
    return "final_response"


def _planner_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    try:
        memory_context = build_context_for_state({key: state.get(key) for key in ("user_id", "session_id", "task_id", "run_id")})
        memory_event = _event(state, store, "memory.loaded", "Memory Agent loaded scoped profile and session context.", agent_name="memory_agent")
        memory_artifact: Artifact = {
            "artifact_id": f"memory-{uuid4().hex[:12]}", "kind": "memory", "producer": "memory_agent",
            "correlation_id": "memory", "data": memory_context, "citations": [], "confidence": 1.0, "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        memory_context = {"available": False, "error": str(exc)}
        memory_event = _event(state, store, "memory.degraded", "Memory context unavailable; continuing without it.", agent_name="memory_agent")
        memory_artifact = {"artifact_id": f"memory-{uuid4().hex[:12]}", "kind": "memory", "producer": "memory_agent", "correlation_id": "memory", "data": memory_context, "citations": [], "confidence": 0.0, "error": {"type": "memory_unavailable"}}

    # Inject recent verification feedback so the Planner can adapt parameters
    # (e.g. increase top_k when retrieval_empty failures are trending).
    feedback_summary = get_feedback_summary_for_prompt(
        kb_id=state.get("kb_id"), window_days=14,
    )
    plan, source, error = generate_plan(
        {**state, "memory_context": memory_context, "feedback_summary": feedback_summary},
        on_token=_llm_token_callback(state, "planner_agent"),
    )
    if len(plan.research_tasks) > int((state.get("budget") or {}).get("max_steps") or 8):
        raise AgentBudgetExceeded("Supervisor plan exceeds max_steps.")
    status: Literal["waiting_user", "running"] = "waiting_user" if plan.approval_required else "running"
    _record_memory_event(state, event_type="user_goal_set", category="learning_goal", content=str(state.get("user_input") or ""), metadata={"confidence": 0.7, "source": "planner_agent"})
    plan_payload = plan.model_dump(mode="json")
    store.save_plan({"task_id": state["task_id"], "run_id": state["run_id"], "plan_version": int(state.get("repair_count") or 0) + 1, "source": source, "status": "awaiting_approval" if plan.approval_required else "active", "goal": plan.goal, "intent": plan.intent, "steps": plan_payload.get("research_tasks", []), "error_message": error})
    plan_event = _event(state, store, "plan.created", "Planner Agent created an executable research plan.", agent_name="planner_agent", payload={"source": source, "plan": plan_payload, "error": error})
    return validate_node_update({"memory_context": memory_context, "plan": plan_payload, "goal": plan.goal, "intent": plan.intent, "planning_source": source, "planner_error": error, "status": status, "artifacts": [memory_artifact], "emitted_events": [memory_event, plan_event]})


def _plan_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    route = "approval_gate" if (state.get("plan") or {}).get("approval_required") else "research_agent"
    return validate_plan_route(route)


def _approval_gate(state: AgentTaskState, store: RuntimeStore) -> Any:
    state = validate_agent_state(state)
    if interrupt is None or Command is None:
        raise RuntimeError("LangGraph interrupt support is required for approval gates.")
    request = {"task_id": state["task_id"], "reason": (state.get("plan") or {}).get("approval_reason") or "Review the Supervisor plan before execution.", "plan": state.get("plan"), "allowed_actions": ["approve", "edit", "reject"]}
    decision = interrupt(request)
    action = str((decision or {}).get("action") or "reject") if isinstance(decision, dict) else "reject"
    event = _event(state, store, "approval.resolved", f"User approval action: {action}.", agent_name="human_gate", payload={"decision": decision})
    if action == "edit":
        edited_input = str((decision or {}).get("user_input") or state["user_input"])
        return Command(update=validate_node_update({"user_input": edited_input, "approval": {"decision": decision}, "status": "running", "emitted_events": [event]}), goto="planner_agent")
    if action == "approve":
        return Command(update=validate_node_update({"approval": {"decision": decision}, "status": "running", "emitted_events": [event]}), goto="research_agent")
    return Command(update=validate_node_update({"approval": {"decision": decision}, "status": "running", "emitted_events": [event]}), goto="fallback_response")


def _research_tasks_from_state(state: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = list((state.get("plan") or {}).get("research_tasks") or [])
    if tasks:
        return tasks
    query = str((state.get("route_decision") or {}).get("query") or state.get("user_input") or "").strip()
    return [{
        "task_id": "evidence",
        "query": query or str(state.get("user_input") or ""),
        "objective": "Retrieve evidence for the user request.",
        "top_k": 5,
    }]


def _citations_from_artifacts(artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [citation for artifact in artifacts for citation in artifact.get("citations", [])]


def _record_research_verification_failures(state: dict[str, Any], verification: dict[str, Any], artifacts: list[dict[str, Any]]) -> None:
    if verification.get("status") == "passed":
        return
    all_citations = _citations_from_artifacts(artifacts)
    for issue in verification.get("issues") or []:
        try:
            record_verification_failure(
                FailureSignal(
                    task_id=state["task_id"],
                    run_id=state.get("run_id"),
                    failure_type=str(issue.get("type", "other")),
                    message=str(issue.get("message", "")),
                    repair_strategy_used="rewrite_query" if verification.get("status") == "repair" else None,
                    kb_id=state.get("kb_id"),
                    query_snippet=str(state.get("user_input", ""))[:200],
                    top_k_used=(state.get("budget") or {}).get("top_k", 5),
                    source_count=len({citation.get("document_id") or citation.get("source") for citation in all_citations}),
                    citation_count=len(all_citations),
                    confidence=float(verification.get("score") or 0),
                )
            )
        except Exception:  # noqa: BLE001
            continue


def _research_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    try:
        _enforce_policy(state, store)
    except AgentBudgetExceeded as exc:
        event = _event(
            state,
            store,
            "agent.failed",
            str(exc),
            agent_name="research_agent",
            payload={"error_type": "budget_exceeded"},
        )
        return validate_node_update({
            "status": "failed",
            "next_action": "fallback",
            "errors": [{
                "source": "research_agent",
                "error_type": "budget_exceeded",
                "message": str(exc),
                "retryable": False,
                "correlation_id": "research_agent",
            }],
            "emitted_events": [event],
        })

    registered_tools = {
        item["name"]: item
        for item in tool_registry.describe()
        if item["name"] in RESEARCH_AGENT_TOOLS
    }
    start = _event(
        state,
        store,
        "agent.started",
        "Research Agent started autonomous registered-tool loop.",
        agent_name="research_agent",
        payload={"available_tools": list(registered_tools.values())},
    )

    emitted_events: list[AgentEvent] = [start]
    messages: list[AgentMessage] = []
    artifacts: list[Artifact] = []
    errors: list[dict[str, Any]] = []
    feedback: dict[str, Any] = {}
    proposal: dict[str, Any] = dict(state.get("proposal") or {})
    verification: dict[str, Any] = dict(state.get("verification") or {})
    plan = dict(state.get("plan") or {})
    repair_count = int(state.get("repair_count") or 0)
    next_action: Literal["ask_user", "fallback", "complete"] = "fallback"
    tasks = _research_tasks_from_state(state)
    source_policy = resolve_source_policy(state)

    def working_state() -> dict[str, Any]:
        merged_artifacts = [*list(state.get("artifacts") or []), *artifacts]
        return {
            **state,
            "artifacts": merged_artifacts,
            "proposal": proposal,
            "verification": verification,
            "plan": plan,
            "repair_count": repair_count,
            "citations": _citations_from_artifacts(merged_artifacts),
            "grounding": next((item.get("grounding") for item in reversed(merged_artifacts) if item.get("grounding")), state.get("grounding") or {}),
        }

    def call_research_tool(
        tool_name: str,
        arguments: dict[str, Any],
        *,
        reason: str,
        step_id: str,
        precomputed_execution: Any | None = None,
    ) -> dict[str, Any]:
        try:
            _enforce_policy(working_state(), store)
        except AgentBudgetExceeded as exc:
            if not (tool_name == "verification.verify_proposal" and "max_tool_calls" in str(exc)):
                raise
        tool_request: AgentMessage = {
            "message_id": str(uuid4()),
            "from_agent": "research_agent",
            "to_agent": tool_name,
            "kind": "tool_request",
            "correlation_id": step_id,
            "payload": {"tool_name": tool_name, "arguments": arguments, "reason": reason},
        }
        messages.append(tool_request)
        emitted_events.append(_event(
            working_state(),
            store,
            "tool.requested",
            f"Research Agent requested {tool_name}.",
            agent_name="research_agent",
            tool_name=tool_name,
            step_id=step_id,
            payload={"request": tool_request},
        ))
        if precomputed_execution is not None:
            result = precomputed_execution.result
            artifact = precomputed_execution.artifact
            call_id = precomputed_execution.call_id
        elif tool_name.startswith(("architecture.", "verification.", "planning.")):
            trusted_arguments = {
                **arguments,
                "user_input": state.get("user_input", ""),
                "task_id": state.get("task_id", ""),
                "run_id": state.get("run_id", ""),
                "username": state.get("username", ""),
                "user_id": state.get("user_id"),
                "kb_id": state.get("kb_id"),
                "memory_context": state.get("memory_context") or {},
                "scenario": state.get("scenario") or {},
                "budget": state.get("budget") or {},
            }
            assert_tool_allowed("research_agent", tool_name)
            result = tool_registry._handlers[tool_name](trusted_arguments)
            artifact_kind = "proposal" if tool_name.startswith("architecture.") else "verification" if tool_name.startswith("verification.") else "plan"
            artifact = {
                "artifact_id": f"tool-{uuid4().hex[:12]}",
                "kind": artifact_kind,
                "producer": "research_agent",
                "correlation_id": step_id,
                "data": result.get("data") or {},
                "citations": result.get("citations") or [],
                "confidence": float(result.get("confidence") or 0),
                "error": result.get("error"),
                "grounding": result.get("grounding") or {},
            }
            store.save_tool_call({
                "task_id": state["task_id"],
                "run_id": state["run_id"],
                "step_id": step_id,
                "agent_name": "research_agent",
                "skill_name": "autonomous_research_tools",
                "tool_name": tool_name,
                "input": trusted_arguments,
                "output": result,
                "result": result,
                "ok": bool(result.get("ok")),
                "error_type": (result.get("error") or {}).get("type"),
                "error_message": (result.get("error") or {}).get("message"),
                "retry_count": 0,
                "latency_ms": result.get("latency_ms"),
            })
            call_id = step_id
        else:
            execution = execute_managed_tool(
                working_state(),
                store,
                tool_name=tool_name,
                arguments=arguments,
                agent_name="research_agent",
                skill_name="autonomous_research_tools",
                step_id=step_id,
                call_id=step_id,
            )
            result = execution.result
            artifact = execution.artifact
            call_id = execution.call_id
        artifacts.append(artifact)
        feedback_item = {
            "requester": "research_agent",
            "executor": "research_agent",
            "call_id": call_id,
            "tool_name": tool_name,
            "ok": bool(result.get("ok")),
            "artifact_id": artifact["artifact_id"],
            "confidence": float(result.get("confidence") or 0),
            "grounding": result.get("grounding") or {},
            "error": result.get("error"),
        }
        feedback[step_id] = feedback_item
        messages.append({
            "message_id": str(uuid4()),
            "from_agent": tool_name,
            "to_agent": "research_agent",
            "kind": "tool_result",
            "correlation_id": step_id,
            "payload": feedback_item,
        })
        emitted_events.append(_event(
            working_state(),
            store,
            "tool.completed" if result.get("ok") else "tool.failed",
            f"Registered tool {tool_name} completed.",
            agent_name="research_agent",
            tool_name=tool_name,
            step_id=step_id,
            payload=feedback_item,
        ))
        if result.get("error"):
            error = result.get("error") or {}
            errors.append({
                "source": tool_name,
                "error_type": error.get("type", "tool_failed"),
                "message": error.get("message", "Registered tool failed."),
                "retryable": bool(error.get("retryable", True)),
                "correlation_id": step_id,
            })
        return {"result": result, "artifact": artifact, "feedback": feedback_item}

    while True:
        evidence_requests: list[tuple[str, str, dict[str, Any], str]] = []
        selected_sources: dict[str, set[str]] = {}
        source_tools = [
            item for name, item in registered_tools.items()
            if name in {"knowledge.answer", "web.search_duckduckgo", "github.search_repositories"}
        ]
        for item in tasks:
            task_id = str(item.get("task_id") or f"research-{uuid4().hex[:8]}")
            query = str(item.get("query") or state.get("user_input") or "")
            objective = str(item.get("objective") or "Retrieve supporting evidence.")
            top_k = int(item.get("top_k") or 5)
            source_decision, source_decision_source, source_decision_error = generate_research_source_decision(
                state,
                query=query,
                objective=objective,
                available_tools=source_tools,
                on_token=_llm_token_callback(state, "research_agent"),
            )
            chosen = set(source_decision.tool_names)
            selected_sources[task_id] = chosen
            emitted_events.append(_event(
                working_state(),
                store,
                "research.sources_selected",
                "Research Agent selected evidence sources.",
                agent_name="research_agent",
                step_id=task_id,
                payload={
                    "tool_names": sorted(chosen),
                    "reason": source_decision.reason,
                    "source": source_decision_source,
                    "error": source_decision_error,
                },
            ))
            if "knowledge.answer" in chosen:
                evidence_requests.append((
                    f"{task_id}-rag",
                    "knowledge.answer",
                    {
                        "query": query,
                        "top_k": top_k,
                        "rewrite_query": True,
                        "username": state.get("username"),
                        "user_id": state.get("user_id"),
                        "kb_id": state.get("kb_id"),
                        "document_id": state.get("document_id"),
                        "conversation_id": state.get("conversation_id"),
                    },
                    objective,
                ))
            if "web.search_duckduckgo" in chosen:
                evidence_requests.append((
                    f"{task_id}-web",
                    "web.search_duckduckgo",
                    {"query": query, "limit": min(10, top_k)},
                    f"Retrieve public web evidence: {objective}",
                ))
            if "github.search_repositories" in chosen:
                evidence_requests.append((
                    f"{task_id}-github",
                    "github.search_repositories",
                    {"query": query, "limit": min(10, top_k), "sort": "stars"},
                    f"Find relevant open-source project entry points: {objective}",
                ))

        result_by_step: dict[str, dict[str, Any]] = {}
        web_requests = [item for item in evidence_requests if item[1] == "web.search_duckduckgo"]
        # Start network-only web searches first. RAG continues on the runtime
        # thread because its database session is intentionally thread-bound.
        with ThreadPoolExecutor(max_workers=min(4, max(1, len(web_requests)))) as executor:
            web_futures = {
                request_id: executor.submit(
                    execute_managed_tool,
                    working_state(),
                    store,
                    tool_name=tool_name,
                    arguments=arguments,
                    agent_name="research_agent",
                    skill_name="parallel_evidence_retrieval",
                    step_id=request_id,
                    call_id=request_id,
                )
                for request_id, tool_name, arguments, _reason in web_requests
            }
            for request_id, tool_name, arguments, reason in evidence_requests:
                if tool_name == "web.search_duckduckgo":
                    continue
                result_by_step[request_id] = call_research_tool(
                    tool_name,
                    arguments,
                    reason=reason,
                    step_id=request_id,
                )["result"]
            for request_id, tool_name, arguments, reason in web_requests:
                result_by_step[request_id] = call_research_tool(
                    tool_name,
                    arguments,
                    reason=reason,
                    step_id=request_id,
                    precomputed_execution=web_futures[request_id].result(),
                )["result"]

        for item in tasks:
            task_id = str(item.get("task_id") or "evidence")
            query = str(item.get("query") or state.get("user_input") or "")
            top_k = int(item.get("top_k") or 5)
            local_result = result_by_step.get(f"{task_id}-rag") or {}
            local_has_evidence = bool(local_result.get("citations"))
            any_selected_evidence = any(
                bool(result_by_step.get(f"{task_id}-{suffix}", {}).get("citations"))
                for suffix in ("rag", "web", "github")
            )
            local_error = local_result.get("error") or {}

            # A local retry is an Agent-selected repair, not an automatic
            # consequence of an intentionally omitted local source.
            if (
                "knowledge.answer" in selected_sources.get(task_id, set())
                and
                not local_has_evidence
                and (source_policy == "local_only" or not any_selected_evidence)
                and bool(local_error.get("retryable", True))
            ):
                call_research_tool(
                    "knowledge.repair_retrieval",
                    {"query": query, "top_k": top_k, "repair_reason": local_error.get("type") or "retrieval_insufficient"},
                    reason="Repair local retrieval only when local evidence is required or no public evidence was found.",
                    step_id=f"{task_id}-rag-repair",
                )

        proposal_result = call_research_tool(
            "architecture.generate_proposal",
            {
                "user_input": state.get("user_input", ""),
                "artifacts": [*list(state.get("artifacts") or []), *artifacts],
                "memory_context": state.get("memory_context") or {},
                "scenario": state.get("scenario") or {},
            },
            reason="Generate proposal from accumulated research artifacts.",
            step_id=f"proposal-{uuid4().hex[:8]}",
        )["result"]
        proposal = dict(proposal_result.get("data") or {})

        merged_artifacts = [*list(state.get("artifacts") or []), *artifacts]
        verification_result = call_research_tool(
            "verification.verify_proposal",
            {
                "proposal": proposal,
                "artifacts": merged_artifacts,
                "citations": _citations_from_artifacts(merged_artifacts),
                "grounding": next((item.get("grounding") for item in reversed(merged_artifacts) if item.get("grounding")), {}),
            },
            reason="Verify proposal evidence coverage.",
            step_id="proposal-verification",
        )["result"]
        verification = dict(verification_result.get("data") or {})
        store.save_verification({
            "task_id": state["task_id"],
            "run_id": state["run_id"],
            "step_id": "proposal",
            "status": "passed" if verification.get("status") == "passed" else "failed",
            "score": float(verification.get("score") or 0),
            "issues": verification.get("issues") or [],
            "evidence": {"artifact_count": len(merged_artifacts)},
        })
        _record_research_verification_failures(state, verification, merged_artifacts)

        status = str(verification.get("status") or "")
        if status == "passed":
            next_action = "complete"
            break
        if status == "needs_approval":
            next_action = "ask_user"
            break
        if status == "repair" and repair_count < MAX_REPAIR_COUNT:
            try:
                repair_result = call_research_tool(
                    "planning.repair_research_tasks",
                    {"user_input": state.get("user_input", ""), "verification": verification, "repair_count": repair_count},
                    reason="Create focused repair research tasks from verification feedback.",
                    step_id=f"repair-plan-{repair_count + 1}",
                )["result"]
            except AgentBudgetExceeded:
                next_action = "fallback"
                break
            repair_payload = repair_result.get("data") or {}
            repair_tasks = list(repair_payload.get("research_tasks") or [])
            if not repair_tasks:
                next_action = "fallback"
                break
            repair_count = int(repair_payload.get("repair_count") or repair_count + 1)
            tasks = repair_tasks
            plan = {**plan, "research_tasks": repair_tasks}
            emitted_events.append(_event(
                working_state(),
                store,
                "repair.started",
                "Research Agent scheduled targeted registered-tool repair.",
                agent_name="research_agent",
                payload={"repair_count": repair_count, "research_tasks": repair_tasks},
            ))
            continue
        next_action = "fallback"
        break

    done = _event(
        working_state(),
        store,
        "agent.completed" if next_action == "complete" else "agent.delegated",
        "Research Agent completed autonomous registered-tool loop.",
        agent_name="research_agent",
        payload={"next_action": next_action, "verification": verification},
    )
    emitted_events.append(done)
    return validate_node_update({
        "status": "running",
        "next_action": next_action,
        "plan": plan,
        "proposal": proposal,
        "verification": verification,
        "repair_count": repair_count,
        "artifacts": artifacts,
        "messages": messages,
        "tool_feedback": feedback,
        "errors": errors,
        "emitted_events": emitted_events,
    })


def _research_agent_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    action = state.get("next_action")
    if action == "complete":
        return "review_agent"
    if action == "ask_user":
        return "approval_gate"
    return "fallback_response"


def _review_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    """Apply the final publication gate without performing new research or tool calls."""
    state = validate_agent_state(state)
    verification = dict(state.get("verification") or {})
    proposal = dict(state.get("proposal") or {})
    verification_status = str(verification.get("status") or "fallback")
    has_deliverable = bool(proposal.get("summary") or proposal.get("sections"))

    if verification_status == "needs_approval":
        next_action = "ask_user"
        outcome = "needs_confirmation"
        reason = "The research result requires user confirmation before publication."
    elif verification_status == "passed" and has_deliverable:
        next_action = "complete"
        outcome = "approved"
        reason = "Research evidence and the deliverable passed the publication review."
    else:
        next_action = "fallback"
        outcome = "rejected"
        reason = "The research result is missing a verified deliverable or sufficient evidence."

    review = {
        "outcome": outcome,
        "reason": reason,
        "verification_status": verification_status,
        "proposal_present": has_deliverable,
        "citation_count": len([citation for item in state.get("artifacts") or [] for citation in item.get("citations") or []]),
    }
    event = _event(
        state,
        store,
        "review.completed",
        "Review Agent completed the final publication check.",
        agent_name="review_agent",
        payload=review,
    )
    return validate_node_update({
        "status": "running",
        "review": review,
        "next_action": next_action,
        "emitted_events": [event],
    })


def _review_agent_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    action = state.get("next_action")
    if action == "complete":
        return "final_response"
    if action == "ask_user":
        return "approval_gate"
    return "fallback_response"


def _final_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    answer_draft = str(state.get("answer_draft") or "").strip()
    if answer_draft:
        citations = [citation for item in state.get("artifacts", []) for citation in item.get("citations", [])]
        grounding = state.get("grounding") or {"mode": "answer_agent", "rag_used": bool(citations)}
        event = _event(
            state,
            store,
            "task.completed",
            "Answer task completed.",
            agent_name="runtime",
            payload={"citation_count": len(citations)},
        )
        return validate_node_update({
            "status": "completed",
            "final_answer": answer_draft,
            "citations": citations,
            "grounding": grounding,
            "emitted_events": [event],
        })
    proposal = state.get("proposal") or {}
    sections = proposal.get("sections") or []
    lines = [f"# {proposal.get('title', 'Project Improvement Proposal')}", "", proposal.get("summary", "")]
    lines.extend(_format_scorecard(proposal.get("scorecard") or (state.get("verification") or {}).get("rubric_scores")))
    for section in sections:
        lines.extend(["", f"## {section.get('title', 'Recommendation')}", *[f"- {item}" for item in section.get("items", [])]])
    citations = [citation for item in state.get("artifacts", []) for citation in item.get("citations", [])]
    groundings = [item.get("grounding") or {} for item in state.get("artifacts", [])]
    rag_used = any(bool(item.get("rag_used")) for item in groundings)
    web_used = any(item.get("external_source") == "duckduckgo" for item in groundings)
    grounding_mode = "multi_source" if rag_used and web_used else "web_grounded" if web_used else "rag_grounded" if rag_used else "general"
    final_answer = "\n".join(line for line in lines if line is not None).strip()
    _record_memory_event(state, event_type="task_completed", category="project_context", content=final_answer[:1200], metadata={"confidence": (state.get("verification") or {}).get("score", 0.7), "citation_count": len(citations)})
    try:
        memory_updates = consolidate_task_memory({**state, "final_answer": final_answer})
        session_summary = summarize_task_session({**state, "final_answer": final_answer})
    except Exception:  # noqa: BLE001
        memory_updates, session_summary = [], None
    try:
        with SessionLocal() as db:
            record_agent_learning_outputs(db, {**state, "citations": citations}, final_answer)
    except Exception:  # noqa: BLE001
        pass
    event = _event(state, store, "task.completed", "Multi-agent task completed.", agent_name="runtime", payload={"citation_count": len(citations)})
    return validate_node_update({"status": "completed", "final_answer": final_answer, "citations": citations, "grounding": {"mode": grounding_mode, "rag_used": rag_used, "web_used": web_used, "source_policy": resolve_source_policy(state)}, "memory_updates": memory_updates, "session_summary": session_summary, "emitted_events": [event]})


def _fallback_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    final_answer = "当前无法取得足够的可验证证据来完成该方案。请补充或索引相关项目材料后重试。"
    event = _event(state, store, "fallback.used", "Evidence is insufficient; returned safe fallback.", agent_name="runtime", payload={"verification": state.get("verification")})
    return validate_node_update({"status": "completed", "final_answer": final_answer, "grounding": {"mode": "insufficient_evidence", "rag_used": False}, "emitted_events": [event]})


def _maybe_run_skill_evolution(final_state: AgentTaskState) -> None:
    if not SKILL_ONLINE_EVOLUTION_ENABLED:
        return
    user_input = str(final_state.get("user_input") or "").strip()
    assistant_text = str(final_state.get("final_answer") or "").strip()
    if not user_input or not assistant_text:
        return
    try:
        from app.skills.online_evolution import online_ingest

        def side_query(system: str, user_message: str) -> str:
            return llm_gateway.invoke(role="judge", prompt=f"{system}\n\n{user_message}").content

        online_ingest(
            messages=[
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": assistant_text},
            ],
            side_query=side_query,
            retrieved_reference=final_state.get("skill_context") if isinstance(final_state.get("skill_context"), dict) else None,
            hint=str(final_state.get("task_type") or ""),
            target="generated",
        )
    except Exception:  # noqa: BLE001
        return


def build_agent_graph(store: RuntimeStore, *, checkpointer: Any) -> Any:
    if StateGraph is None:
        raise RuntimeError(f"langgraph is required for Agent Runtime: {LANGGRAPH_IMPORT_ERROR}")
    builder = StateGraph(AgentTaskState)
    builder.add_node("supervisor_agent", lambda state: _supervisor_agent(state, store))
    builder.add_node("answer_agent", lambda state: _answer_agent(state, store))
    builder.add_node("planner_agent", lambda state: _planner_agent(state, store))
    builder.add_node("approval_gate", lambda state: _approval_gate(state, store))
    builder.add_node("research_agent", lambda state: _research_agent(state, store))
    builder.add_node("review_agent", lambda state: _review_agent(state, store))
    builder.add_node("final_response", lambda state: _final_response(state, store))
    builder.add_node("fallback_response", lambda state: _fallback_response(state, store))
    builder.add_edge(START, "supervisor_agent")
    builder.add_conditional_edges("supervisor_agent", _supervisor_route, {
        "answer": "answer_agent",
        "research": "planner_agent",
    })
    builder.add_conditional_edges("answer_agent", _answer_agent_route, {"final_response": "final_response", "fallback_response": "fallback_response"})
    builder.add_conditional_edges("planner_agent", _plan_route, {"approval_gate": "approval_gate", "research_agent": "research_agent"})
    builder.add_conditional_edges("research_agent", _research_agent_route, {"review_agent": "review_agent", "approval_gate": "approval_gate", "fallback_response": "fallback_response"})
    builder.add_conditional_edges("review_agent", _review_agent_route, {"final_response": "final_response", "approval_gate": "approval_gate", "fallback_response": "fallback_response"})
    builder.add_edge("final_response", END)
    builder.add_edge("fallback_response", END)
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def postgres_checkpointer() -> Iterator[Any]:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("langgraph-checkpoint-postgres must be installed for durable Agent Runtime execution.") from exc
    with PostgresSaver.from_conn_string(DATABASE_URL) as saver:
        if LANGGRAPH_CHECKPOINT_SETUP:
            saver.setup()
        yield saver


def _invoke_graph(initial_state: AgentTaskState, store: RuntimeStore, checkpointer: Any, resume_payload: dict[str, Any] | None) -> AgentTaskState:
    initial_state = validate_agent_state(initial_state)
    graph = build_agent_graph(store, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": initial_state["task_id"]}}
    if resume_payload is not None:
        input_value: Any = Command(resume=resume_payload)
    else:
        snapshot = graph.get_state(config)
        input_value = None if snapshot.values and snapshot.next else initial_state
    # Consume LangGraph updates as they are produced.  Durable lifecycle events
    # and transient LLM token chunks are published inside nodes immediately.
    for _ in graph.stream(input_value, config=config, stream_mode="updates"):
        pass
    final_state = validate_agent_state(dict(graph.get_state(config).values))
    store.save_task_state(initial_state["task_id"], final_state)
    _maybe_run_skill_evolution(final_state)
    return final_state


def run_agent_task(initial_state: AgentTaskState, store: RuntimeStore, *, resume_payload: dict[str, Any] | None = None, checkpointer: Any | None = None) -> AgentTaskState:
    if StateGraph is None:
        raise RuntimeError(f"langgraph is required for Agent Runtime: {LANGGRAPH_IMPORT_ERROR}")
    initial_state = validate_agent_state(initial_state)
    event = _event(initial_state, store, "task.resumed" if resume_payload else "task.started", "Agent task resumed." if resume_payload else "Agent task started.", agent_name="runtime")
    initial_state = validate_agent_state({**initial_state, "emitted_events": [event]})
    if checkpointer is not None:
        return _invoke_graph(initial_state, store, checkpointer, resume_payload)
    with postgres_checkpointer() as durable_checkpointer:
        return _invoke_graph(initial_state, store, durable_checkpointer, resume_payload)
