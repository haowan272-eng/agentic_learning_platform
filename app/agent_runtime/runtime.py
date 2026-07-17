from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal
from uuid import uuid4

from app.agent_runtime.event_bus import publish_task_event
from app.core.config import DATABASE_URL, LANGGRAPH_CHECKPOINT_SETUP
from app.memory.service import build_context_for_state, consolidate_task_memory, record_memory_event, summarize_task_session
from app.memory.short_term import append_recent_event
from app.observability import increment, observe_ms

from .planner import AgentPlan, ResearchTask, generate_plan, generate_proposal, verify_proposal
from .schemas import AgentError, AgentEvent, AgentMessage, AgentTaskState, Artifact, ResearchWorkItem, RuntimeStore
from .tools import call_tool
from .feedback import FailureSignal, get_feedback_summary_for_prompt, record_verification_failure

try:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, Send, interrupt
except Exception as exc:  # noqa: BLE001
    END = "__end__"
    START = "__start__"
    StateGraph = None
    Command = None
    Send = None
    interrupt = None
    LANGGRAPH_IMPORT_ERROR = exc
else:
    LANGGRAPH_IMPORT_ERROR = None


MAX_REPAIR_COUNT = 2


class AgentTaskCancelled(RuntimeError):
    pass


class AgentBudgetExceeded(RuntimeError):
    pass


def _event(state: dict[str, Any], store: RuntimeStore, event_type: str, message: str, *, agent_name: str, payload: dict[str, Any] | None = None, tool_name: str | None = None, step_id: str | None = None) -> AgentEvent:
    event: AgentEvent = {
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
    }
    event_index = store.append_event(event)
    if event_index is not None:
        event["event_index"] = int(event_index)
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


def _supervisor_plan(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
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

    # Inject recent Verifier feedback so the Planner can adapt parameters
    # (e.g. increase top_k when retrieval_empty failures are trending).
    feedback_summary = get_feedback_summary_for_prompt(
        kb_id=state.get("kb_id"), window_days=14,
    )
    plan, source, error = generate_plan(
        {**state, "memory_context": memory_context, "feedback_summary": feedback_summary},
        on_token=_llm_token_callback(state, "supervisor"),
    )
    if len(plan.research_tasks) > int((state.get("budget") or {}).get("max_steps") or 8):
        raise AgentBudgetExceeded("Supervisor plan exceeds max_steps.")
    status: Literal["waiting_user", "running"] = "waiting_user" if plan.approval_required else "running"
    _record_memory_event(state, event_type="user_goal_set", category="learning_goal", content=str(state.get("user_input") or ""), metadata={"confidence": 0.7, "source": "supervisor"})
    plan_payload = plan.model_dump(mode="json")
    store.save_plan({"task_id": state["task_id"], "run_id": state["run_id"], "plan_version": int(state.get("repair_count") or 0) + 1, "source": source, "status": "awaiting_approval" if plan.approval_required else "active", "goal": plan.goal, "intent": plan.intent, "steps": plan_payload.get("research_tasks", []), "error_message": error})
    plan_event = _event(state, store, "plan.created", "Supervisor created a parallel multi-agent plan.", agent_name="supervisor", payload={"source": source, "plan": plan_payload, "error": error})
    return {"memory_context": memory_context, "plan": plan_payload, "goal": plan.goal, "intent": plan.intent, "planning_source": source, "planner_error": error, "status": status, "artifacts": [memory_artifact], "emitted_events": [memory_event, plan_event]}


def _plan_route(state: AgentTaskState) -> str:
    return "approval_gate" if (state.get("plan") or {}).get("approval_required") else "dispatch_research"


def _approval_gate(state: AgentTaskState, store: RuntimeStore) -> Any:
    if interrupt is None or Command is None:
        raise RuntimeError("LangGraph interrupt support is required for approval gates.")
    request = {"task_id": state["task_id"], "reason": (state.get("plan") or {}).get("approval_reason") or "Review the Supervisor plan before execution.", "plan": state.get("plan"), "allowed_actions": ["approve", "edit", "reject"]}
    decision = interrupt(request)
    action = str((decision or {}).get("action") or "reject") if isinstance(decision, dict) else "reject"
    event = _event(state, store, "approval.resolved", f"User approval action: {action}.", agent_name="human_gate", payload={"decision": decision})
    if action == "edit":
        edited_input = str((decision or {}).get("user_input") or state["user_input"])
        return Command(update={"user_input": edited_input, "approval": {"decision": decision}, "status": "running", "emitted_events": [event]}, goto="supervisor_plan")
    if action == "approve":
        return Command(update={"approval": {"decision": decision}, "status": "running", "emitted_events": [event]}, goto="dispatch_research")
    return Command(update={"approval": {"decision": decision}, "status": "running", "emitted_events": [event]}, goto="fallback_response")


def _dispatch_research(state: AgentTaskState) -> list[Any]:
    if Send is None:
        raise RuntimeError("LangGraph Send support is required for parallel research.")
    tasks = (state.get("plan") or {}).get("research_tasks") or []
    return [
        Send("research_agent", {
            "session_id": state["session_id"], "task_id": state["task_id"], "run_id": state["run_id"], "username": state["username"], "user_id": state.get("user_id"),
            "kb_id": state.get("kb_id"), "document_id": state.get("document_id"), "conversation_id": state.get("conversation_id"),
            "correlation_id": item["task_id"], "query": item["query"], "objective": item["objective"], "top_k": item.get("top_k", 5),
        })
        for item in tasks
    ]


def _research_agent(work: ResearchWorkItem, store: RuntimeStore) -> dict[str, Any]:
    _enforce_policy(work, store)
    start = _event(work, store, "agent.started", work["objective"], agent_name="research_agent", payload={"correlation_id": work["correlation_id"]})
    result = call_tool(
        "knowledge.answer",
        {
            "query": work["query"], "username": work["username"],
            "user_id": work.get("user_id"), "kb_id": work.get("kb_id"),
            "document_id": work.get("document_id"), "conversation_id": work.get("conversation_id"),
            "top_k": work.get("top_k", 5), "use_memory": True, "rewrite_query": True,
        },
        agent="research_agent",
    )
    observe_ms("agent_tool_latency_ms", float(result.get("latency_ms") or 0), {"tool": "knowledge.answer", "ok": bool(result.get("ok"))})
    increment("agent_tool_calls_total", {"tool": "knowledge.answer", "ok": bool(result.get("ok"))})
    usage = result.get("usage") or {}
    artifact: Artifact = {"artifact_id": f"research-{uuid4().hex[:12]}", "kind": "research", "producer": "research_agent", "correlation_id": work["correlation_id"], "data": result.get("data") or {}, "citations": result.get("citations") or [], "confidence": float(result.get("confidence") or 0), "error": result.get("error")}
    store.save_tool_call({"task_id": work["task_id"], "run_id": work["run_id"], "step_id": work["correlation_id"], "agent_name": "research_agent", "skill_name": "knowledge_grounding", "tool_name": "knowledge.answer", "input": {"query": work["query"], "top_k": work.get("top_k", 5)}, "output": result, "result": result, "ok": bool(result.get("ok")), "error_type": (result.get("error") or {}).get("type"), "error_message": (result.get("error") or {}).get("message"), "retry_count": 0, "latency_ms": result.get("latency_ms")})
    done = _event(work, store, "agent.completed" if result.get("ok") else "agent.failed", "Research Agent finished.", agent_name="research_agent", tool_name="knowledge.answer", step_id=work["correlation_id"], payload={"artifact_id": artifact["artifact_id"], "ok": bool(result.get("ok"))})
    message: AgentMessage = {"message_id": str(uuid4()), "from_agent": "research_agent", "to_agent": "architect_agent", "kind": "result", "correlation_id": work["correlation_id"], "payload": {"artifact_id": artifact["artifact_id"]}}
    updates: dict[str, Any] = {"artifacts": [artifact], "messages": [message], "emitted_events": [start, done]}
    if result.get("error"):
        updates["errors"] = [{"source": "research_agent", "error_type": (result.get("error") or {}).get("type", "research_failed"), "message": (result.get("error") or {}).get("message", "Research failed."), "retryable": bool((result.get("error") or {}).get("retryable", True)), "correlation_id": work["correlation_id"]}]
    return updates


def _architect_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    proposal, error, source = generate_proposal(state, on_token=_llm_token_callback(state, "architect_agent"))
    artifact: Artifact = {"artifact_id": f"proposal-{uuid4().hex[:12]}", "kind": "proposal", "producer": "architect_agent", "correlation_id": "proposal", "data": proposal.model_dump(mode="json"), "citations": [citation for item in state.get("artifacts", []) for citation in item.get("citations", [])], "confidence": 0.8 if error is None else 0.45, "error": {"type": "architect_degraded", "message": error} if error else None}
    event = _event(state, store, "agent.completed", "Architect Agent generated a structured proposal.", agent_name="architect_agent", payload={"artifact_id": artifact["artifact_id"], "degraded": bool(error), "model_source": source})
    return {"proposal": artifact["data"], "artifacts": [artifact], "emitted_events": [event]}


def _verifier_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    decision, error, source = verify_proposal(state, on_token=_llm_token_callback(state, "verifier_agent"))
    verification = decision.model_dump(mode="json")
    verification["error"] = error
    store.save_verification({"task_id": state["task_id"], "run_id": state["run_id"], "step_id": "proposal", "status": "passed" if decision.status == "passed" else "failed", "score": decision.score, "issues": decision.issues, "evidence": {"artifact_count": len(state.get("artifacts", []))}})
    event = _event(state, store, "verification.passed" if decision.status == "passed" else "verification.failed", "Verifier Agent evaluated proposal evidence coverage.", agent_name="verifier_agent", payload={**verification, "model_source": source})

    # ── Feedback Loop: collect failure signals for continuous improvement ──
    if decision.status != "passed":
        all_citations = [c for a in state.get("artifacts", []) for c in a.get("citations", [])]
        for issue in decision.issues:
            record_verification_failure(
                FailureSignal(
                    task_id=state["task_id"],
                    run_id=state.get("run_id"),
                    failure_type=str(issue.get("type", "other")),
                    message=str(issue.get("message", "")),
                    repair_strategy_used="rewrite_query" if decision.status == "repair" else None,
                    kb_id=state.get("kb_id"),
                    query_snippet=str(state.get("user_input", ""))[:200],
                    top_k_used=state.get("budget", {}).get("top_k", 5),
                    source_count=len({c.get("document_id") or c.get("source") for c in all_citations}),
                    citation_count=len(all_citations),
                    confidence=decision.score,
                )
            )

    return {"verification": verification, "emitted_events": [event]}


def _verification_route(state: AgentTaskState) -> str:
    status = (state.get("verification") or {}).get("status")
    if status == "passed":
        return "final_response"
    if status == "needs_approval":
        return "approval_gate"
    if status == "repair" and int(state.get("repair_count") or 0) < MAX_REPAIR_COUNT:
        return "repair_plan"
    return "fallback_response"


def _repair_plan(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    current = state.get("verification") or {}
    queries = current.get("repair_queries") or [state.get("user_input", "")]
    tasks = [ResearchTask(task_id=f"repair-{index + 1}-{uuid4().hex[:6]}", query=str(query), objective="补充 Verifier 指出的缺失证据。", top_k=8).model_dump(mode="json") for index, query in enumerate(queries[:3]) if str(query).strip()]
    event = _event(state, store, "repair.started", "Supervisor dispatched targeted evidence repair.", agent_name="supervisor", payload={"queries": queries})
    return {"repair_count": int(state.get("repair_count") or 0) + 1, "plan": {**(state.get("plan") or {}), "research_tasks": tasks}, "emitted_events": [event]}


def _final_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    proposal = state.get("proposal") or {}
    sections = proposal.get("sections") or []
    lines = [f"# {proposal.get('title', '项目改造方案')}", "", proposal.get("summary", "")]
    for section in sections:
        lines.extend(["", f"## {section.get('title', '建议')}", *[f"- {item}" for item in section.get("items", [])]])
    citations = [citation for item in state.get("artifacts", []) for citation in item.get("citations", [])]
    final_answer = "\n".join(line for line in lines if line is not None).strip()
    _record_memory_event(state, event_type="task_completed", category="project_context", content=final_answer[:1200], metadata={"confidence": (state.get("verification") or {}).get("score", 0.7), "citation_count": len(citations)})
    try:
        memory_updates = consolidate_task_memory({**state, "final_answer": final_answer})
        session_summary = summarize_task_session({**state, "final_answer": final_answer})
    except Exception:  # noqa: BLE001
        memory_updates, session_summary = [], None
    event = _event(state, store, "task.completed", "Multi-agent task completed.", agent_name="runtime", payload={"citation_count": len(citations)})
    return {"status": "completed", "final_answer": final_answer, "citations": citations, "grounding": {"mode": "rag_grounded" if citations else "general", "rag_used": bool(citations)}, "memory_updates": memory_updates, "session_summary": session_summary, "emitted_events": [event]}


def _fallback_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    final_answer = "当前无法取得足够的可验证证据来完成该方案。请补充或索引相关项目材料后重试。"
    event = _event(state, store, "fallback.used", "Evidence is insufficient; returned safe fallback.", agent_name="runtime", payload={"verification": state.get("verification")})
    return {"status": "completed", "final_answer": final_answer, "grounding": {"mode": "insufficient_evidence", "rag_used": False}, "emitted_events": [event]}


def build_agent_graph(store: RuntimeStore, *, checkpointer: Any) -> Any:
    if StateGraph is None:
        raise RuntimeError(f"langgraph is required for Agent Runtime: {LANGGRAPH_IMPORT_ERROR}")
    builder = StateGraph(AgentTaskState)
    builder.add_node("supervisor_plan", lambda state: _supervisor_plan(state, store))
    builder.add_node("approval_gate", lambda state: _approval_gate(state, store))
    builder.add_node("dispatch_research", lambda state: state)
    builder.add_node("research_agent", lambda state: _research_agent(state, store))
    builder.add_node("architect_agent", lambda state: _architect_agent(state, store))
    builder.add_node("verifier_agent", lambda state: _verifier_agent(state, store))
    builder.add_node("repair_plan", lambda state: _repair_plan(state, store))
    builder.add_node("final_response", lambda state: _final_response(state, store))
    builder.add_node("fallback_response", lambda state: _fallback_response(state, store))
    builder.add_edge(START, "supervisor_plan")
    builder.add_conditional_edges("supervisor_plan", _plan_route, {"approval_gate": "approval_gate", "dispatch_research": "dispatch_research"})
    # Dynamically fan-out research tasks via Send; Architect runs as a fan-in
    # barrier once every dispatched Research Agent has completed.
    builder.add_conditional_edges("dispatch_research", _dispatch_research, ["research_agent"])
    builder.add_edge("research_agent", "architect_agent")
    builder.add_edge("architect_agent", "verifier_agent")
    builder.add_conditional_edges("verifier_agent", _verification_route, {"final_response": "final_response", "repair_plan": "repair_plan", "approval_gate": "approval_gate", "fallback_response": "fallback_response"})
    builder.add_edge("repair_plan", "dispatch_research")
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
    final_state = dict(graph.get_state(config).values)
    store.save_task_state(initial_state["task_id"], final_state)
    return final_state


def run_agent_task(initial_state: AgentTaskState, store: RuntimeStore, *, resume_payload: dict[str, Any] | None = None, checkpointer: Any | None = None) -> AgentTaskState:
    if StateGraph is None:
        raise RuntimeError(f"langgraph is required for Agent Runtime: {LANGGRAPH_IMPORT_ERROR}")
    event = _event(initial_state, store, "task.resumed" if resume_payload else "task.started", "Agent task resumed." if resume_payload else "Agent task started.", agent_name="runtime")
    initial_state = {**initial_state, "emitted_events": [event]}
    if checkpointer is not None:
        return _invoke_graph(initial_state, store, checkpointer, resume_payload)
    with postgres_checkpointer() as durable_checkpointer:
        return _invoke_graph(initial_state, store, durable_checkpointer, resume_payload)
