from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Literal
from uuid import uuid4

from app.agent_runtime.event_bus import publish_task_event
from app.core.config import DATABASE_URL, LANGGRAPH_CHECKPOINT_SETUP
from app.core.database import SessionLocal
from app.memory.service import build_context_for_state, consolidate_task_memory, record_memory_event, summarize_task_session
from app.memory.short_term import append_recent_event
from app.observability import increment
from app.services.learning_service import record_agent_learning_outputs

from .llm_gateway import llm_gateway
from .planner import AgentPlan, ResearchTask, generate_plan, generate_proposal, generate_supervisor_decision, generate_tool_plan, verify_proposal
from .schemas import (
    AgentEvent,
    AgentMessage,
    AgentTaskState,
    Artifact,
    ResearchWorkItem,
    RuntimeStore,
    validate_architect_route,
    validate_agent_event,
    validate_agent_state,
    validate_node_update,
    validate_plan_route,
    validate_research_work_item,
    validate_supervisor_route,
    validate_tool_executor_route,
    validate_verification_route,
)
from .tool_manager import execute_managed_tool
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
    return validate_supervisor_route(str(decision.get("route") or "supervisor_plan"))


def _direct_answer(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    source = "fallback"
    try:
        response = llm_gateway.invoke(
            role="planner",
            prompt=(
                "你是学习提升平台的轻量答疑助手。请直接回答用户问题，"
                "不要声称执行了知识库检索、工具调用或多 Agent 工作流。\n\n"
                f"用户问题：{state.get('user_input', '')}"
            ),
            on_token=_llm_token_callback(state, "direct_answer"),
        )
        answer = response.content.strip()
        source = f"{response.provider}:{response.model}"
    except Exception:  # noqa: BLE001
        answer = (
            "这是一个轻量问题，不需要进入完整多 Agent 提优链路。"
            f"\n\n你的问题：{state.get('user_input', '')}"
        )
    event = _event(
        state,
        store,
        "task.completed",
        "Router completed the task through direct answer.",
        agent_name="direct_answer",
        payload={"route_decision": state.get("route_decision") or {}, "model_source": source},
    )
    return validate_node_update({
        "status": "completed",
        "final_answer": answer,
        "grounding": {"mode": "direct", "rag_used": False},
        "emitted_events": [event],
    })


def _tool_planner(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    plan, source, error = generate_tool_plan(state, on_token=_llm_token_callback(state, "tool_planner"))
    plan_payload = plan.model_dump(mode="json")
    event = _event(
        state,
        store,
        "tool.plan_created",
        "Tool Planner created an autonomous tool call plan.",
        agent_name="tool_planner",
        payload={"source": source, "plan": plan_payload, "error": error},
    )
    return validate_node_update({
        "tool_plan": {**plan_payload, "source": source, "error": error},
        "emitted_events": [event],
    })


def _tool_executor(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    plan = state.get("tool_plan") or {}
    calls = list(plan.get("calls") or [])
    artifacts: list[Artifact] = []
    errors: list[dict[str, Any]] = []
    result_summaries: list[dict[str, Any]] = []
    emitted_events: list[AgentEvent] = []

    for index, call in enumerate(calls[:4]):
        tool_name = str(call.get("tool_name") or "")
        arguments = dict(call.get("arguments") or {})
        call_id = str(call.get("call_id") or f"tool-{index + 1}")
        started = _event(state, store, "tool.started", f"Executing {tool_name}.", agent_name="tool_executor", tool_name=tool_name, step_id=call_id, payload={"reason": call.get("reason")})
        emitted_events.append(started)
        execution = execute_managed_tool(
            state,
            store,
            tool_name=tool_name,
            arguments=arguments,
            agent_name="tool_agent",
            skill_name="autonomous_tool_use",
            step_id=call_id,
            call_id=call_id,
        )
        artifacts.append(execution.artifact)
        result = execution.result
        error = result.get("error") or {}
        result_summaries.append({
            "call_id": call_id,
            "tool_name": tool_name,
            "ok": bool(result.get("ok")),
            "artifact_id": execution.artifact["artifact_id"],
            "confidence": float(result.get("confidence") or 0),
            "grounding": result.get("grounding") or {},
            "error": error or None,
        })
        done = _event(state, store, "tool.completed" if result.get("ok") else "tool.failed", f"Tool {tool_name} completed.", agent_name="tool_executor", tool_name=tool_name, step_id=call_id, payload=result_summaries[-1])
        emitted_events.append(done)
        if error:
            errors.append({"source": "tool_executor", "error_type": error.get("type", "tool_failed"), "message": error.get("message", "Tool execution failed."), "retryable": bool(error.get("retryable", True)), "correlation_id": call_id})
        if tool_name == "knowledge.answer" and error.get("retryable", True) and not result.get("ok"):
            repair_id = f"{call_id}-repair"
            repair_event = _event(state, store, "tool.feedback_repair", "Tool feedback triggered retrieval repair.", agent_name="tool_executor", tool_name="knowledge.repair_retrieval", step_id=repair_id, payload={"failed_call_id": call_id, "error": error})
            emitted_events.append(repair_event)
            repair_execution = execute_managed_tool(
                state,
                store,
                tool_name="knowledge.repair_retrieval",
                arguments={**arguments, "repair_reason": error.get("type") or "retrieval_insufficient"},
                agent_name="tool_agent",
                skill_name="autonomous_tool_use",
                step_id=repair_id,
                call_id=repair_id,
            )
            artifacts.append(repair_execution.artifact)
            repair_result = repair_execution.result
            repair_error = repair_result.get("error") or {}
            result_summaries.append({
                "call_id": repair_id,
                "tool_name": "knowledge.repair_retrieval",
                "ok": bool(repair_result.get("ok")),
                "artifact_id": repair_execution.artifact["artifact_id"],
                "confidence": float(repair_result.get("confidence") or 0),
                "grounding": repair_result.get("grounding") or {},
                "error": repair_error or None,
                "feedback_for": call_id,
            })
            repair_done = _event(state, store, "tool.completed" if repair_result.get("ok") else "tool.failed", "Retrieval repair completed.", agent_name="tool_executor", tool_name="knowledge.repair_retrieval", step_id=repair_id, payload=result_summaries[-1])
            emitted_events.append(repair_done)
            if repair_error:
                errors.append({"source": "tool_executor", "error_type": repair_error.get("type", "tool_failed"), "message": repair_error.get("message", "Tool repair failed."), "retryable": bool(repair_error.get("retryable", True)), "correlation_id": repair_id})

    success_count = len([item for item in result_summaries if item.get("ok")])
    retryable_failures = len([item for item in result_summaries if (item.get("error") or {}).get("retryable")])
    planned_next = str(plan.get("next_node") or "tool_response")
    if success_count == 0 and result_summaries and retryable_failures == 0:
        next_node = "fallback_response"
    elif bool(plan.get("stop_after_tools", True)):
        next_node = "tool_response"
    else:
        next_node = planned_next
    feedback = {
        "success_count": success_count,
        "failure_count": max(0, len(result_summaries) - success_count),
        "retryable_failures": retryable_failures,
        "next_node": next_node,
        "results": result_summaries,
    }
    event = _event(state, store, "tool.feedback_ready", "Tool Executor produced structured feedback.", agent_name="tool_executor", payload=feedback)
    emitted_events.append(event)
    return validate_node_update({"tool_feedback": feedback, "artifacts": artifacts, "errors": errors, "emitted_events": emitted_events})


def _tool_executor_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    feedback = state.get("tool_feedback") or {}
    return validate_tool_executor_route(str(feedback.get("next_node") or "tool_response"))


def _tool_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    feedback = state.get("tool_feedback") or {}
    artifacts = state.get("artifacts") or []
    answer = ""
    for artifact in reversed(artifacts):
        data = artifact.get("data") or {}
        if data.get("answer"):
            answer = str(data["answer"])
            break
    if not answer:
        answer = "工具已执行，但没有形成足够明确的答案。请补充材料或换一个更具体的问题。"
    citations = [citation for artifact in artifacts for citation in artifact.get("citations", [])]
    grounding = next((artifact.get("grounding") for artifact in reversed(artifacts) if artifact.get("grounding")), None)
    event = _event(state, store, "task.completed", "Tool response completed from autonomous tool feedback.", agent_name="tool_response", payload={"tool_feedback": feedback})
    return validate_node_update({
        "status": "completed",
        "final_answer": answer,
        "citations": citations,
        "grounding": grounding or {"mode": "tool_response", "rag_used": bool(citations)},
        "emitted_events": [event],
    })


def _diagnostic_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    user_input = str(state.get("user_input") or "").strip()
    topics = []
    for keyword in ["系统设计", "项目表达", "RAG", "Agent", "数据库", "缓存", "索引", "面试"]:
        if keyword.lower() in user_input.lower():
            topics.append(keyword)
    if not topics:
        topics = ["学习目标澄清", "知识结构", "表达闭环"]
    diagnosis = {
        "goal": user_input,
        "target_role": "面试提优" if "面试" in user_input else "学习提升",
        "current_level": "needs_diagnostic",
        "weaknesses": [
            {"topic": topic, "severity": round(0.72 - index * 0.1, 2), "category": "interview" if topic in {"项目表达", "面试"} else "knowledge"}
            for index, topic in enumerate(topics[:3])
        ],
        "success_criteria": ["能用项目证据回答", "能解释关键取舍", "能完成追问复盘"],
    }
    artifact: Artifact = {
        "artifact_id": f"diagnostic-{uuid4().hex[:12]}",
        "kind": "learning_diagnostic",
        "producer": "diagnostic_agent",
        "correlation_id": "diagnostic",
        "data": diagnosis,
        "citations": [],
        "confidence": 0.72,
        "error": None,
    }
    event = _event(state, store, "agent.completed", "Diagnostic Agent created a learning profile and weakness map.", agent_name="diagnostic_agent", payload={"artifact_id": artifact["artifact_id"], "weakness_count": len(diagnosis["weaknesses"])})
    _record_memory_event(state, event_type="learning_diagnostic", category="learning_goal", content=user_input[:1200], metadata={"weaknesses": diagnosis["weaknesses"]})
    return validate_node_update({"artifacts": [artifact], "memory_context": {"learning_diagnostic": diagnosis}, "emitted_events": [event]})


def _practice_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    diagnostic = next((item for item in reversed(state.get("artifacts") or []) if item.get("kind") == "learning_diagnostic"), {})
    weaknesses = ((diagnostic.get("data") or {}).get("weaknesses") or [])[:3]
    practices = []
    for item in weaknesses or [{"topic": "学习目标澄清", "severity": 0.5}]:
        topic = str(item.get("topic") or "学习目标澄清")
        practices.append({
            "topic": topic,
            "difficulty": "hard" if float(item.get("severity") or 0.5) >= 0.65 else "medium",
            "question": f"围绕「{topic}」做一次面试式回答：先讲核心概念，再讲项目证据，最后讲风险和改进。",
            "expected_answer": "包含概念、证据、取舍、风险、追问准备。",
        })
    artifact: Artifact = {
        "artifact_id": f"practice-{uuid4().hex[:12]}",
        "kind": "learning_practice_plan",
        "producer": "practice_agent",
        "correlation_id": "practice",
        "data": {"practices": practices},
        "citations": [],
        "confidence": 0.74,
        "error": None,
    }
    event = _event(state, store, "agent.completed", "Practice Agent generated targeted exercises.", agent_name="practice_agent", payload={"artifact_id": artifact["artifact_id"], "practice_count": len(practices)})
    return validate_node_update({"artifacts": [artifact], "emitted_events": [event]})


def _coach_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    diagnostic = next((item for item in reversed(state.get("artifacts") or []) if item.get("kind") == "learning_diagnostic"), {})
    practice_plan = next((item for item in reversed(state.get("artifacts") or []) if item.get("kind") == "learning_practice_plan"), {})
    weaknesses = ((diagnostic.get("data") or {}).get("weaknesses") or [])[:3]
    practices = ((practice_plan.get("data") or {}).get("practices") or [])[:3]
    lines = [
        "# 学习提优短链路结果",
        "",
        "## 诊断",
        f"- 目标：{(diagnostic.get('data') or {}).get('goal') or state.get('user_input')}",
        *[f"- 薄弱点：{item.get('topic')}，优先级 {item.get('severity')}" for item in weaknesses],
        "",
        "## 练习",
        *[f"- {item.get('question')}" for item in practices],
        "",
        "## 教练建议",
        "- 今天先完成 1 道高优先级练习，并把回答压缩到 3 分钟。",
        "- 每次回答后记录一个证据、一个风险、一个可被追问的问题。",
        "- 未来 3 天按复习项回看薄弱点，优先补齐项目证据。",
    ]
    final_answer = "\n".join(lines)
    try:
        with SessionLocal() as db:
            record_agent_learning_outputs(db, {**state, "citations": state.get("citations") or []}, final_answer)
    except Exception:  # noqa: BLE001
        pass
    event = _event(state, store, "task.completed", "Coach Agent persisted learning profile, practice, and review items.", agent_name="coach_agent", payload={"practice_count": len(practices), "weakness_count": len(weaknesses)})
    return validate_node_update({
        "status": "completed",
        "final_answer": final_answer,
        "grounding": {"mode": "learning_coach", "rag_used": False},
        "emitted_events": [event],
    })


def _supervisor_plan(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
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
    return validate_node_update({"memory_context": memory_context, "plan": plan_payload, "goal": plan.goal, "intent": plan.intent, "planning_source": source, "planner_error": error, "status": status, "artifacts": [memory_artifact], "emitted_events": [memory_event, plan_event]})


def _plan_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    route = "approval_gate" if (state.get("plan") or {}).get("approval_required") else "dispatch_research"
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
        return Command(update=validate_node_update({"user_input": edited_input, "approval": {"decision": decision}, "status": "running", "emitted_events": [event]}), goto="supervisor_plan")
    if action == "approve":
        return Command(update=validate_node_update({"approval": {"decision": decision}, "status": "running", "emitted_events": [event]}), goto="dispatch_research")
    return Command(update=validate_node_update({"approval": {"decision": decision}, "status": "running", "emitted_events": [event]}), goto="fallback_response")


def _dispatch_research(state: AgentTaskState) -> list[Any]:
    state = validate_agent_state(state)
    if Send is None:
        raise RuntimeError("LangGraph Send support is required for parallel research.")
    tasks = (state.get("plan") or {}).get("research_tasks") or []
    return [
        Send("research_agent", validate_research_work_item({
            "session_id": state["session_id"], "task_id": state["task_id"], "run_id": state["run_id"], "username": state["username"], "user_id": state.get("user_id"),
            "kb_id": state.get("kb_id"), "document_id": state.get("document_id"), "conversation_id": state.get("conversation_id"),
            "correlation_id": item["task_id"], "query": item["query"], "objective": item["objective"], "top_k": item.get("top_k", 5),
            "budget": state.get("budget") or {},
        }))
        for item in tasks
    ]


def _research_agent(work: ResearchWorkItem, store: RuntimeStore) -> dict[str, Any]:
    work = validate_research_work_item(work)
    try:
        _enforce_policy(work, store)
    except AgentBudgetExceeded as exc:
        event = _event(work, store, "agent.failed", str(exc), agent_name="research_agent", step_id=work["correlation_id"], payload={"error_type": "budget_exceeded"})
        artifact: Artifact = {
            "artifact_id": f"research-{uuid4().hex[:12]}",
            "kind": "research",
            "producer": "research_agent",
            "correlation_id": work["correlation_id"],
            "data": {},
            "citations": [],
            "confidence": 0.0,
            "error": {"type": "budget_exceeded", "message": str(exc), "retryable": False},
        }
        message: AgentMessage = {"message_id": str(uuid4()), "from_agent": "research_agent", "to_agent": "architect_agent", "kind": "result", "correlation_id": work["correlation_id"], "payload": {"artifact_id": artifact["artifact_id"], "error": artifact["error"]}}
        return validate_node_update({
            "artifacts": [artifact],
            "messages": [message],
            "errors": [{"source": "research_agent", "error_type": "budget_exceeded", "message": str(exc), "retryable": False, "correlation_id": work["correlation_id"]}],
            "emitted_events": [event],
        })
    start = _event(work, store, "agent.started", work["objective"], agent_name="research_agent", payload={"correlation_id": work["correlation_id"]})
    tool_arguments = {"query": work["query"], "top_k": work.get("top_k", 5), "use_memory": True, "rewrite_query": True}
    tool_request: AgentMessage = {
        "message_id": str(uuid4()),
        "from_agent": "research_agent",
        "to_agent": "tool_agent",
        "kind": "tool_request",
        "correlation_id": work["correlation_id"],
        "payload": {
            "tool_name": "knowledge.answer",
            "arguments": tool_arguments,
            "reason": work["objective"],
        },
    }
    requested = _event(
        work,
        store,
        "tool.requested",
        "Research Agent requested evidence retrieval from Tool Agent.",
        agent_name="research_agent",
        tool_name="knowledge.answer",
        step_id=work["correlation_id"],
        payload={"request": tool_request},
    )
    execution = execute_managed_tool(
        work,
        store,
        tool_name="knowledge.answer",
        arguments=tool_arguments,
        agent_name="tool_agent",
        skill_name="knowledge_grounding",
        step_id=work["correlation_id"],
    )
    result = execution.result
    artifact: Artifact = execution.artifact
    feedback_item = {
        "requester": "research_agent",
        "executor": "tool_agent",
        "call_id": execution.call_id,
        "tool_name": execution.tool_name,
        "ok": bool(result.get("ok")),
        "artifact_id": artifact["artifact_id"],
        "confidence": float(result.get("confidence") or 0),
        "grounding": result.get("grounding") or {},
        "error": result.get("error"),
    }
    tool_done = _event(
        work,
        store,
        "tool.completed" if result.get("ok") else "tool.failed",
        "Tool Agent completed Research Agent tool request.",
        agent_name="tool_agent",
        tool_name="knowledge.answer",
        step_id=work["correlation_id"],
        payload=feedback_item,
    )
    done = _event(work, store, "agent.completed" if result.get("ok") else "agent.failed", "Research Agent received Tool Agent feedback.", agent_name="research_agent", tool_name="knowledge.answer", step_id=work["correlation_id"], payload={"artifact_id": artifact["artifact_id"], "ok": bool(result.get("ok")), "tool_feedback": feedback_item})
    message: AgentMessage = {"message_id": str(uuid4()), "from_agent": "research_agent", "to_agent": "architect_agent", "kind": "result", "correlation_id": work["correlation_id"], "payload": {"artifact_id": artifact["artifact_id"]}}
    tool_result_message: AgentMessage = {
        "message_id": str(uuid4()),
        "from_agent": "tool_agent",
        "to_agent": "research_agent",
        "kind": "tool_result",
        "correlation_id": work["correlation_id"],
        "payload": feedback_item,
    }
    updates: dict[str, Any] = {
        "artifacts": [artifact],
        "messages": [tool_request, tool_result_message, message],
        "tool_feedback": {work["correlation_id"]: feedback_item},
        "emitted_events": [start, requested, tool_done, done],
    }
    if result.get("error"):
        updates["errors"] = [{"source": "tool_agent", "error_type": (result.get("error") or {}).get("type", "research_tool_failed"), "message": (result.get("error") or {}).get("message", "Research tool request failed."), "retryable": bool((result.get("error") or {}).get("retryable", True)), "correlation_id": work["correlation_id"]}]
    return validate_node_update(updates)


def _architect_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    proposal, error, source = generate_proposal(state, on_token=_llm_token_callback(state, "architect_agent"))
    artifact: Artifact = {"artifact_id": f"proposal-{uuid4().hex[:12]}", "kind": "proposal", "producer": "architect_agent", "correlation_id": "proposal", "data": proposal.model_dump(mode="json"), "citations": [citation for item in state.get("artifacts", []) for citation in item.get("citations", [])], "confidence": 0.8 if error is None else 0.45, "error": {"type": "architect_degraded", "message": error} if error else None}
    event = _event(state, store, "agent.completed", "Architect Agent generated a structured proposal.", agent_name="architect_agent", payload={"artifact_id": artifact["artifact_id"], "degraded": bool(error), "model_source": source})
    return validate_node_update({"proposal": artifact["data"], "artifacts": [artifact], "emitted_events": [event]})


def _architect_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    decision = state.get("route_decision") or {}
    if decision.get("target_node") == "architect_agent" and not bool(decision.get("needs_verification", True)):
        return validate_architect_route("final_response")
    return validate_architect_route("verifier_agent")


def _verifier_agent(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    decision, error, source = verify_proposal(state, on_token=_llm_token_callback(state, "verifier_agent"))
    verification = decision.model_dump(mode="json")
    verification["error"] = error
    store.save_verification({"task_id": state["task_id"], "run_id": state["run_id"], "step_id": "proposal", "status": "passed" if decision.status == "passed" else "failed", "score": decision.score, "issues": decision.issues, "evidence": {"artifact_count": len(state.get("artifacts", []))}})
    event = _event(state, store, "verification.passed" if decision.status == "passed" else "verification.failed", "Verifier Agent evaluated proposal evidence coverage.", agent_name="verifier_agent", payload={**verification, "model_source": source})

    # ── Feedback Loop: collect failure signals for continuous improvement ──
    if decision.status != "passed":
        all_citations = [c for a in state.get("artifacts", []) for c in a.get("citations", [])]
        for issue in decision.issues:
            try:
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
            except Exception:  # noqa: BLE001
                continue

    return validate_node_update({"verification": verification, "emitted_events": [event]})


def _verification_route(state: AgentTaskState) -> str:
    state = validate_agent_state(state)
    status = (state.get("verification") or {}).get("status")
    if status == "passed":
        return validate_verification_route("final_response")
    if status == "needs_approval":
        return validate_verification_route("approval_gate")
    if status == "repair" and int(state.get("repair_count") or 0) < MAX_REPAIR_COUNT:
        return validate_verification_route("repair_plan")
    return validate_verification_route("fallback_response")


def _repair_plan(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    current = state.get("verification") or {}
    queries = current.get("repair_queries") or [state.get("user_input", "")]
    tasks = [ResearchTask(task_id=f"repair-{index + 1}-{uuid4().hex[:6]}", query=str(query), objective="补充 Verifier 指出的缺失证据。", top_k=8).model_dump(mode="json") for index, query in enumerate(queries[:3]) if str(query).strip()]
    event = _event(state, store, "repair.started", "Supervisor dispatched targeted evidence repair.", agent_name="supervisor", payload={"queries": queries})
    return validate_node_update({"repair_count": int(state.get("repair_count") or 0) + 1, "plan": {**(state.get("plan") or {}), "research_tasks": tasks}, "emitted_events": [event]})


def _final_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
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
    return validate_node_update({"status": "completed", "final_answer": final_answer, "citations": citations, "grounding": {"mode": "rag_grounded" if citations else "general", "rag_used": bool(citations)}, "memory_updates": memory_updates, "session_summary": session_summary, "emitted_events": [event]})


def _fallback_response(state: AgentTaskState, store: RuntimeStore) -> dict[str, Any]:
    state = validate_agent_state(state)
    final_answer = "当前无法取得足够的可验证证据来完成该方案。请补充或索引相关项目材料后重试。"
    event = _event(state, store, "fallback.used", "Evidence is insufficient; returned safe fallback.", agent_name="runtime", payload={"verification": state.get("verification")})
    return validate_node_update({"status": "completed", "final_answer": final_answer, "grounding": {"mode": "insufficient_evidence", "rag_used": False}, "emitted_events": [event]})


def build_agent_graph(store: RuntimeStore, *, checkpointer: Any) -> Any:
    if StateGraph is None:
        raise RuntimeError(f"langgraph is required for Agent Runtime: {LANGGRAPH_IMPORT_ERROR}")
    builder = StateGraph(AgentTaskState)
    builder.add_node("supervisor_agent", lambda state: _supervisor_agent(state, store))
    builder.add_node("direct_answer", lambda state: _direct_answer(state, store))
    builder.add_node("tool_planner", lambda state: _tool_planner(state, store))
    builder.add_node("tool_executor", lambda state: _tool_executor(state, store))
    builder.add_node("tool_response", lambda state: _tool_response(state, store))
    builder.add_node("diagnostic_agent", lambda state: _diagnostic_agent(state, store))
    builder.add_node("practice_agent", lambda state: _practice_agent(state, store))
    builder.add_node("coach_agent", lambda state: _coach_agent(state, store))
    builder.add_node("supervisor_plan", lambda state: _supervisor_plan(state, store))
    builder.add_node("approval_gate", lambda state: _approval_gate(state, store))
    builder.add_node("dispatch_research", lambda state: state)
    builder.add_node("research_agent", lambda state: _research_agent(state, store))
    builder.add_node("architect_agent", lambda state: _architect_agent(state, store))
    builder.add_node("verifier_agent", lambda state: _verifier_agent(state, store))
    builder.add_node("repair_plan", lambda state: _repair_plan(state, store))
    builder.add_node("final_response", lambda state: _final_response(state, store))
    builder.add_node("fallback_response", lambda state: _fallback_response(state, store))
    builder.add_edge(START, "supervisor_agent")
    builder.add_conditional_edges("supervisor_agent", _supervisor_route, {
        "direct_answer": "direct_answer",
        "tool_planner": "tool_planner",
        "supervisor_plan": "supervisor_plan",
        "learning_coach": "diagnostic_agent",
        "architect_agent": "architect_agent",
        "verifier_agent": "verifier_agent",
        "final_response": "final_response",
        "fallback_response": "fallback_response",
    })
    builder.add_edge("tool_planner", "tool_executor")
    builder.add_conditional_edges("tool_executor", _tool_executor_route, {"tool_response": "tool_response", "architect_agent": "architect_agent", "verifier_agent": "verifier_agent", "fallback_response": "fallback_response"})
    builder.add_edge("diagnostic_agent", "practice_agent")
    builder.add_edge("practice_agent", "coach_agent")
    builder.add_conditional_edges("supervisor_plan", _plan_route, {"approval_gate": "approval_gate", "dispatch_research": "dispatch_research"})
    # Dynamically fan-out research tasks via Send; Architect runs as a fan-in
    # barrier once every dispatched Research Agent has completed.
    builder.add_conditional_edges("dispatch_research", _dispatch_research, ["research_agent"])
    builder.add_edge("research_agent", "architect_agent")
    builder.add_conditional_edges("architect_agent", _architect_route, {"verifier_agent": "verifier_agent", "final_response": "final_response"})
    builder.add_conditional_edges("verifier_agent", _verification_route, {"final_response": "final_response", "repair_plan": "repair_plan", "approval_gate": "approval_gate", "fallback_response": "fallback_response"})
    builder.add_edge("repair_plan", "dispatch_research")
    builder.add_edge("direct_answer", END)
    builder.add_edge("tool_response", END)
    builder.add_edge("coach_agent", END)
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
