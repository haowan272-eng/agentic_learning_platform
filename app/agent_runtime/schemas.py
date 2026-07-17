from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Annotated, Literal, Protocol, TypedDict


TaskStatus = Literal["pending", "running", "waiting_user", "completed", "failed", "cancelled"]
NextAction = Literal["dispatch", "repair", "replan", "ask_user", "fallback", "complete"]


class Artifact(TypedDict, total=False):
    """An immutable, auditable result published by an agent.

    Agents communicate through artifacts rather than by mutating one another's
    private state.  Every artifact must identify its producer and correlation
    id so the Supervisor can safely aggregate parallel work.
    """

    artifact_id: str
    kind: str
    producer: str
    correlation_id: str
    data: dict[str, Any]
    citations: list[dict[str, Any]]
    confidence: float
    error: dict[str, Any] | None


class AgentMessage(TypedDict, total=False):
    message_id: str
    from_agent: str
    to_agent: str
    kind: Literal["delegate", "result", "repair_request", "replan_request", "approval_request"]
    correlation_id: str
    payload: dict[str, Any]


class AgentError(TypedDict, total=False):
    source: str
    error_type: str
    message: str
    retryable: bool
    correlation_id: str


class AgentTaskState(TypedDict, total=False):
    """Root state for the durable multi-agent orchestration graph.

    Append-only channels use reducers, which makes `Send` fan-out safe.  Task
    identity and scopes are injected by the API/worker, never accepted from an
    LLM-generated plan or tool arguments.
    """

    session_id: str
    task_id: str
    run_id: str
    user_id: int | None
    username: str
    user_input: str
    task_type: str
    budget: dict[str, Any]
    token_usage: int
    estimated_cost_usd: float
    status: TaskStatus
    goal: str
    intent: str
    kb_id: int | None
    document_id: int | None
    conversation_id: int | None
    memory_context: dict[str, Any]
    plan: dict[str, Any]
    planning_source: str
    planner_error: str | None
    repair_count: int
    approval: dict[str, Any] | None
    next_action: NextAction
    proposal: dict[str, Any]
    verification: dict[str, Any]
    final_answer: str
    citations: list[dict[str, Any]]
    grounding: dict[str, Any]
    artifacts: Annotated[list[Artifact], operator.add]
    messages: Annotated[list[AgentMessage], operator.add]
    errors: Annotated[list[AgentError], operator.add]
    emitted_events: Annotated[list["AgentEvent"], operator.add]
    memory_updates: list[dict[str, Any]]
    session_summary: dict[str, Any] | None


class ResearchWorkItem(TypedDict, total=False):
    task_id: str
    run_id: str
    username: str
    user_id: int | None
    kb_id: int | None
    document_id: int | None
    conversation_id: int | None
    correlation_id: str
    query: str
    objective: str
    top_k: int
    repair_reason: str | None


class ToolResult(TypedDict, total=False):
    ok: bool
    tool_name: str
    data: dict[str, Any]
    confidence: float
    citations: list[dict[str, Any]]
    grounding: dict[str, Any]
    trace: list[dict[str, Any]]
    latency_ms: int
    usage: dict[str, int]
    error: dict[str, Any] | None


class ToolCall(TypedDict, total=False):
    call_id: str
    tool_name: str
    input: dict[str, Any]
    depends_on: list[str]


class ToolPlanResult(TypedDict, total=False):
    calls: dict[str, ToolResult]
    execution_order: list[list[str]]
    error: dict[str, Any] | None


class AgentEvent(TypedDict, total=False):
    session_id: str
    task_id: str
    run_id: str | None
    event_type: str
    event_index: int
    agent_name: str | None
    skill_name: str | None
    tool_name: str | None
    step_id: str | None
    message: str
    payload: dict[str, Any]
    created_at: datetime


class RuntimeStore(Protocol):
    def append_event(self, event: AgentEvent) -> int | None: ...
    def save_task_state(self, task_id: str, state: dict[str, Any]) -> None: ...
    def save_plan(self, payload: dict[str, Any]) -> None: ...
    def upsert_step(self, payload: dict[str, Any]) -> None: ...
    def save_tool_call(self, payload: dict[str, Any]) -> None: ...
    def save_verification(self, payload: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class ToolDescriptor:
    name: str
    category: str
    description: str
    input_schema: dict[str, Any]
    retryable: bool = True
    timeout_seconds: float = 30.0
    risk_level: Literal["safe", "restricted", "sandboxed"] = "safe"
    side_effect: Literal["none", "read", "write"] = "none"
    version: str = "1"
    requires_approval: bool = False
