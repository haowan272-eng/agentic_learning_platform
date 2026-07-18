from __future__ import annotations

import operator
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Annotated, Literal, Protocol, TypedDict

from pydantic import BaseModel, ConfigDict, Field


TaskStatus = Literal["pending", "running", "waiting_user", "completed", "failed", "cancelled"]
NextAction = Literal["dispatch", "repair", "replan", "ask_user", "fallback", "complete"]
ChildAgent = Literal["direct_answer_agent", "tool_agent", "research_agent", "architect_agent", "verifier_agent", "diagnostic_agent", "practice_agent", "coach_agent"]
SupervisorRoute = Literal["direct_answer", "tool_planner", "supervisor_plan", "learning_coach", "architect_agent", "verifier_agent", "final_response", "fallback_response"]
ArchitectRoute = Literal["verifier_agent", "final_response"]
ToolExecutorRoute = Literal["tool_response", "architect_agent", "verifier_agent", "fallback_response"]
PlanRoute = Literal["approval_gate", "dispatch_research"]
VerificationRoute = Literal["final_response", "repair_plan", "approval_gate", "fallback_response"]


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
    kind: Literal["delegate", "result", "tool_request", "tool_result", "repair_request", "replan_request", "approval_request"]
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
    supervisor_decision: dict[str, Any]
    supervisor_source: str
    supervisor_error: str | None
    route_decision: dict[str, Any]
    route_source: str
    route_error: str | None
    tool_plan: dict[str, Any]
    tool_feedback: Annotated[dict[str, Any], operator.or_]
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
    session_id: str
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


class RuntimeModel(BaseModel):
    """Strict formatting boundary for runtime payloads.

    LangGraph still consumes TypedDict state above so reducers such as
    `operator.add` work correctly.  These Pydantic models validate and format
    state as it enters and leaves each node.
    """

    model_config = ConfigDict(extra="allow", validate_assignment=True)


class ArtifactModel(RuntimeModel):
    artifact_id: str = Field(..., min_length=1, max_length=128)
    kind: str = Field(..., min_length=1, max_length=64)
    producer: str = Field(..., min_length=1, max_length=64)
    correlation_id: str = Field(..., min_length=1, max_length=128)
    data: dict[str, Any] = Field(default_factory=dict)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    error: dict[str, Any] | None = None


class AgentMessageModel(RuntimeModel):
    message_id: str = Field(..., min_length=1, max_length=128)
    from_agent: str = Field(..., min_length=1, max_length=64)
    to_agent: str = Field(..., min_length=1, max_length=64)
    kind: Literal["delegate", "result", "tool_request", "tool_result", "repair_request", "replan_request", "approval_request"]
    correlation_id: str = Field(..., min_length=1, max_length=128)
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentErrorModel(RuntimeModel):
    source: str = Field(..., min_length=1, max_length=64)
    error_type: str = Field(..., min_length=1, max_length=128)
    message: str = Field(..., min_length=1, max_length=2000)
    retryable: bool = True
    correlation_id: str | None = Field(default=None, max_length=128)


class AgentEventModel(RuntimeModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    task_id: str = Field(..., min_length=1, max_length=128)
    run_id: str | None = Field(default=None, max_length=128)
    event_type: str = Field(..., min_length=1, max_length=128)
    event_index: int | None = Field(default=None, ge=1)
    agent_name: str | None = Field(default=None, max_length=64)
    skill_name: str | None = Field(default=None, max_length=64)
    tool_name: str | None = Field(default=None, max_length=128)
    step_id: str | None = Field(default=None, max_length=128)
    message: str = Field(..., min_length=1, max_length=4000)
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class RuntimeBudgetModel(RuntimeModel):
    deadline_seconds: int = Field(default=900, le=86400)
    max_steps: int = Field(default=8, ge=1, le=64)
    max_tool_calls: int = Field(default=12, ge=0, le=256)
    max_total_tokens: int = Field(default=24000, ge=1000, le=2_000_000)
    max_cost_usd: float = Field(default=2.0, ge=0.0, le=1000.0)
    started_at: str | None = None


class AgentTaskStateModel(RuntimeModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    task_id: str = Field(..., min_length=1, max_length=128)
    run_id: str = Field(..., min_length=1, max_length=128)
    user_id: int | None = Field(default=None, ge=1)
    username: str = Field(..., min_length=1, max_length=128)
    user_input: str = Field(..., min_length=1, max_length=8000)
    task_type: str = Field(default="interview_improvement", min_length=1, max_length=128)
    budget: RuntimeBudgetModel = Field(default_factory=RuntimeBudgetModel)
    token_usage: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    status: TaskStatus = "pending"
    goal: str | None = Field(default=None, max_length=8000)
    intent: str | None = Field(default=None, max_length=128)
    kb_id: int | None = Field(default=None, ge=1)
    document_id: int | None = Field(default=None, ge=1)
    conversation_id: int | None = Field(default=None, ge=1)
    memory_context: dict[str, Any] = Field(default_factory=dict)
    supervisor_decision: dict[str, Any] = Field(default_factory=dict)
    supervisor_source: str | None = Field(default=None, max_length=128)
    supervisor_error: str | None = Field(default=None, max_length=4000)
    route_decision: dict[str, Any] = Field(default_factory=dict)
    route_source: str | None = Field(default=None, max_length=128)
    route_error: str | None = Field(default=None, max_length=4000)
    tool_plan: dict[str, Any] = Field(default_factory=dict)
    tool_feedback: dict[str, Any] = Field(default_factory=dict)
    plan: dict[str, Any] = Field(default_factory=dict)
    planning_source: str | None = Field(default=None, max_length=128)
    planner_error: str | None = Field(default=None, max_length=4000)
    repair_count: int = Field(default=0, ge=0, le=16)
    approval: dict[str, Any] | None = None
    next_action: NextAction | None = None
    proposal: dict[str, Any] = Field(default_factory=dict)
    verification: dict[str, Any] = Field(default_factory=dict)
    final_answer: str | None = Field(default=None, max_length=64000)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    grounding: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactModel] = Field(default_factory=list)
    messages: list[AgentMessageModel] = Field(default_factory=list)
    errors: list[AgentErrorModel] = Field(default_factory=list)
    emitted_events: list[AgentEventModel] = Field(default_factory=list)
    memory_updates: list[dict[str, Any]] = Field(default_factory=list)
    session_summary: dict[str, Any] | None = None


class ResearchWorkItemModel(RuntimeModel):
    session_id: str = Field(..., min_length=1, max_length=128)
    task_id: str = Field(..., min_length=1, max_length=128)
    run_id: str = Field(..., min_length=1, max_length=128)
    username: str = Field(..., min_length=1, max_length=128)
    user_id: int | None = Field(default=None, ge=1)
    kb_id: int | None = Field(default=None, ge=1)
    document_id: int | None = Field(default=None, ge=1)
    conversation_id: int | None = Field(default=None, ge=1)
    correlation_id: str = Field(..., min_length=1, max_length=128)
    query: str = Field(..., min_length=1, max_length=2000)
    objective: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    repair_reason: str | None = Field(default=None, max_length=1000)
    budget: RuntimeBudgetModel = Field(default_factory=RuntimeBudgetModel)


class NodeUpdateModel(RuntimeModel):
    status: TaskStatus | None = None
    user_input: str | None = Field(default=None, min_length=1, max_length=8000)
    memory_context: dict[str, Any] | None = None
    supervisor_decision: dict[str, Any] | None = None
    supervisor_source: str | None = Field(default=None, max_length=128)
    supervisor_error: str | None = Field(default=None, max_length=4000)
    route_decision: dict[str, Any] | None = None
    route_source: str | None = Field(default=None, max_length=128)
    route_error: str | None = Field(default=None, max_length=4000)
    tool_plan: dict[str, Any] | None = None
    tool_feedback: dict[str, Any] | None = None
    plan: dict[str, Any] | None = None
    goal: str | None = Field(default=None, max_length=8000)
    intent: str | None = Field(default=None, max_length=128)
    planning_source: str | None = Field(default=None, max_length=128)
    planner_error: str | None = Field(default=None, max_length=4000)
    repair_count: int | None = Field(default=None, ge=0, le=16)
    approval: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    verification: dict[str, Any] | None = None
    final_answer: str | None = Field(default=None, max_length=64000)
    citations: list[dict[str, Any]] | None = None
    grounding: dict[str, Any] | None = None
    artifacts: list[ArtifactModel] | None = None
    messages: list[AgentMessageModel] | None = None
    errors: list[AgentErrorModel] | None = None
    emitted_events: list[AgentEventModel] | None = None
    memory_updates: list[dict[str, Any]] | None = None
    session_summary: dict[str, Any] | None = None


class PlanRouteModel(RuntimeModel):
    route: PlanRoute


class SupervisorRouteModel(RuntimeModel):
    route: SupervisorRoute


class ArchitectRouteModel(RuntimeModel):
    route: ArchitectRoute


class ToolExecutorRouteModel(RuntimeModel):
    route: ToolExecutorRoute


class VerificationRouteModel(RuntimeModel):
    route: VerificationRoute


def _dump_model(model: BaseModel) -> dict[str, Any]:
    return model.model_dump(mode="python", exclude_none=True)


def validate_agent_state(state: dict[str, Any]) -> dict[str, Any]:
    return _dump_model(AgentTaskStateModel.model_validate(state))


def validate_research_work_item(work: dict[str, Any]) -> dict[str, Any]:
    return _dump_model(ResearchWorkItemModel.model_validate(work))


def validate_node_update(update: dict[str, Any]) -> dict[str, Any]:
    return _dump_model(NodeUpdateModel.model_validate(update))


def validate_agent_event(event: dict[str, Any]) -> dict[str, Any]:
    return _dump_model(AgentEventModel.model_validate(event))


def validate_plan_route(route: str) -> PlanRoute:
    return PlanRouteModel(route=route).route


def validate_supervisor_route(route: str) -> SupervisorRoute:
    return SupervisorRouteModel(route=route).route


def validate_architect_route(route: str) -> ArchitectRoute:
    return ArchitectRouteModel(route=route).route


def validate_tool_executor_route(route: str) -> ToolExecutorRoute:
    return ToolExecutorRouteModel(route=route).route


def validate_verification_route(route: str) -> VerificationRoute:
    return VerificationRouteModel(route=route).route


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
