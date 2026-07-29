from __future__ import annotations

import json
import logging
import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.core.config import AGENT_PLANNER_CANDIDATE_COUNT
from app.services.learning_interview import INTERVIEW_SCORING_RUBRIC

from .llm_gateway import TokenCallback, llm_gateway
from .prompt_registry import ABMetric, prompt_registry
from .scenarios import build_scenario_plan, scenario_for_task_type
from .schemas import ChildAgent, SupervisorRoute
from .source_policy import resolve_source_policy

logger = logging.getLogger(__name__)


class ResearchTask(BaseModel):
    task_id: str = Field(default_factory=lambda: f"research-{uuid4().hex[:8]}")
    query: str = Field(..., min_length=1, max_length=2000)
    objective: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(default=5, ge=1, le=20)


class AgentPlan(BaseModel):
    goal: str = Field(..., min_length=1, max_length=1000)
    intent: str = Field(..., min_length=1, max_length=128)
    research_tasks: list[ResearchTask] = Field(..., min_length=1, max_length=4)
    approval_required: bool = False
    approval_reason: str | None = Field(default=None, max_length=500)


class PlanSelection(BaseModel):
    selected_index: int = Field(ge=0, le=2)
    rationale: str = Field(..., min_length=1, max_length=500)


class SupervisorDecision(BaseModel):
    child_agents: list[ChildAgent] = Field(..., min_length=1, max_length=5)
    route: SupervisorRoute
    intent: str = Field(..., min_length=1, max_length=128)
    reason: str = Field(..., min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)
    needs_rag: bool
    needs_tools: bool
    needs_verification: bool
    stop_after_children: bool
    response_mode: Literal["answer", "research"]
    query: str | None = Field(default=None, max_length=2000)

    @model_validator(mode="after")
    def validate_route_consistency(self) -> "SupervisorDecision":
        expected_agents = _child_agents_for_route(
            self.route,
            needs_verification=self.needs_verification,
        )
        if self.child_agents != expected_agents:
            raise ValueError(
                f"child_agents must be {expected_agents!r} when route={self.route!r} "
                f"and needs_verification={self.needs_verification!r}."
            )
        if self.route == "answer" and self.needs_verification:
            raise ValueError("answer cannot require research verification.")
        if self.route == "research" and not self.needs_verification:
            raise ValueError("research must set needs_verification=true.")
        return self


class ToolCallSpec(BaseModel):
    call_id: str = Field(default_factory=lambda: f"tool-{uuid4().hex[:8]}", min_length=1, max_length=128)
    tool_name: str = Field(..., min_length=1, max_length=128)
    arguments: dict[str, Any] = Field(default_factory=dict)
    reason: str = Field(..., min_length=1, max_length=500)


class AnswerAgentDecision(BaseModel):
    calls: list[ToolCallSpec] = Field(default_factory=list, max_length=4)
    next_action: Literal["complete", "fallback"]
    reason: str = Field(..., min_length=1, max_length=1000)
    confidence: float = Field(ge=0, le=1)


class ResearchSourceDecision(BaseModel):
    tool_names: list[str] = Field(..., min_length=1, max_length=3)
    reason: str = Field(..., min_length=1, max_length=1000)


class ProposalSection(BaseModel):
    title: str = Field(..., min_length=1, max_length=120)
    items: list[str] = Field(..., min_length=1, max_length=8)
    evidence_artifact_ids: list[str] = Field(default_factory=list)


class UpgradeProposal(BaseModel):
    title: str = Field(..., min_length=1, max_length=160)
    summary: str = Field(..., min_length=1, max_length=1600)
    sections: list[ProposalSection] = Field(..., min_length=1, max_length=8)
    open_questions: list[str] = Field(default_factory=list, max_length=6)


class VerificationDecision(BaseModel):
    status: str = Field(pattern=r"^(passed|repair|fallback|needs_approval)$")
    score: float = Field(ge=0, le=1)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    repair_queries: list[str] = Field(default_factory=list, max_length=3)
    user_question: str | None = Field(default=None, max_length=500)


def default_plan(state: dict[str, Any]) -> AgentPlan:
    query = str(state.get("user_input") or "生成 AI 工程师学习提升与项目补强方案")
    intent = str(state.get("task_type") or "project_upgrade")
    scenario_key = str(state.get("scenario_key") or "").strip()
    if not scenario_key:
        scenario_key, _scenario = scenario_for_task_type(intent)
    if scenario_key:
        scenario_plan = build_scenario_plan(scenario_key, query)
        if scenario_plan:
            return AgentPlan.model_validate(scenario_plan)

    if "interview" in intent.lower():
        rubric_titles = " ".join(str(item["title"]) for item in INTERVIEW_SCORING_RUBRIC)
        return AgentPlan(
            goal=query,
            intent=intent,
            research_tasks=[
                ResearchTask(
                    task_id="interview-rubric-evidence",
                    query=f"{query} 面试评分 简历 JD 项目证据 {rubric_titles}",
                    objective=f"按评分 Rubric 检索可验证证据：{rubric_titles}。",
                    top_k=8,
                ),
                ResearchTask(
                    task_id="jd-resume-match",
                    query=f"{query} JD 岗位匹配 简历证据 项目表达 可信度 面试适配度",
                    objective="检索 JD 要求、简历项目证据和岗位匹配信号，支撑面试能力诊断。",
                    top_k=8,
                ),
                ResearchTask(
                    task_id="technical-followups",
                    query=f"{query} 技术追问 原理 取舍 边界 风险 结构化回答",
                    objective="检索技术深度、追问风险和结构化表达素材，形成可练习的改进方向。",
                    top_k=8,
                ),
            ],
        )

    return AgentPlan(
        goal=query,
        intent=intent,
        research_tasks=[
            ResearchTask(task_id="evidence", query=query, objective="检索与用户任务直接相关的知识库证据。"),
            ResearchTask(
                task_id="architecture",
                query=f"{query} 架构 状态流转 工具 记忆 验证",
                objective="检索实现架构、风险和可验证的改造依据。",
            ),
        ],
    )




def _child_agents_for_route(route: SupervisorRoute, *, needs_verification: bool) -> list[ChildAgent]:
    if route == "answer":
        return ["answer_agent"]
    if route == "research":
        return ["planner_agent", "research_agent"]
    return ["answer_agent"]


def default_supervisor_decision(state: dict[str, Any], *, reason: str | None = None) -> SupervisorDecision:
    user_input = str(state.get("user_input") or "").strip()
    task_type = str(state.get("task_type") or "").lower()
    text = user_input.lower()
    has_rag_scope = bool(state.get("kb_id") or state.get("document_id") or state.get("conversation_id"))
    heavy_terms = (
        "面试", "提优", "提升", "学习计划", "学习提升", "学习路线", "学习路径", "方案", "规划",
        "计划", "设计", "架构", "重构", "工程化", "评估", "诊断", "薄弱", "练习", "训练",
        "模拟", "复盘", "批改", "评价", "准备", "安排", "制定", "简历", "岗位", "agent", "rag",
    )
    rag_terms = ("知识库", "文档", "资料", "有没有", "查", "检索", "引用", "根据", "材料", "是什么", "为什么", "区别", "原理", "？", "解释", "对比", "举例", "怎么理解")
    direct_terms = ("是什么", "解释", "什么意思", "区别", "怎么理解", "你好", "谢谢", "早上好", "好的", "hello", "hi", "在吗")

    if any(term in task_type for term in ("learning_coach", "diagnostic", "practice", "coach")):
        route = "research"
        intent = "learning_improvement"
        confidence = 0.82
        needs_rag = has_rag_scope
        needs_verification = True
        stop_after_children = False
        response_mode = "research"
        query = user_input
        fallback_reason = "学习提升任务走诊断、练习、教练短链路。"
    elif any(term in task_type for term in ("project", "upgrade", "improvement", "interview")) or any(term in text for term in heavy_terms):
        route: SupervisorRoute = "research"
        intent = str(state.get("task_type") or "interview_improvement")
        confidence = 0.72
        needs_rag = True
        needs_verification = True
        stop_after_children = False
        response_mode: Literal["answer", "research"] = "research"
        query = user_input
        fallback_reason = "请求涉及规划、诊断、练习或长期提升，需要进入多 Agent 方案链路。"
    elif has_rag_scope or any(term in text for term in rag_terms):
        route = "answer"
        intent = "rag_question"
        confidence = 0.68
        needs_rag = True
        needs_verification = False
        stop_after_children = True
        response_mode = "answer"
        query = user_input
        fallback_reason = "请求包含知识库上下文或具体资料问题，交给 Tool Agent 做检索问答。"
    elif len(user_input) <= 160 or any(term in text for term in direct_terms):
        route = "answer"
        intent = "direct_chat"
        confidence = 0.62
        needs_rag = False
        needs_verification = False
        stop_after_children = True
        response_mode = "answer"
        query = None
        fallback_reason = "请求较轻量或偏闲聊，无需调用 RAG 或多 Agent 链路。"
    else:
        route = "research"
        intent = str(state.get("task_type") or "interview_improvement")
        confidence = 0.55
        needs_rag = True
        needs_verification = True
        stop_after_children = False
        response_mode = "research"
        query = user_input
        fallback_reason = "输入较开放，按学习提升或方案设计任务进入多 Agent 链路。"

    needs_tools = route == "answer" and needs_rag
    return SupervisorDecision(
        child_agents=_child_agents_for_route(route, needs_verification=needs_verification),
        route=route,
        intent=intent,
        reason=reason or fallback_reason,
        confidence=confidence,
        needs_rag=needs_rag,
        needs_tools=needs_tools,
        needs_verification=needs_verification,
        stop_after_children=stop_after_children,
        response_mode=response_mode,
        query=query,
    )


def _pick_tool(available_tools: list[dict[str, Any]], *, category: str, preferred: str | None = None) -> str:
    names = {str(item.get("name") or "") for item in available_tools if item.get("name")}
    if preferred and preferred in names:
        return preferred
    for item in available_tools:
        if item.get("category") == category and item.get("name"):
            return str(item["name"])
    return preferred or next(iter(names), "")


def default_answer_agent_decision(
    state: dict[str, Any],
    *,
    available_tools: list[dict[str, Any]] | None = None,
    reason: str | None = None,
) -> AnswerAgentDecision:
    decision = state.get("route_decision") or {}
    query = str(decision.get("query") or state.get("user_input") or "").strip()
    tools = available_tools or json.loads(_describe_available_tools())
    if decision and not bool(decision.get("needs_tools")):
        return AnswerAgentDecision(
            calls=[],
            next_action="complete",
            reason=reason or "The question can be answered directly without retrieval or external tools.",
            confidence=0.7,
        )
    has_proposal = bool(state.get("proposal"))
    source_policy = resolve_source_policy(state)
    if has_proposal:
        return AnswerAgentDecision(
            calls=[
                ToolCallSpec(
                    tool_name=_pick_tool(tools, category="verification", preferred="knowledge.verify_claim"),
                    arguments={"s2": state.get("proposal") or {}, "knowledge": {"citations": state.get("citations") or [], "grounding": state.get("grounding") or {}}},
                    reason="已有方案时优先检查证据支撑。",
                )
            ],
            next_action="complete",
            reason=reason or "使用确定性工具计划校验已有方案。",
            confidence=0.62,
        )
    calls = [
        ToolCallSpec(
            tool_name=_pick_tool(tools, category="rag", preferred="knowledge.answer"),
            arguments={"query": query, "top_k": 5, "rewrite_query": True},
            reason="Retrieve private knowledge-base evidence when it is available.",
        )
    ]
    if source_policy == "auto":
        calls.append(
            ToolCallSpec(
                tool_name=_pick_tool(tools, category="web", preferred="web.search_duckduckgo"),
                arguments={"query": query, "limit": 5},
                reason="Supplement private retrieval with current public web sources.",
            )
        )
    return AnswerAgentDecision(
        calls=calls,
        next_action="complete",
        reason=reason or "Use local and public sources together unless the user explicitly restricts the task to local knowledge.",
        confidence=0.66,
    )


def _render_prompt(name: str, variables: dict[str, Any], *, role: str = "planner", state: dict[str, Any] | None = None) -> str:
    """Render a prompt via the registry, with few-shot example injection."""
    result = prompt_registry.render(name, variables, role=role)

    if state:
        try:
            from app.skills import build_skill_injection_context

            skill_context, skill_reference = build_skill_injection_context(
                str(state.get("user_input") or ""),
                role=role,
            )
            if skill_context:
                state["skill_context"] = skill_reference or {}
                result.text += "\n\n" + skill_context
        except Exception as exc:  # noqa: BLE001
            logger.debug("Skill context injection skipped for %s: %s", name, exc)

    # Inject few-shot examples when available.
    if state and result.variant_id:
        examples = prompt_registry.pick_examples(
            name, str(state.get("user_input", "")), top_k=2,
        )
        if examples:
            example_block = "\n\n## 参考示例\n" + "\n".join(
                f"示例{i+1}:\n输入: {e['input'][:300]}\n输出: {e['output'][:300]}\n"
                for i, e in enumerate(examples)
            )
            result.text += example_block

    return result.text


def _supervisor_agent_prompt(state: dict[str, Any]) -> str:
    return _render_prompt("supervisor_agent", {
        "user_input": state.get("user_input", ""),
        "task_type": state.get("task_type", "interview_improvement"),
        "memory_context": state.get("memory_context", {}),
        "has_rag_scope": bool(state.get("kb_id") or state.get("document_id") or state.get("conversation_id")),
        "existing_artifact_count": len(state.get("artifacts", []) or []),
        "has_proposal": bool(state.get("proposal")),
        "allowed_child_agents": ["answer_agent", "planner_agent", "research_agent"],
        "allowed_routes": ["answer", "research"],
    }, role="planner", state=state)


def _answer_agent_prompt(
    state: dict[str, Any],
    *,
    tool_package: str,
    available_tools: list[dict[str, Any]],
) -> str:
    prompt = _render_prompt("answer_agent", {
        "user_input": state.get("user_input", ""),
        "route_decision": state.get("route_decision", {}),
        "tool_package": tool_package,
        "available_tools": available_tools,
        "artifacts": state.get("artifacts", []),
        "proposal": state.get("proposal", {}),
        "source_policy": resolve_source_policy(state),
    }, role="planner", state=state)
    return (
        f"{prompt}\n\n"
        "This is the Answer Agent. Return only the registered tool calls needed for one response. "
        "For a self-contained answer, return calls=[] and next_action=complete. "
        "next_action must be either complete or fallback. Never delegate to research_agent."
    )


def _research_source_prompt(
    state: dict[str, Any],
    *,
    query: str,
    objective: str,
    available_tools: list[dict[str, Any]],
) -> str:
    return _render_prompt("research_sources", {
        "user_input": state.get("user_input", ""),
        "query": query,
        "objective": objective,
        "source_policy": resolve_source_policy(state),
        "available_tools": available_tools,
    }, role="planner", state=state)


def _planner_prompt(state: dict[str, Any]) -> str:
    return _render_prompt("planner_agent", {
        "user_input": state.get("user_input", ""),
        "task_type": state.get("task_type", "project_upgrade"),
        "memory_context": state.get("memory_context", {}),
        "available_tools": json.loads(_describe_available_tools()),
        "feedback_summary": state.get("feedback_summary", "暂无反馈数据。"),
    }, role="planner", state=state)


def _plan_judge_prompt(state: dict[str, Any], candidates: list[AgentPlan]) -> str:
    return _render_prompt("plan_judge", {
        "user_input": state.get("user_input", ""),
        "candidates": [item.model_dump(mode="json") for item in candidates],
    }, role="judge", state=state)


def _architecture_prompt(state: dict[str, Any]) -> str:
    artifacts = [item for item in state.get("artifacts", []) if item.get("kind") in {"research", "memory"}]
    return _render_prompt("architect_proposal", {
        "user_input": state.get("user_input", ""),
        "artifacts": artifacts,
    }, role="architect", state=state)


def _verifier_prompt(state: dict[str, Any]) -> str:
    return _render_prompt("verifier_decision", {
        "proposal": state.get("proposal", {}),
        "artifacts": state.get("artifacts", []),
    }, role="judge", state=state)


def _safe_json_dumps(obj: Any) -> str:
    import json
    return json.dumps(obj, ensure_ascii=False, default=str)


def _describe_available_tools() -> str:
    """Return a JSON description of available tools for injection into LLM prompts."""
    try:
        from deerflow.tools import list_tools as _list_tools
        tools = _list_tools()
        return _safe_json_dumps([
            {"name": t["name"], "category": t["category"], "description": t["description"]}
            for t in tools
        ])
    except Exception:
        return "[]"


# ── Structured-output helpers ─────────────────────────────────────────


def _invoke_structured(
    role: str, prompt: str, schema: type[BaseModel],
    on_token: TokenCallback | None = None,
    *,
    template_name: str = "",
    task_id: str = "",
) -> tuple[BaseModel, str, int, int]:
    """Call the LLM gateway for structured output.

    Records A/B metrics when template_name and task_id are provided.
    Returns (validated_model, "provider:model", prompt_tokens, completion_tokens).
    """
    started = time.perf_counter()
    retry_count = 0
    validation_errors: list[str] = []
    prompt_tk = 0
    completion_tk = 0
    try:
        validated, response = llm_gateway.invoke_structured(
            role=role, prompt=prompt, schema=schema, on_token=on_token,  # type: ignore[arg-type]
        )
        source = f"{response.provider}:{response.model}"
        prompt_tk = response.prompt_tokens
        completion_tk = response.completion_tokens
    except ValidationError as exc:
        retry_count = 1
        validation_errors = [str(e) for e in exc.errors()]
        raise
    except Exception:
        retry_count = 1
        raise
    finally:
        if template_name and task_id:
            latency = round((time.perf_counter() - started) * 1000)
            prompt_registry.record_evaluation(
                ABMetric(
                    template_name=template_name,
                    variant_version=0,
                    task_id=task_id,
                    success=not validation_errors,
                    latency_ms=latency,
                    prompt_tokens=prompt_tk,
                    completion_tokens=completion_tk,
                    retry_count=retry_count,
                    validation_errors=validation_errors or None,
                    verification_score=None,
                )
            )
    return validated, source, prompt_tk, completion_tk


# ── Public orchestration entry points ─────────────────────────────────


def generate_supervisor_decision(
    state: dict[str, Any], *, on_token: TokenCallback | None = None,
) -> tuple[SupervisorDecision, str, str | None]:
    try:
        decision, source, _ptk, _ctk = _invoke_structured(
            "planner", _supervisor_agent_prompt(state), SupervisorDecision, on_token,
            template_name="supervisor_agent", task_id=str(state.get("task_id", "")),
        )
        return decision, source, None  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        error = f"Supervisor Agent LLM call failed; applied deterministic delegation: {exc}"
        logger.warning(error)
        return default_supervisor_decision(state, reason=error), "fallback", error


def generate_answer_agent_decision(
    state: dict[str, Any],
    *,
    tool_package: str,
    available_tools: list[dict[str, Any]],
    on_token: TokenCallback | None = None,
) -> tuple[AnswerAgentDecision, str, str | None]:
    try:
        decision, source, _ptk, _ctk = _invoke_structured(
            "planner",
            _answer_agent_prompt(state, tool_package=tool_package, available_tools=available_tools),
            AnswerAgentDecision,
            on_token,
            template_name="answer_agent",
            task_id=str(state.get("task_id", "")),
        )
        return decision, source, None  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        error = f"Answer Agent LLM call failed; applied deterministic registered-tool decision: {exc}"
        logger.warning(error)
        return default_answer_agent_decision(state, available_tools=available_tools, reason=error), "fallback", error


def default_research_source_decision(state: dict[str, Any]) -> ResearchSourceDecision:
    if resolve_source_policy(state) == "local_only":
        return ResearchSourceDecision(
            tool_names=["knowledge.answer"],
            reason="The user restricted this task to the local knowledge base.",
        )
    return ResearchSourceDecision(
        tool_names=["knowledge.answer", "web.search_duckduckgo"],
        reason="Fallback source plan uses complementary private and public evidence.",
    )


def generate_research_source_decision(
    state: dict[str, Any],
    *,
    query: str,
    objective: str,
    available_tools: list[dict[str, Any]],
    on_token: TokenCallback | None = None,
) -> tuple[ResearchSourceDecision, str, str | None]:
    try:
        decision, source, _ptk, _ctk = _invoke_structured(
            "planner",
            _research_source_prompt(
                state,
                query=query,
                objective=objective,
                available_tools=available_tools,
            ),
            ResearchSourceDecision,
            on_token,
            template_name="research_sources",
            task_id=str(state.get("task_id", "")),
        )
        allowed_names = {str(item.get("name") or "") for item in available_tools}
        selected = [name for name in decision.tool_names if name in allowed_names]
        if resolve_source_policy(state) == "local_only":
            selected = [name for name in selected if name.startswith("knowledge.")]
        if not selected:
            raise ValueError("Research source selection did not include an allowed tool.")
        return ResearchSourceDecision(tool_names=selected, reason=decision.reason), source, None
    except Exception as exc:  # noqa: BLE001
        error = f"Research source selection failed; applied fallback: {exc}"
        logger.warning(error)
        return default_research_source_decision(state), "fallback", error


def generate_plan(
    state: dict[str, Any], *, on_token: TokenCallback | None = None,
) -> tuple[AgentPlan, str, str | None]:
    """Generate N candidate plans via LLM, then use a Judge LLM to select the best.

    Returns (plan, source_description, aggregated_errors_or_None).
    """
    candidates: list[AgentPlan] = []
    sources: list[str] = []
    errors: list[str] = []
    total_prompt_tk = 0
    total_completion_tk = 0

    for _ in range(AGENT_PLANNER_CANDIDATE_COUNT):
        try:
            candidate, source, ptk, ctk = _invoke_structured(
                "planner", _planner_prompt(state), AgentPlan, on_token,
                template_name="planner_agent", task_id=str(state.get("task_id", "")),
            )
            candidates.append(candidate)  # type: ignore[arg-type]
            sources.append(source)
            total_prompt_tk += ptk
            total_completion_tk += ctk
        except (ValidationError, ValueError) as exc:
            errors.append(f"Supervisor plan validation failed: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Supervisor LLM call failed: {exc}")
            break

    if not candidates:
        return default_plan(state), "fallback", "; ".join(errors) or "No LLM provider is configured."

    if len(candidates) == 1:
        return candidates[0], sources[0], "; ".join(errors) or None

    try:
        selection, judge_source, jptk, jctk = _invoke_structured(
            "judge", _plan_judge_prompt(state, candidates), PlanSelection, on_token,
            template_name="plan_judge", task_id=str(state.get("task_id", "")),
        )
        total_prompt_tk += jptk
        total_completion_tk += jctk
        selected = candidates[selection.selected_index]  # type: ignore[index]
        return (
            selected,
            f"{sources[selection.selected_index]} -> planning_judge:{judge_source}",
            "; ".join(errors) or None,
        )  # type: ignore[index]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Planning Judge failed; using first candidate. %s", exc)
        return candidates[0], f"{sources[0]} -> planning_judge:fallback", f"Planning Judge failed: {exc}"


def generate_proposal(
    state: dict[str, Any], *, on_token: TokenCallback | None = None,
) -> tuple[UpgradeProposal, str | None, str]:
    """Generate an upgrade proposal from research artifacts via the proposal LLM tool."""
    try:
        proposal, source, _ptk, _ctk = _invoke_structured(
            "architect", _architecture_prompt(state), UpgradeProposal, on_token,
            template_name="architect_proposal", task_id=str(state.get("task_id", "")),
        )
        return proposal, None, source  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        error = f"Proposal LLM tool call failed: {exc}"
        logger.warning(error)

    evidence = [
        item for item in state.get("artifacts", [])
        if item.get("kind") == "research" and not item.get("error")
    ]
    references = [item.get("artifact_id", "unknown") for item in evidence]
    return (
        UpgradeProposal(
            title="证据待补全的改造草案",
            summary="架构方案生成模型不可用；以下仅汇总已取得的证据，不应视为最终建议。",
            sections=[
                ProposalSection(
                    title="已取得证据",
                    items=[
                        str(item.get("data", {}).get("answer", "未返回可用答案"))[:600]
                        for item in evidence
                    ] or ["暂无可用知识库证据。"],
                    evidence_artifact_ids=references,
                )
            ],
            open_questions=["恢复 LLM 后重新生成并校验最终方案。"],
        ),
        error,
        "fallback",
    )


def verify_proposal(
    state: dict[str, Any], *, on_token: TokenCallback | None = None,
) -> tuple[VerificationDecision, str | None, str]:
    """Verify proposal evidence coverage — deterministic gate first, then LLM judge."""

    # Deterministic evidence gate evaluated before any LLM call.
    evidence = [
        item for item in state.get("artifacts", [])
        if item.get("kind") == "research" and item.get("citations")
    ]
    if not evidence:
        return (
            VerificationDecision(
                status="repair", score=0.2,
                issues=[{"type": "citation_missing", "retryable": True}],
                repair_queries=[str(state.get("user_input") or "")],
            ),
            None,
            "deterministic_evidence_gate",
        )

    try:
        decision, source, _ptk, _ctk = _invoke_structured(
            "judge", _verifier_prompt(state), VerificationDecision, on_token,
            template_name="verifier_decision", task_id=str(state.get("task_id", "")),
        )
        return decision, None, source  # type: ignore[return-value]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Verification LLM tool failed; applying evidence-coverage gate. %s", exc)
        return (
            VerificationDecision(
                status="passed", score=0.65,
                issues=[{"type": "judge_unavailable", "message": str(exc), "retryable": True}],
            ),
            f"Verification LLM tool failed; applied evidence coverage gate: {exc}",
            "deterministic_evidence_gate",
        )
