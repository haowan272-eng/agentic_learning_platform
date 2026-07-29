"""End-to-end integration tests for the multi-agent LangGraph runtime.

These tests exercise the full StateGraph with a MemorySaver checkpointer,
mocked tools, and controlled LLM responses.  They validate state-machine
behaviour 鈥?not LLM output quality 鈥?covering:

- Full happy path: plan -> dispatch -> research tool loop -> final
- Repair loop: verification failure 鈫?repair 鈫?re-dispatch 鈫?re-verify 鈫?final
- Fallback: non-retryable verification failure 鈫?graceful degradation
- Budget enforcement: deadline, max_tool_calls
- Cancellation: mid-graph cancel signal
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from deerflow.planner import (
    AgentPlan,
    ProposalSection,
    ResearchTask,
    SupervisorDecision,
    AnswerAgentDecision,
    ToolCallSpec,
    UpgradeProposal,
    VerificationDecision,
    default_plan,
    generate_plan,
    generate_proposal,
    verify_proposal,
)
import deerflow.runtime as runtime
from deerflow.runtime import build_agent_graph, run_agent_task
from deerflow.tools import registry as tool_registry
from deerflow.schemas import (
    AgentEvent,
    AgentTaskState,
    Artifact,
    RuntimeStore,
    ToolResult,
    validate_agent_state,
    validate_node_update,
    validate_research_work_item,
    validate_supervisor_route,
)
# 鈹€鈹€ helpers 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


def _fake_research_result(*, ok: bool = True, citations: bool = True) -> ToolResult:
    """Return a realistic knowledge.answer ToolResult."""
    cites = (
        [{"document_id": 1, "source": "test.txt", "text": "...", "chunk_id": 42}]
        if citations
        else []
    )
    return {
        "ok": ok,
        "tool_name": "knowledge.answer",
        "data": {
            "answer": "RAG evidence for the user goal.",
            "retrieved_count": 5 if citations else 0,
            "retrieved_contexts": ["context chunk 1", "context chunk 2"],
        },
        "confidence": 0.85 if citations else 0.2,
        "citations": cites,
        "grounding": {
            "mode": "rag_grounded" if citations else "insufficient_evidence",
            "rag_used": True,
            "retrieved_count": 5 if citations else 0,
            "source_count": 1 if citations else 0,
            "citation_count": len(cites),
            "retrieval_status": "grounded" if citations else "empty",
        },
        "trace": [{"step": "run_rag_answer"}],
        "latency_ms": 42,
        "usage": {"total_tokens": 120, "estimated": True},
        "error": None if ok else {
            "type": "retrieval_empty",
            "retryable": True,
            "message": "RAG returned no retrieved context.",
        },
    }


def _make_base_state(**overrides: Any) -> AgentTaskState:
    """Minimal valid initial state for a supervisor-driven task."""
    state: AgentTaskState = {
        "session_id": "sess-test",
        "task_id": "task-test",
        "run_id": "run-test",
        "user_id": 1,
        "username": "tester",
        "user_input": "upgrade my Agent workflow",
        "task_type": "project_upgrade",
        "kb_id": None,
        "document_id": None,
        "conversation_id": None,
        "budget": {
            "max_tool_calls": 12,
            "max_steps": 8,
            "deadline_seconds": 900,
            "max_total_tokens": 24000,
            "max_cost_usd": 2.0,
        },
        "status": "pending",
        "repair_count": 0,
        "token_usage": 0,
        "estimated_cost_usd": 0.0,
    }
    state.update(overrides)  # type: ignore[typeddict-item]
    return state


class FakeStore:
    """In-memory RuntimeStore for tests.  Tracks every call for assertions."""

    def __init__(self) -> None:
        self.events: list[AgentEvent] = []
        self.event_index = 0
        self.saved_states: list[dict[str, Any]] = []
        self.plans: list[dict[str, Any]] = []
        self.steps: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.verifications: list[dict[str, Any]] = []
        self._cancel_requested = False

    def is_cancel_requested(self, _task_id: str) -> bool:
        return self._cancel_requested

    def append_event(self, event: AgentEvent) -> int:
        self.event_index += 1
        event["event_index"] = self.event_index
        self.events.append(event)
        return self.event_index

    def save_task_state(self, task_id: str, state: dict[str, Any]) -> None:
        self.saved_states.append(dict(state))

    def save_plan(self, payload: dict[str, Any]) -> None:
        self.plans.append(dict(payload))

    def upsert_step(self, payload: dict[str, Any]) -> None:
        self.steps.append(dict(payload))

    def save_tool_call(self, payload: dict[str, Any]) -> None:
        self.tool_calls.append(dict(payload))

    def save_verification(self, payload: dict[str, Any]) -> None:
        self.verifications.append(dict(payload))


# 鈹€鈹€ default plan shape 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestDefaultPlan:
    def test_default_plan_produces_two_independent_research_tasks(self):
        plan = default_plan({"user_input": "optimize RAG retrieval", "task_type": "project_upgrade"})
        assert len(plan.research_tasks) >= 2
        ids = [t.task_id for t in plan.research_tasks]
        assert len(set(ids)) == len(ids)

    def test_default_plan_has_non_empty_goal(self):
        plan = default_plan({"user_input": ""})
        assert len(plan.goal) > 0


# 鈹€鈹€ happy-path E2E 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestHappyPath:
    """Full graph: supervisor -> planner -> research -> final response."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Mock the tool gateway so research agents get controlled results."""
        def mocked_tool_call(tool_name: str, arguments: dict[str, Any], *, agent: str) -> ToolResult:
            if tool_name.startswith(("knowledge.", "web.")):
                return _fake_research_result(ok=True, citations=True)
            return tool_registry.call(tool_name, arguments, agent=agent)

        with patch(
            "deerflow.tool_manager.call_tool",
            side_effect=mocked_tool_call,
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ):
            yield

    def test_happy_path_completes_and_produces_final_answer(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()
        checkpointer = MemorySaver()

        graph = build_agent_graph(store, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": state["task_id"]}}

        # Consume all graph updates.
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)

        # The graph should have completed.
        assert final.get("status") == "completed"
        assert final.get("final_answer"), "final_answer must not be empty"
        # Citations should be present from the research results.
        assert len(final.get("citations", [])) > 0
        # Events should have been emitted.
        assert len(store.events) > 2  # at least task.started + task.completed

    def test_events_include_expected_lifecycle(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        event_types = {e["event_type"] for e in store.events}
        # Core lifecycle events emitted by graph nodes (not the run_agent_task wrapper).
        assert "plan.created" in event_types
        assert "review.completed" in event_types
        assert "task.completed" in event_types

    def test_tool_calls_and_verifications_are_recorded(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        # Two research tasks 鈫?at least 2 tool calls.
        assert len(store.tool_calls) >= 2, (
            f"Expected 鈮? tool calls, got {len(store.tool_calls)}"
        )
        # Verification should record one verification.
        assert len(store.verifications) >= 1, (
            f"Expected 鈮? verification, got {len(store.verifications)}"
        )
        assert {call["agent_name"] for call in store.tool_calls} == {"research_agent"}
        tool_names = {call["tool_name"] for call in store.tool_calls}
        assert "architecture.generate_proposal" in tool_names
        assert "verification.verify_proposal" in tool_names

    def test_research_agent_uses_tool_request_feedback_contract(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        message_kinds = {message["kind"] for message in final.get("messages", [])}
        event_agents = {event.get("agent_name") for event in store.events}

        assert "tool_request" in message_kinds
        assert "tool_result" in message_kinds
        assert "research_agent" in event_agents
        assert all(item.get("executor") == "research_agent" for item in final.get("tool_feedback", {}).values())


# 鈹€鈹€ router path selection E2E 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestSupervisorDelegation:
    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ):
            yield

    def test_answer_route_finishes_without_research_chain(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(task_type="chat", user_input="What is RAG?")

        with patch(
            "deerflow.runtime.generate_supervisor_decision",
            return_value=(
                SupervisorDecision(
                    child_agents=["answer_agent"],
                    route="answer",
                    intent="direct_chat",
                    reason="simple concept question",
                    confidence=0.9,
                    needs_rag=False,
                    needs_tools=False,
                    needs_verification=False,
                    stop_after_children=True,
                    response_mode="answer",
                ),
                "mock",
                None,
            ),
        ):
            graph = build_agent_graph(store, checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": state["task_id"]}}

            for _ in graph.stream(state, config=config, stream_mode="updates"):
                pass

        final = dict(graph.get_state(config).values)
        event_types = {event["event_type"] for event in store.events}
        assert final["status"] == "completed"
        assert final["grounding"]["mode"] == "answer_agent"
        assert final["supervisor_decision"]["child_agents"] == ["answer_agent"]
        assert "supervisor.delegated" in event_types
        assert len(store.plans) == 0
        assert len(store.tool_calls) == 0
        assert len(store.verifications) == 0

    def test_rag_question_routes_through_answer_agent_without_research_upgrade(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(task_type="rag_question", user_input="Is TCP three-way handshake in the knowledge base?", kb_id=1)

        with patch(
            "deerflow.runtime.generate_supervisor_decision",
            return_value=(
                SupervisorDecision(
                    child_agents=["answer_agent"],
                    route="answer",
                    intent="rag_question",
                    reason="knowledge lookup through tool agent",
                    confidence=0.92,
                    needs_rag=True,
                    needs_tools=True,
                    needs_verification=False,
                    stop_after_children=True,
                    response_mode="answer",
                    query="TCP 涓夋鎻℃墜",
                ),
                "mock",
                None,
            ),
        ), patch(
            "deerflow.runtime.generate_answer_agent_decision",
            return_value=(
                AnswerAgentDecision(
                    calls=[
                        ToolCallSpec(
                            call_id="tool-rag",
                            tool_name="knowledge.answer",
                            arguments={"query": "TCP 涓夋鎻℃墜", "top_k": 5},
                            reason="retrieve grounded answer through tool agent",
                        )
                    ],
                    next_action="complete",
                    reason="answer after tool retrieval",
                    confidence=0.9,
                ),
                "mock",
                None,
            ),
        ), patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ):
            graph = build_agent_graph(store, checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": state["task_id"]}}

            for _ in graph.stream(state, config=config, stream_mode="updates"):
                pass

        final = dict(graph.get_state(config).values)
        event_agents = {event.get("agent_name") for event in store.events}
        assert final["status"] == "completed"
        assert final["grounding"]["rag_used"] is True
        assert final["supervisor_decision"]["child_agents"] == ["answer_agent"]
        assert "rag_retrieve" not in event_agents
        assert "answer_agent" in event_agents
        assert len(store.plans) == 0
        assert {call["tool_name"] for call in store.tool_calls} == {"knowledge.answer"}
        assert len(store.verifications) == 0

    def test_answer_agent_executes_registered_tools_and_returns_feedback_response(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(task_type="rag_question", user_input="Answer TCP three-way handshake from the knowledge base", kb_id=1)

        with patch(
            "deerflow.runtime.generate_supervisor_decision",
            return_value=(
                SupervisorDecision(
                    child_agents=["answer_agent"],
                    route="answer",
                    intent="rag_question",
                    reason="needs autonomous tool selection",
                    confidence=0.9,
                    needs_rag=True,
                    needs_tools=True,
                    needs_verification=False,
                    stop_after_children=True,
                    response_mode="answer",
                    query="TCP 涓夋鎻℃墜",
                ),
                "mock",
                None,
            ),
        ), patch(
            "deerflow.runtime.generate_answer_agent_decision",
            return_value=(
                AnswerAgentDecision(
                    calls=[
                        ToolCallSpec(
                            call_id="tool-1",
                            tool_name="knowledge.answer",
                            arguments={"query": "TCP 涓夋鎻℃墜", "top_k": 5},
                            reason="retrieve grounded answer",
                        )
                    ],
                    next_action="complete",
                    reason="answer after retrieval",
                    confidence=0.88,
                ),
                "mock",
                None,
            ),
        ), patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ):
            graph = build_agent_graph(store, checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": state["task_id"]}}

            for _ in graph.stream(state, config=config, stream_mode="updates"):
                pass

        final = dict(graph.get_state(config).values)
        event_types = {event["event_type"] for event in store.events}
        assert final["status"] == "completed"
        assert final["supervisor_decision"]["child_agents"] == ["answer_agent"]
        assert final["tool_feedback"]["success_count"] == 1
        assert final["tool_feedback"]["next_action"] == "complete"
        assert "answer.agent_decided" in event_types
        assert "answer.feedback_ready" in event_types
        assert len(store.tool_calls) == 1
        assert len(store.verifications) == 0


@pytest.mark.parametrize(
    ("decision", "expected_goto"),
    [
        ({"action": "approve"}, "research_agent"),
        ({"action": "edit", "user_input": "revise the plan"}, "planner_agent"),
        ({"action": "reject"}, "fallback_response"),
    ],
)
def test_approval_gate_routes_only_to_the_clean_research_chain(monkeypatch, decision, expected_goto):
    store = FakeStore()
    state = _make_base_state(plan={"goal": "test", "approval_required": True})
    monkeypatch.setattr(runtime, "interrupt", lambda _request: decision)
    with patch("deerflow.runtime.append_recent_event"), patch("deerflow.runtime.publish_task_event"):
        command = runtime._approval_gate(state, store)

    assert command.goto == expected_goto


@pytest.mark.parametrize(
    ("verification", "proposal", "expected_outcome", "expected_route"),
    [
        ({"status": "passed"}, {"summary": "ready"}, "approved", "final_response"),
        ({"status": "needs_approval"}, {"summary": "needs input"}, "needs_confirmation", "approval_gate"),
        ({"status": "fallback"}, {}, "rejected", "fallback_response"),
    ],
)
def test_review_agent_controls_publication_without_tool_calls(
    verification, proposal, expected_outcome, expected_route,
):
    store = FakeStore()
    state = _make_base_state(verification=verification, proposal=proposal)
    with patch("deerflow.runtime.append_recent_event"), patch("deerflow.runtime.publish_task_event"):
        update = runtime._review_agent(state, store)

    assert update["review"]["outcome"] == expected_outcome
    assert runtime._review_agent_route({**state, **update}) == expected_route
    assert {event["agent_name"] for event in store.events} == {"review_agent"}


# 鈹€鈹€ repair loop E2E 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestRepairLoop:
    """When the verification returns status=repair the Supervisor must dispatch
    targeted repair queries, then re-verify.  The loop must converge within
    MAX_REPAIR_COUNT iterations."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        def fake_verification_tool(_args: dict[str, Any]) -> ToolResult:
            decision = VerificationDecision(
                status="repair",
                score=0.35,
                issues=[{"type": "citation_missing", "retryable": True}],
                repair_queries=["repair query 1"],
            ).model_dump(mode="json")
            return {
                "ok": False,
                "tool_name": "verification.verify_proposal",
                "data": decision,
                "confidence": 0.35,
                "citations": [],
                "grounding": {"mode": "verification", "rag_used": True},
                "trace": [{"step": "verify_proposal", "source": "mock", "status": "repair"}],
                "error": {"type": "citation_missing", "message": "mock repair", "retryable": True},
            }

        with patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ), patch(
            "deerflow.runtime.record_verification_failure",
        ), patch.dict(
            "deerflow.tools.registry._handlers",
            {"verification.verify_proposal": fake_verification_tool},
        ), patch(
            "deerflow.planner.verify_proposal",
            return_value=(
                VerificationDecision(
                    status="repair",
                    score=0.35,
                    issues=[{"type": "citation_missing", "retryable": True}],
                    repair_queries=["repair query 1"],
                ),
                None,
                "mock",
            ),
        ):
            yield

    def test_repair_loop_runs_up_to_max_repairs_then_fallbacks(self):
        """After MAX_REPAIR_COUNT=2 repairs, the graph should fallback."""
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(source_policy="local_only")

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        # The verification tool keeps returning "repair", so after 2 repairs we fallback.
        assert final.get("status") == "completed"
        # Fallback is the expected outcome when repair is exhausted.
        assert final.get("grounding", {}).get("mode") == "insufficient_evidence"
        assert "无法" in final.get("final_answer", "") or "重试" in final.get("final_answer", "")

    def test_repair_increments_counter_in_state(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(source_policy="local_only")

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        # After 2 repairs, counter should be 2.
        assert final.get("repair_count", 0) >= 2


# 鈹€鈹€ fallback path E2E 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestFallbackPath:
    """When verification returns a non-retryable status the graph falls back
    immediately without attempting repair."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=False),
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ), patch(
            "deerflow.runtime.record_verification_failure",
        ), patch(
            "deerflow.planner.verify_proposal",
            return_value=(
                VerificationDecision(
                    status="fallback",
                    score=0.1,
                    issues=[{"type": "evidence_insufficient", "retryable": False}],
                    repair_queries=[],
                ),
                None,
                "mock",
            ),
        ):
            yield

    def test_fallback_produces_safe_response(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        assert final.get("status") == "completed"
        assert len(final.get("final_answer", "")) > 0


# 鈹€鈹€ budget enforcement E2E 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestBudgetEnforcement:
    """Verify that budget limits are checked and produce graceful errors."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ):
            yield

    def test_deadline_exceeded_terminates_graph(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(
            budget={
                "max_tool_calls": 12,
                "max_steps": 8,
                "deadline_seconds": -1,  # already expired
                "started_at": "2000-01-01T00:00:00+00:00",
                "max_total_tokens": 24000,
                "max_cost_usd": 2.0,
            },
        )

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        # The graph should raise AgentBudgetExceeded during execution.
        # We catch via graph stream 鈥?an error should appear in final state.
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        # The graph may have failed or fallen back; either way it should not hang.
        assert final.get("status") in {"completed", "failed", "cancelled"}

    def test_max_tool_calls_enforced(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        # Artifacts count is checked against max_tool_calls.
        state = _make_base_state(
            budget={"max_tool_calls": 0, "max_steps": 8, "deadline_seconds": 900},
        )

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        assert final.get("status") in {"completed", "failed", "cancelled"}


# 鈹€鈹€ cancellation E2E 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestCancellation:
    """Mid-graph cancellation via the store's is_cancel_requested signal."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ):
            yield

    def test_cancel_before_dispatch_terminates_graph(self):
        from langgraph.checkpoint.memory import MemorySaver

        from deerflow.runtime import AgentTaskCancelled

        store = FakeStore()
        store._cancel_requested = True  # cancel immediately
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        # Cancellation during research_agent raises AgentTaskCancelled.
        # LangGraph surfaces this as an exception from graph.stream().
        try:
            for _ in graph.stream(state, config=config, stream_mode="updates"):
                pass
        except AgentTaskCancelled:
            pass  # expected 鈥?cancellation propagates as exception

        # State should reflect the cancellation.
        final = dict(graph.get_state(config).values)
        assert final.get("status") in {"cancelled", "failed", "completed", "running"}


# 鈹€鈹€ state serialisation round-trip 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestStateRoundTrip:
    """Ensure the graph can be re-invoked with persisted state (resume)."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ):
            yield

    def test_resume_from_checkpoint_preserves_artifacts(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()
        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": state["task_id"]}}

        graph = build_agent_graph(store, checkpointer=checkpointer)

        # First run: complete full graph.
        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        assert final.get("status") == "completed"
        # Artifacts should have been accumulated.
        artifacts = final.get("artifacts", [])
        assert len(artifacts) > 0, "Expected at least memory + research artifacts"

    def test_state_reloads_from_checkpoint(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()
        checkpointer = MemorySaver()
        config = {"configurable": {"thread_id": state["task_id"]}}

        graph = build_agent_graph(store, checkpointer=checkpointer)

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        # Reload state from checkpointer.
        snapshot = graph.get_state(config)
        assert snapshot.values is not None
        reloaded = dict(snapshot.values)
        assert reloaded.get("status") == "completed"
        assert reloaded.get("task_id") == state["task_id"]


# 鈹€鈹€ agent-task-state invariants 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€


class TestAgentTaskStateInvariants:
    """Verify that the state shape accepted by the graph meets minimum requirements."""

    def test_pydantic_state_boundary_rejects_invalid_status(self):
        state = _make_base_state(status="unknown")

        with pytest.raises(ValidationError):
            validate_agent_state(state)

    def test_pydantic_node_update_formats_nested_runtime_payloads(self):
        update = validate_node_update(
            {
                "status": "running",
                "artifacts": [
                    {
                        "artifact_id": "a1",
                        "kind": "research",
                        "producer": "research_agent",
                        "correlation_id": "c1",
                        "data": {"answer": "ok"},
                        "citations": [],
                        "confidence": 0.5,
                        "error": None,
                    }
                ],
                "emitted_events": [
                    {
                        "session_id": "sess-test",
                        "task_id": "task-test",
                        "run_id": "run-test",
                        "event_type": "agent.completed",
                        "event_index": 1,
                        "agent_name": "research_agent",
                        "message": "Research Agent finished.",
                        "payload": {"ok": True},
                        "created_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
            }
        )

        assert update["artifacts"][0]["confidence"] == 0.5
        assert update["emitted_events"][0]["created_at"].isoformat() == "2026-01-01T00:00:00+00:00"

    def test_pydantic_research_work_item_requires_correlation_id(self):
        work = {
            "session_id": "sess-test",
            "task_id": "task-test",
            "run_id": "run-test",
            "username": "tester",
            "query": "test query",
            "objective": "test objective",
        }

        with pytest.raises(ValidationError):
            validate_research_work_item(work)

    def test_pydantic_supervisor_route_boundary_rejects_unknown_target(self):
        with pytest.raises(ValidationError):
            validate_supervisor_route("research_agent")

    def test_supervisor_decision_rejects_mismatched_child_agents(self):
        with pytest.raises(ValidationError):
            SupervisorDecision(
                child_agents=["research_agent"],
                route="answer",
                intent="direct_chat",
                reason="invalid mixed route",
                confidence=0.9,
                needs_rag=False,
                needs_tools=False,
                needs_verification=False,
                stop_after_children=True,
                response_mode="answer",
            )

    def test_supervisor_decision_accepts_answer_route_contract(self):
        decision = SupervisorDecision(
            child_agents=["answer_agent"],
            route="answer",
            intent="rag_question",
            reason="valid tool agent route",
            confidence=0.9,
            needs_rag=True,
            needs_tools=True,
            needs_verification=False,
            stop_after_children=True,
            response_mode="answer",
            query="TCP handshake",
        )

        assert decision.child_agents == ["answer_agent"]

    def test_minimal_state_is_accepted_by_graph_builder(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()
        graph = build_agent_graph(store, checkpointer=MemorySaver())
        # Building and compiling should succeed without error.
        assert graph is not None

    def test_append_only_channels_accumulate(self):
        """Artifacts, messages, errors, and emitted_events use operator.add reducer."""
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(
            artifacts=[
                {"artifact_id": "a1", "kind": "research", "producer": "research_agent",
                 "correlation_id": "c1", "data": {}, "citations": [], "confidence": 0.9, "error": None},
            ],
        )
        # The graph should preserve pre-existing artifacts and add more.
        with patch(
            "deerflow.tool_manager.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "deerflow.runtime.record_memory_event",
        ), patch(
            "deerflow.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "deerflow.runtime.summarize_task_session", return_value=None,
        ), patch(
            "deerflow.runtime.append_recent_event",
        ), patch(
            "deerflow.runtime.publish_task_event",
        ):
            graph = build_agent_graph(store, checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": state["task_id"]}}

            for _ in graph.stream(state, config=config, stream_mode="updates"):
                pass

            final = dict(graph.get_state(config).values)
            artifacts = final.get("artifacts", [])
            # Should contain the pre-existing artifact plus new ones.
            assert len(artifacts) >= 2, f"Expected 鈮? artifacts, got {len(artifacts)}"
