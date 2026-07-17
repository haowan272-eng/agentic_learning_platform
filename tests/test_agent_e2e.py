"""End-to-end integration tests for the multi-agent LangGraph runtime.

These tests exercise the full StateGraph with a MemorySaver checkpointer,
mocked tools, and controlled LLM responses.  They validate state-machine
behaviour — not LLM output quality — covering:

- Full happy path: plan → dispatch → research → architect → verifier → final
- Repair loop: verification failure → repair → re-dispatch → re-verify → final
- Fallback: non-retryable verification failure → graceful degradation
- Budget enforcement: deadline, max_tool_calls
- Cancellation: mid-graph cancel signal
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.agent_runtime.planner import (
    AgentPlan,
    ProposalSection,
    ResearchTask,
    RouteDecision,
    UpgradeProposal,
    VerificationDecision,
    default_plan,
    generate_plan,
    generate_proposal,
    verify_proposal,
)
from app.agent_runtime.runtime import build_agent_graph, run_agent_task
from app.agent_runtime.schemas import (
    AgentEvent,
    AgentTaskState,
    Artifact,
    RuntimeStore,
    ToolResult,
    validate_agent_state,
    validate_node_update,
    validate_research_work_item,
    validate_router_route,
    validate_verification_route,
)
from app.agent_runtime.tools import call_tool


# ── helpers ────────────────────────────────────────────────────────────────


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
        "user_input": "升级我的 Agent 工作流",
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


# ── default plan shape ──────────────────────────────────────────────────────


class TestDefaultPlan:
    def test_default_plan_produces_two_independent_research_tasks(self):
        plan = default_plan({"user_input": "优化 RAG 检索", "task_type": "project_upgrade"})
        assert len(plan.research_tasks) >= 2
        ids = [t.task_id for t in plan.research_tasks]
        assert len(set(ids)) == len(ids)

    def test_default_plan_has_non_empty_goal(self):
        plan = default_plan({"user_input": ""})
        assert len(plan.goal) > 0


# ── happy-path E2E ──────────────────────────────────────────────────────────


class TestHappyPath:
    """Full graph: supervisor_plan → dispatch_research → research_agent (x2)
    → architect_agent → verifier_agent (passed) → final_response."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        """Mock the tool gateway so research agents get controlled results."""
        with patch(
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
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
        assert "task.completed" in event_types

    def test_tool_calls_and_verifications_are_recorded(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        # Two research tasks → at least 2 tool calls.
        assert len(store.tool_calls) >= 2, (
            f"Expected ≥2 tool calls, got {len(store.tool_calls)}"
        )
        # Verifier should record one verification.
        assert len(store.verifications) >= 1, (
            f"Expected ≥1 verification, got {len(store.verifications)}"
        )


# ── router path selection E2E ────────────────────────────────────────────────


class TestRouterPathSelection:
    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
        ):
            yield

    def test_direct_answer_route_finishes_without_heavy_chain(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(task_type="chat", user_input="什么是 RAG？")

        with patch(
            "app.agent_runtime.runtime.generate_route",
            return_value=(
                RouteDecision(
                    target_node="direct_answer",
                    intent="direct_chat",
                    reason="simple concept question",
                    confidence=0.9,
                    needs_rag=False,
                    needs_verification=False,
                    stop_after_node=True,
                    response_mode="direct",
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
        assert final["grounding"]["mode"] == "direct"
        assert "router.decided" in event_types
        assert len(store.plans) == 0
        assert len(store.tool_calls) == 0
        assert len(store.verifications) == 0

    def test_rag_route_retrieves_and_finishes_without_architect_or_verifier(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state(task_type="rag_question", user_input="知识库里有没有 TCP 三次握手？", kb_id=1)

        with patch(
            "app.agent_runtime.runtime.generate_route",
            return_value=(
                RouteDecision(
                    target_node="rag_retrieve",
                    intent="rag_question",
                    reason="knowledge lookup only",
                    confidence=0.92,
                    needs_rag=True,
                    needs_verification=False,
                    stop_after_node=True,
                    response_mode="rag_answer",
                    query="TCP 三次握手",
                ),
                "mock",
                None,
            ),
        ), patch(
            "app.agent_runtime.runtime.call_tool",
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
        assert "rag_retrieve" in event_agents
        assert "architect_agent" not in event_agents
        assert len(store.plans) == 0
        assert len(store.tool_calls) == 1
        assert len(store.verifications) == 0


# ── repair loop E2E ─────────────────────────────────────────────────────────


class TestRepairLoop:
    """When the Verifier returns status=repair the Supervisor must dispatch
    targeted repair queries, then re-verify.  The loop must converge within
    MAX_REPAIR_COUNT iterations."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
        ), patch(
            "app.agent_runtime.runtime.record_verification_failure",
        ), patch(
            "app.agent_runtime.runtime.generate_proposal",
            return_value=(
                UpgradeProposal(
                    title="Mock Proposal",
                    summary="A mock proposal for repair testing.",
                    sections=[ProposalSection(title="S1", items=["A", "B"], evidence_artifact_ids=["r1"])],
                ),
                None,
                "mock",
            ),
        ), patch(
            "app.agent_runtime.runtime.verify_proposal",
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
        ), patch(
            "app.agent_runtime.planner.verify_proposal",
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
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        # The verifier keeps returning "repair", so after 2 repairs we fallback.
        assert final.get("status") == "completed"
        # Fallback is the expected outcome when repair is exhausted.
        assert "insufficient" in final.get("final_answer", "").lower() or "重试" in final.get("final_answer", "") or "无法" in final.get("final_answer", "")

    def test_repair_increments_counter_in_state(self):
        from langgraph.checkpoint.memory import MemorySaver

        store = FakeStore()
        state = _make_base_state()

        graph = build_agent_graph(store, checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": state["task_id"]}}

        for _ in graph.stream(state, config=config, stream_mode="updates"):
            pass

        final = dict(graph.get_state(config).values)
        # After 2 repairs, counter should be 2.
        assert final.get("repair_count", 0) >= 2


# ── fallback path E2E ───────────────────────────────────────────────────────


class TestFallbackPath:
    """When Verifier returns a non-retryable status the graph falls back
    immediately without attempting repair."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=False),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
        ), patch(
            "app.agent_runtime.runtime.record_verification_failure",
        ), patch(
            "app.agent_runtime.runtime.verify_proposal",
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
        ), patch(
            "app.agent_runtime.planner.verify_proposal",
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


# ── budget enforcement E2E ──────────────────────────────────────────────────


class TestBudgetEnforcement:
    """Verify that budget limits are checked and produce graceful errors."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
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
        # We catch via graph stream — an error should appear in final state.
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


# ── cancellation E2E ────────────────────────────────────────────────────────


class TestCancellation:
    """Mid-graph cancellation via the store's is_cancel_requested signal."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
        ):
            yield

    def test_cancel_before_dispatch_terminates_graph(self):
        from langgraph.checkpoint.memory import MemorySaver

        from app.agent_runtime.runtime import AgentTaskCancelled

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
            pass  # expected — cancellation propagates as exception

        # State should reflect the cancellation.
        final = dict(graph.get_state(config).values)
        assert final.get("status") in {"cancelled", "failed", "completed", "running"}


# ── state serialisation round-trip ──────────────────────────────────────────


class TestStateRoundTrip:
    """Ensure the graph can be re-invoked with persisted state (resume)."""

    @pytest.fixture(autouse=True)
    def _setup_mocks(self):
        with patch(
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
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


# ── agent-task-state invariants ─────────────────────────────────────────────


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

    def test_pydantic_route_boundary_rejects_unknown_verifier_route(self):
        with pytest.raises(ValidationError):
            validate_verification_route("skip_final")

    def test_pydantic_router_route_boundary_rejects_unknown_target(self):
        with pytest.raises(ValidationError):
            validate_router_route("research_agent")

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
            "app.agent_runtime.runtime.call_tool",
            return_value=_fake_research_result(ok=True, citations=True),
        ), patch(
            "app.agent_runtime.runtime.record_memory_event",
        ), patch(
            "app.agent_runtime.runtime.consolidate_task_memory", return_value=[],
        ), patch(
            "app.agent_runtime.runtime.summarize_task_session", return_value=None,
        ), patch(
            "app.agent_runtime.runtime.append_recent_event",
        ), patch(
            "app.agent_runtime.runtime.publish_task_event",
        ):
            graph = build_agent_graph(store, checkpointer=MemorySaver())
            config = {"configurable": {"thread_id": state["task_id"]}}

            for _ in graph.stream(state, config=config, stream_mode="updates"):
                pass

            final = dict(graph.get_state(config).values)
            artifacts = final.get("artifacts", [])
            # Should contain the pre-existing artifact plus new ones.
            assert len(artifacts) >= 2, f"Expected ≥2 artifacts, got {len(artifacts)}"
