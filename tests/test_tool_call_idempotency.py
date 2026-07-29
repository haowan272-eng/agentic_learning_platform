from __future__ import annotations

from typing import Any

from deerflow import tool_manager


class ReplayStore:
    def __init__(self) -> None:
        self.completed: dict[str, dict[str, Any]] = {}
        self.saved_calls: list[dict[str, Any]] = []

    def get_completed_tool_call(self, idempotency_key: str) -> dict[str, Any] | None:
        return self.completed.get(idempotency_key)

    def save_tool_call(self, payload: dict[str, Any]) -> None:
        self.saved_calls.append(dict(payload))
        self.completed[payload["idempotency_key"]] = {"result": dict(payload["result"])}


def test_managed_tool_replays_a_completed_call_without_reexecuting(monkeypatch) -> None:
    store = ReplayStore()
    invocations: list[dict[str, Any]] = []

    def fake_call_tool(_tool_name: str, arguments: dict[str, Any], *, agent: str) -> dict[str, Any]:
        invocations.append({"arguments": arguments, "agent": agent})
        return {
            "ok": True,
            "data": {"answer": "cached"},
            "citations": [],
            "confidence": 0.9,
            "grounding": {},
            "latency_ms": 3,
        }

    monkeypatch.setattr(tool_manager, "call_tool", fake_call_tool)
    state = {"task_id": "task-1", "run_id": "run-1", "username": "alice"}

    first = tool_manager.execute_managed_tool(
        state,
        store,
        tool_name="knowledge.answer",
        arguments={"query": "idempotency"},
        agent_name="tool_agent",
        skill_name="test",
        call_id="call-1",
    )
    second = tool_manager.execute_managed_tool(
        state,
        store,
        tool_name="knowledge.answer",
        arguments={"query": "idempotency"},
        agent_name="tool_agent",
        skill_name="test",
        call_id="call-1",
    )

    assert first.result == second.result
    assert len(invocations) == 1
    assert invocations[0]["arguments"]["idempotency_key"] == "task-1:call-1"
    assert store.saved_calls[0]["idempotency_key"] == "task-1:call-1"


def test_trusted_arguments_override_agent_supplied_runtime_context() -> None:
    trusted = tool_manager._trusted_args(
        {
            "task_id": "trusted-task",
            "run_id": "trusted-run",
            "username": "alice",
            "user_id": 7,
            "kb_id": 8,
            "proposal": {"source": "state"},
            "artifacts": [{"id": "state"}],
        },
        "verification.verify_proposal",
        {
            "task_id": "spoofed-task",
            "user_id": 999,
            "proposal": {"source": "agent"},
            "artifacts": [{"id": "agent"}],
        },
    )

    assert trusted["task_id"] == "trusted-task"
    assert trusted["user_id"] == 7
    assert trusted["proposal"] == {"source": "state"}
    assert trusted["artifacts"] == [{"id": "state"}]
