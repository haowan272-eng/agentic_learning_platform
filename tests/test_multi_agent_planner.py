from __future__ import annotations

from app.agent_runtime.planner import default_plan
from app.agent_runtime.tools import memory_read_context


def test_fallback_supervisor_plan_contains_parallel_research_tasks():
    plan = default_plan({"user_input": "评估并升级我的 Agent 工作流", "task_type": "project_upgrade"})

    assert len(plan.research_tasks) >= 2
    assert len({item.task_id for item in plan.research_tasks}) == len(plan.research_tasks)
    assert all(item.query for item in plan.research_tasks)


def test_memory_context_tool_ignores_untrusted_nested_state(monkeypatch):
    captured: dict = {}

    def fake_context(state):
        captured.update(state)
        return {"available": True}

    monkeypatch.setattr("app.agent_runtime.tools.build_context_for_state", fake_context)
    result = memory_read_context(
        {
            "user_id": 7,
            "session_id": "trusted-session",
            "task_id": "trusted-task",
            "state": {"user_id": 999, "session_id": "attacker-session", "task_id": "attacker-task"},
        }
    )

    assert result["ok"] is True
    assert captured["user_id"] == 7
    assert captured["session_id"] == "trusted-session"
    assert captured["task_id"] == "trusted-task"
