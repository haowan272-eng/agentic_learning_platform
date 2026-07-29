from __future__ import annotations

from deerflow.planner import default_plan
from app.memory.events import sanitize_memory_content


def test_default_plan_creates_independent_parallel_research_tasks():
    plan = default_plan({"user_input": "升级我的 Agent 工作流", "task_type": "project_upgrade"})

    assert len(plan.research_tasks) >= 2
    assert len({task.task_id for task in plan.research_tasks}) == len(plan.research_tasks)


def test_memory_sanitizer_redacts_common_sensitive_values():
    value = sanitize_memory_content("email=a@b.com token=super-secret password: pass123 13800138000")
    assert "a@b.com" not in value
    assert "13800138000" not in value
    assert "super-secret" not in value
