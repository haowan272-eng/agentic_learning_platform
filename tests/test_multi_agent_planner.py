from __future__ import annotations

from app.services.learning_service import INTERVIEW_SCORING_RUBRIC
from deerflow.planner import default_plan, default_supervisor_decision
from deerflow.tools import call_tool, list_tools


def test_fallback_supervisor_plan_contains_parallel_research_tasks():
    plan = default_plan({"user_input": "评估并升级我的 Agent 工作流", "task_type": "project_upgrade"})

    assert len(plan.research_tasks) >= 2
    assert len({item.task_id for item in plan.research_tasks}) == len(plan.research_tasks)
    assert all(item.query for item in plan.research_tasks)


def test_interview_default_plan_front_loads_scoring_rubric_into_research():
    plan = default_plan({"user_input": "请基于我的简历和 JD 做面试能力诊断", "task_type": "interview_improvement"})
    joined_queries = "\n".join(task.query for task in plan.research_tasks)
    joined_objectives = "\n".join(task.objective for task in plan.research_tasks)

    assert len(plan.research_tasks) == 3
    for rubric in INTERVIEW_SCORING_RUBRIC:
        assert rubric["title"] in joined_queries or rubric["title"] in joined_objectives


def test_business_scenario_default_plan_uses_scenario_blueprint():
    plan = default_plan(
        {
            "user_input": "请针对 Java 后端 JD 做面试准备",
            "task_type": "scenario_jd_interview_prep",
            "scenario_key": "jd_interview_prep",
        }
    )

    assert plan.intent == "scenario_jd_interview_prep"
    assert [item.task_id for item in plan.research_tasks] == [
        "jd-requirements",
        "resume-match",
        "interview-risks",
    ]
    assert all("Java 后端 JD" in item.query for item in plan.research_tasks)


def test_business_scenario_routes_to_verified_agent_workflow():
    decision = default_supervisor_decision(
        {
            "user_input": "把我的 RAG 项目改造成 STAR 表达",
            "task_type": "scenario_resume_star_rewrite",
            "kb_id": 1,
        }
    )

    assert decision.route == "supervisor_plan"
    assert decision.needs_rag is True
    assert decision.needs_verification is True
    assert decision.child_agents == ["research_agent"]


def test_learning_scenario_blueprint_tool_is_registered_and_allowed():
    tool_names = {item["name"] for item in list_tools()}
    assert "learning.scenario_blueprint" in tool_names

    result = call_tool(
        "learning.scenario_blueprint",
        {"scenario_key": "technical_mock_30m"},
        agent="tool_agent",
    )

    assert result["ok"] is True
    assert result["data"]["scenario"]["task_type"] == "scenario_technical_mock_30m"
    assert "30 分钟面试议程" in result["data"]["scenario"]["deliverables"]


def test_memory_is_not_exposed_through_the_tool_registry():
    tool_names = {item["name"] for item in list_tools()}

    assert not any(name.startswith("memory.") for name in tool_names)
