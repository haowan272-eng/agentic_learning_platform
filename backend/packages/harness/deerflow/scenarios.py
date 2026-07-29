from __future__ import annotations

from copy import deepcopy
from typing import Any


SCENARIO_CATALOG: dict[str, dict[str, Any]] = {
    "jd_interview_prep": {
        "label": "JD interview preparation",
        "task_type": "scenario_jd_interview_prep",
        "required_inputs": ["target JD", "resume or project material", "interview round / level"],
        "agents": ["memory_agent", "research_agent"],
        "tools": ["knowledge.answer", "knowledge.verify_claim"],
        "deliverables": ["JD 能力地图", "简历证据匹配表", "追问问题", "一面回答提纲", "训练优先级"],
        "research_tasks": [
            {"task_id": "jd-requirements", "query_template": "{goal} JD responsibilities required skills bonus points level interview focus", "objective": "Extract explicit JD requirements for skills, experience, project evidence, and technical depth.", "top_k": 8},
            {"task_id": "resume-match", "query_template": "{goal} resume projects experience tech stack business results STAR evidence match", "objective": "Retrieve evidence that supports matching the resume and projects to the JD.", "top_k": 8},
            {"task_id": "interview-risks", "query_template": "{goal} interview follow-up risks gaps weak evidence missing project details", "objective": "Identify likely interview gaps and risks that need reinforcement.", "top_k": 8},
        ],
    },
    "project_deep_dive": {
        "label": "Project deep dive",
        "task_type": "scenario_project_deep_dive",
        "required_inputs": ["project name/background", "candidate role", "core technical plan", "known difficulties"],
        "agents": ["research_agent"],
        "tools": ["knowledge.answer", "knowledge.verify_claim"],
        "deliverables": ["项目证据", "技术取舍", "边界追问", "风险地图"],
        "research_tasks": [
            {"task_id": "project-facts", "query_template": "{goal} project background role contribution result metrics evidence", "objective": "Retrieve project facts, personal contribution, and measurable results.", "top_k": 8},
            {"task_id": "technical-tradeoffs", "query_template": "{goal} technical selection architecture tradeoff bottleneck optimization alternative", "objective": "Retrieve technical tradeoffs, hard problems, and follow-up details.", "top_k": 8},
            {"task_id": "boundary-followups", "query_template": "{goal} follow-up boundary failure performance scalability reliability", "objective": "Prepare deeper boundary and reliability follow-up questions.", "top_k": 8},
        ],
    },
    "knowledge_ladder": {
        "label": "Knowledge ladder",
        "task_type": "scenario_knowledge_ladder",
        "required_inputs": ["knowledge point", "target role", "current mastery"],
        "agents": ["research_agent"],
        "tools": ["knowledge.answer", "knowledge.verify_claim"],
        "deliverables": ["概念阶梯", "进阶追问", "项目关联"],
        "research_tasks": [
            {"task_id": "concept-base", "query_template": "{goal} concept principle comparison scenario", "objective": "Retrieve first-layer concepts and core principles.", "top_k": 6},
            {"task_id": "advanced-followup", "query_template": "{goal} advanced details edge cases performance failures", "objective": "Retrieve advanced details for deeper follow-up.", "top_k": 8},
            {"task_id": "project-link", "query_template": "{goal} project evidence interview expression", "objective": "Connect the knowledge point to project evidence and interview expression.", "top_k": 6},
        ],
    },
    "resume_star_rewrite": {
        "label": "Resume STAR rewrite",
        "task_type": "scenario_resume_star_rewrite",
        "required_inputs": ["resume/project material", "target JD", "desired positioning"],
        "agents": ["memory_agent", "research_agent"],
        "tools": ["knowledge.answer", "knowledge.verify_claim"],
        "deliverables": ["STAR 改写", "证据表", "JD 关键词对齐"],
        "research_tasks": [
            {"task_id": "star-evidence", "query_template": "{goal} STAR situation task action result project evidence", "objective": "Retrieve project facts that can be rewritten into STAR form.", "top_k": 8},
            {"task_id": "measurable-results", "query_template": "{goal} metrics result business impact credibility", "objective": "Retrieve measurable results and credible business impact.", "top_k": 8},
            {"task_id": "jd-alignment", "query_template": "{goal} JD match keywords skills requirements resume expression", "objective": "Align project expression with target JD language and skills.", "top_k": 8},
        ],
    },
    "technical_mock_30m": {
        "label": "30-minute technical mock interview",
        "task_type": "scenario_technical_mock_30m",
        "required_inputs": ["target role/JD", "resume", "desired focus"],
        "agents": ["research_agent"],
        "tools": ["knowledge.answer", "knowledge.verify_claim"],
        "deliverables": ["30 分钟面试议程", "分阶段问题", "评分 Rubric", "追问策略", "复盘建议"],
        "research_tasks": [
            {"task_id": "opening", "query_template": "{goal} self introduction project overview role match interview opening", "objective": "Prepare opening screening and project overview questions.", "top_k": 6},
            {"task_id": "core-technical", "query_template": "{goal} core technology system design principle coding performance reliability", "objective": "Prepare core technical and system design follow-ups.", "top_k": 8},
            {"task_id": "closing-review", "query_template": "{goal} behavioral review candidate questions risk recap", "objective": "Prepare closing review and candidate question prompts.", "top_k": 6},
        ],
    },
    "weekly_training_plan": {
        "label": "Weekly training plan",
        "task_type": "scenario_weekly_training_plan",
        "required_inputs": ["recent answers", "weaknesses", "target role"],
        "agents": ["memory_agent", "research_agent"],
        "tools": ["knowledge.answer"],
        "deliverables": ["周训练日历", "薄弱点优先级", "每日练习", "复盘指标", "间隔复习"],
        "research_tasks": [
            {"task_id": "weakness-patterns", "query_template": "{goal} repeated weaknesses interview answers gaps", "objective": "Extract repeated weaknesses from recent answers.", "top_k": 6},
            {"task_id": "practice-material", "query_template": "{goal} practice material knowledge base exercises examples", "objective": "Retrieve materials that can support next-week practice.", "top_k": 8},
            {"task_id": "plan-constraints", "query_template": "{goal} training plan constraints rhythm acceptance metrics", "objective": "Organize constraints, rhythm, and acceptance metrics for training.", "top_k": 5},
        ],
    },
}


def get_scenario(key: str) -> dict[str, Any] | None:
    item = SCENARIO_CATALOG.get(key)
    return deepcopy(item) if item else None


def list_scenarios() -> list[dict[str, Any]]:
    return [{"key": key, **deepcopy(value)} for key, value in SCENARIO_CATALOG.items()]


def scenario_for_task_type(task_type: str) -> tuple[str | None, dict[str, Any] | None]:
    for key, value in SCENARIO_CATALOG.items():
        if value.get("task_type") == task_type:
            return key, deepcopy(value)
    return None, None


def build_scenario_plan(key: str, goal: str) -> dict[str, Any] | None:
    scenario = SCENARIO_CATALOG.get(key)
    if not scenario:
        return None
    return {
        "goal": goal,
        "intent": scenario["task_type"],
        "research_tasks": [
            {
                "task_id": item["task_id"],
                "query": item["query_template"].format(goal=goal),
                "objective": item["objective"],
                "top_k": item.get("top_k", 5),
            }
            for item in scenario.get("research_tasks", [])
        ],
        "approval_required": False,
        "approval_reason": None,
    }
