from __future__ import annotations

import re
from typing import Any


INTERVIEW_CAPABILITY_DIMENSIONS: tuple[dict[str, str], ...] = (
    {
        "key": "knowledge_understanding",
        "title": "知识理解",
        "focus": "概念、原理、对比、场景",
        "practice": "请围绕简历中最相关的一个技术点，说明概念、原理、同类方案对比和适用场景。",
    },
    {
        "key": "project_storytelling",
        "title": "项目表达",
        "focus": "背景、难点、方案、取舍、结果",
        "practice": "请选择简历中的一个项目，用背景、难点、方案、取舍、结果讲一段 3 分钟项目表达。",
    },
    {
        "key": "followup_handling",
        "title": "追问应对",
        "focus": "边界问题、性能问题、故障问题",
        "practice": "请针对简历中的项目准备 3 个追问：边界、性能、故障，并分别给出答题要点。",
    },
    {
        "key": "system_design",
        "title": "系统设计",
        "focus": "架构、瓶颈、扩展、可靠性",
        "practice": "请把简历中的核心项目画成口头架构说明，并解释瓶颈、扩展策略和可靠性设计。",
    },
    {
        "key": "communication",
        "title": "沟通表达",
        "focus": "结构化、简洁度、可信度",
        "practice": "请把一个项目回答压缩成 90 秒版本，要求先结论、再证据、最后风险与复盘。",
    },
    {
        "key": "jd_match",
        "title": "岗位匹配",
        "focus": "是否贴合 JD",
        "practice": "请用简历证据逐条匹配目标 JD，并标出最强匹配点和最需要补证据的点。",
    },
)

INTERVIEW_CAPABILITY_TITLES = tuple(item["title"] for item in INTERVIEW_CAPABILITY_DIMENSIONS)

INTERVIEW_SCORING_RUBRIC: tuple[dict[str, Any], ...] = (
    {
        "key": "accuracy",
        "title": "准确性",
        "weight": 0.20,
        "criterion": "是否基于参考资料、简历/JD 与可验证事实；不编造经历、指标、技术结论或因果关系。",
    },
    {
        "key": "completeness",
        "title": "完整性",
        "weight": 0.18,
        "criterion": "是否覆盖问题核心点、JD 关键要求、项目背景/动作/结果，以及必要的限制条件。",
    },
    {
        "key": "structure",
        "title": "结构性",
        "weight": 0.16,
        "criterion": "是否先结论后原因，能按 STAR、分层或总分结构组织，层次清楚、便于面试官追问。",
    },
    {
        "key": "depth",
        "title": "深度",
        "weight": 0.16,
        "criterion": "是否讲到原理、取舍、边界、风险、替代方案和复盘，而不是停留在表层描述。",
    },
    {
        "key": "credibility",
        "title": "可信度",
        "weight": 0.15,
        "criterion": "是否有项目证据、可量化结果、个人贡献、上下文约束和可被追问验证的细节。",
    },
    {
        "key": "interview_fit",
        "title": "面试适配度",
        "weight": 0.15,
        "criterion": "是否贴合目标岗位/JD、面试场景和回答时长，能突出岗位最关心的能力信号。",
    },
)

INTERVIEW_SCORING_RUBRIC_TITLES = tuple(item["title"] for item in INTERVIEW_SCORING_RUBRIC)

_INTERVIEW_SCOPE_TERMS = (
    "简历", "resume", "cv", "项目经历", "项目资料", "岗位", "jd", "职位", "面试", "interview", "求职",
)
_INTERVIEW_INTENT_TERMS = (
    "提升", "提优", "诊断", "薄弱", "weakness", "gap", "不足", "复盘", "总结", "追问", "模拟",
    "准备", "匹配", "能力", "评价", "评估", "改进", "练习",
)


def interview_scoring_rubric_for_prompt() -> str:
    lines = []
    for item in INTERVIEW_SCORING_RUBRIC:
        lines.append(f"- {item['title']}（权重 {int(float(item['weight']) * 100)}%）：{item['criterion']}")
    return "\n".join(lines)


def interview_scoring_rubric_payload() -> list[dict[str, Any]]:
    return [
        {
            "key": item["key"],
            "title": item["title"],
            "weight": item["weight"],
            "criterion": item["criterion"],
            "scale": "0-5，0=缺失或严重失真，3=基本合格但证据/深度不足，5=事实充分且面试表达优秀",
        }
        for item in INTERVIEW_SCORING_RUBRIC
    ]


def build_interview_scorecard(
    *,
    text: str = "",
    citation_count: int = 0,
    weakness_count: int = 0,
) -> dict[str, Any]:
    normalized = text or ""
    score_by_key: dict[str, float] = {}
    explanations: dict[str, str] = {}
    for item in INTERVIEW_SCORING_RUBRIC:
        key = str(item["key"])
        title = str(item["title"])
        score = 3.2
        reason = "已有基础回答，但仍需要更多可验证证据和面试化表达。"
        if citation_count <= 0 and key in {"accuracy", "credibility"}:
            score -= 1.0
            reason = "缺少引用或项目证据，事实支撑不足。"
        if _WEAKNESS_SIGNAL_RE.search(normalized):
            score -= 0.35 + min(weakness_count, 6) * 0.08
        if title in normalized:
            score += 0.25
            reason = "回答已显式覆盖该评分维度，但仍需按证据和追问继续打磨。"
        score_by_key[key] = round(max(0.0, min(5.0, score)), 1)
        explanations[key] = reason

    weighted_total = sum(
        score_by_key[str(item["key"])] * float(item["weight"]) * 20
        for item in INTERVIEW_SCORING_RUBRIC
    )
    return {
        "scale": "每项 0-5 分，总分按权重折算到 100 分。",
        "total_score": round(weighted_total, 1),
        "items": [
            {
                "key": item["key"],
                "title": item["title"],
                "weight": item["weight"],
                "score": score_by_key[str(item["key"])],
                "reason": explanations[str(item["key"])],
                "criterion": item["criterion"],
            }
            for item in INTERVIEW_SCORING_RUBRIC
        ],
    }
_WEAKNESS_SIGNAL_RE = re.compile(
    r"(薄弱|不足|缺少|缺失|未明确|不明确|不清晰|欠缺|风险|短板|待补|需要补|无法判断|没有体现|证据不足|不够)",
    re.IGNORECASE,
)
_STRONG_WEAKNESS_SIGNAL_RE = re.compile(
    r"(缺失|未明确|无法判断|没有体现|证据不足|严重|高风险|核心短板)",
    re.IGNORECASE,
)


def is_interview_capability_request(text: str, *, has_material_scope: bool = False) -> bool:
    """Whether a RAG request should use the interview capability rubric."""
    normalized = (text or "").strip().lower()
    if not normalized:
        return False
    has_scope = has_material_scope or any(term.lower() in normalized for term in _INTERVIEW_SCOPE_TERMS)
    has_intent = any(term.lower() in normalized for term in _INTERVIEW_INTENT_TERMS)
    return has_scope and has_intent


def build_interview_capability_query(user_query: str) -> str:
    dimensions = "\n".join(
        f"- {item['title']}：{item['focus']}"
        for item in INTERVIEW_CAPABILITY_DIMENSIONS
    )
    scoring_rubric = interview_scoring_rubric_for_prompt()
    return (
        f"{user_query.strip()}\n\n"
        "请把这次回答作为“面试能力提升”的第一次 RAG 诊断回答。"
        "只能基于参考资料中的简历、项目资料或 JD 进行提取和判断。\n"
        "请固定按以下能力模型组织输出：\n"
        f"{dimensions}\n\n"
        "同时使用以下可解释评分 Rubric 贯穿证据提取、问题总结、weakness 与第一次回答：\n"
        f"{scoring_rubric}\n"
        "评分要求：每项 0-5 分，并给出扣分原因、引用证据或资料缺口；总分按权重折算到 100 分。\n\n"
        "输出结构：\n"
        "1. 简历/JD 要点提取：按上述 6 个能力标题分别提取证据，并标出对应评分项。\n"
        "2. 面试问题总结：每个标题给出 1-2 个最可能被问的问题，并说明主要考察哪几项评分 Rubric。\n"
        "3. 评分基线：按准确性、完整性、结构性、深度、可信度、面试适配度逐项给 0-5 分、扣分原因和改进动作。\n"
        "4. Weakness 标记：说明薄弱点、证据是否充分、优先级，并映射到具体评分项。\n"
        "5. 第一次回答：选择最高优先级问题，给出一版可直接练习的面试回答；回答必须主动覆盖低分评分项。\n"
        "所有事实性判断都必须标注引用；资料未体现时明确写“参考资料未明确说明”。"
    )


def _dimension_by_title(title: str) -> dict[str, str]:
    return next((item for item in INTERVIEW_CAPABILITY_DIMENSIONS if item["title"] == title), INTERVIEW_CAPABILITY_DIMENSIONS[0])


def _dimension_segment(text: str, title: str) -> str:
    start = text.find(title)
    if start < 0:
        return ""
    end = len(text)
    for other in INTERVIEW_CAPABILITY_TITLES:
        if other == title:
            continue
        position = text.find(other, start + len(title))
        if position >= 0:
            end = min(end, position)
    return text[start:end].strip()


def _weakness_severity(segment: str, index: int) -> float:
    if _STRONG_WEAKNESS_SIGNAL_RE.search(segment):
        return max(0.72, 0.84 - index * 0.04)
    if _WEAKNESS_SIGNAL_RE.search(segment):
        return max(0.56, 0.72 - index * 0.04)
    return 0.0


def _weakness_topic(title: str, segment: str) -> str:
    for line in segment.splitlines():
        cleaned = line.strip(" -#：:0123456789.、")
        if cleaned and _WEAKNESS_SIGNAL_RE.search(cleaned):
            return f"{title}：{cleaned[:80]}"
    return f"{title}：待补齐面试证据"


