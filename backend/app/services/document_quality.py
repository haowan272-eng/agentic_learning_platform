"""Document type specifications and deterministic quality checks."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models.document import Document
from app.models.document_quality import DocumentQualityReport
from app.rag.chunk_models import DocumentChunk


@dataclass(frozen=True)
class DocumentTypeSpec:
    code: str
    label: str
    primary_use: str
    required_fields: tuple[str, ...]
    recommended_fields: tuple[str, ...]
    chunk_strategy: str
    prompt_template: str
    generated_tasks: tuple[str, ...]


DOCUMENT_TYPE_SPECS: dict[str, DocumentTypeSpec] = {
    "resume": DocumentTypeSpec(
        code="resume",
        label="简历",
        primary_use="画像、岗位匹配、面试追问",
        required_fields=("basic_info", "skills", "projects", "work_experience"),
        recommended_fields=("education", "metrics", "achievements", "target_role"),
        chunk_strategy="split_by_section_then_project",
        prompt_template="extract_resume_profile",
        generated_tasks=("jd_match", "resume_rewrite", "interview_followups"),
    ),
    "jd": DocumentTypeSpec(
        code="jd",
        label="JD",
        primary_use="岗位理解、差距分析、学习路径",
        required_fields=("title", "responsibilities", "hard_skills", "experience_requirements"),
        recommended_fields=("soft_skills", "bonus_items", "seniority", "keyword_weights"),
        chunk_strategy="split_by_requirement_group",
        prompt_template="extract_jd_requirements",
        generated_tasks=("skill_gap_analysis", "learning_plan", "resume_keyword_alignment"),
    ),
    "project_review": DocumentTypeSpec(
        code="project_review",
        label="项目复盘",
        primary_use="项目深挖、STAR 表达、面试追问",
        required_fields=("background", "goal", "solution", "personal_contribution", "result"),
        recommended_fields=("architecture", "technical_details", "tradeoffs", "metrics", "reflection"),
        chunk_strategy="split_by_star_and_technical_blocks",
        prompt_template="extract_project_review",
        generated_tasks=("star_answer", "technical_followups", "resume_project_bullets"),
    ),
    "interview_notes": DocumentTypeSpec(
        code="interview_notes",
        label="八股笔记",
        primary_use="知识问答、刷题、概念补齐",
        required_fields=("topic", "question", "answer", "principle"),
        recommended_fields=("scenario", "pitfalls", "followups", "examples", "mastery_level"),
        chunk_strategy="split_by_question_answer",
        prompt_template="extract_interview_notes",
        generated_tasks=("quiz", "followup_questions", "weakness_review"),
    ),
    "interview_record": DocumentTypeSpec(
        code="interview_record",
        label="面试记录",
        primary_use="复盘、题库沉淀、能力评估",
        required_fields=("company", "role", "round", "questions", "answers"),
        recommended_fields=("feedback", "weaknesses", "next_actions", "result", "date"),
        chunk_strategy="split_by_interview_round_and_question",
        prompt_template="extract_interview_record",
        generated_tasks=("post_interview_review", "weakness_tasks", "company_question_bank"),
    ),
    "company_question_bank": DocumentTypeSpec(
        code="company_question_bank",
        label="公司题库",
        primary_use="定向备战、公司画像",
        required_fields=("company", "role", "questions", "knowledge_points"),
        recommended_fields=("difficulty", "answer", "frequency", "round", "last_seen_at"),
        chunk_strategy="split_by_company_role_question",
        prompt_template="extract_company_question_bank",
        generated_tasks=("targeted_drill", "company_profile", "question_frequency_report"),
    ),
    "code_project_readme": DocumentTypeSpec(
        code="code_project_readme",
        label="代码项目说明",
        primary_use="项目理解、代码 RAG、技术表达",
        required_fields=("overview", "tech_stack", "module_structure", "startup"),
        recommended_fields=("core_flow", "key_implementation", "data_flow", "api", "call_chain"),
        chunk_strategy="split_by_module_and_flow",
        prompt_template="extract_code_project_readme",
        generated_tasks=("codebase_overview", "technical_explanation", "onboarding_checklist"),
    ),
    "learning_note": DocumentTypeSpec(
        code="learning_note",
        label="学习笔记",
        primary_use="知识沉淀、学习路线",
        required_fields=("topic", "knowledge_points", "summary"),
        recommended_fields=("examples", "practice", "usage", "review_points", "target_role"),
        chunk_strategy="split_by_topic_and_example",
        prompt_template="extract_learning_note",
        generated_tasks=("review_plan", "flashcards", "practice_tasks"),
    ),
    "offer_result": DocumentTypeSpec(
        code="offer_result",
        label="Offer / 面试结果",
        primary_use="分析求职转化率、岗位匹配效果",
        required_fields=("company", "role", "result", "stage"),
        recommended_fields=("compensation", "timeline", "reason", "followup", "conversion_notes"),
        chunk_strategy="split_by_company_and_stage",
        prompt_template="extract_offer_result",
        generated_tasks=("conversion_analysis", "offer_comparison", "next_application_strategy"),
    ),
    "technical_design": DocumentTypeSpec(
        code="technical_design",
        label="技术方案 / 架构设计",
        primary_use="沉淀系统设计能力，用于高阶面试和项目包装",
        required_fields=("background", "goal", "architecture", "solution"),
        recommended_fields=("tradeoffs", "risks", "metrics", "alternatives", "implementation_plan"),
        chunk_strategy="split_by_architecture_decision",
        prompt_template="extract_technical_design",
        generated_tasks=("system_design_answer", "architecture_review", "project_packaging"),
    ),
}


FIELD_KEYWORDS: dict[str, tuple[str, ...]] = {
    "answer": ("答案", "回答", "解法", "参考答案"),
    "api": ("API", "接口", "endpoint", "路由"),
    "architecture": ("架构", "架构图", "分层", "系统设计", "拓扑"),
    "background": ("背景", "场景", "问题", "现状", "痛点"),
    "basic_info": ("姓名", "电话", "邮箱", "GitHub", "基本信息"),
    "bonus_items": ("加分", "优先", "bonus", "nice to have"),
    "call_chain": ("调用链", "链路", "调用关系", "时序"),
    "company": ("公司", "企业", "厂", "雇主"),
    "core_flow": ("核心流程", "业务流程", "流程", "主链路"),
    "data_flow": ("数据流", "数据链路", "输入", "输出"),
    "date": ("日期", "时间", "面试时间"),
    "difficulty": ("难度", "简单", "中等", "困难"),
    "education": ("教育", "学历", "学校", "本科", "硕士"),
    "examples": ("示例", "例子", "case", "demo"),
    "experience_requirements": ("经验", "年限", "年以上", "工作经验"),
    "feedback": ("反馈", "评价", "面试官", "复盘"),
    "frequency": ("频率", "出现次数", "高频", "低频"),
    "goal": ("目标", "目的", "期望", "指标"),
    "hard_skills": ("技能", "技术栈", "熟悉", "掌握", "精通"),
    "implementation_plan": ("实施计划", "排期", "里程碑", "落地"),
    "key_implementation": ("关键实现", "核心实现", "源码", "算法"),
    "keyword_weights": ("权重", "关键词", "优先级"),
    "knowledge_points": ("知识点", "考点", "原理", "概念"),
    "mastery_level": ("掌握程度", "未学", "理解", "会讲", "实战"),
    "metrics": ("%", "％", "QPS", "TPS", "耗时", "延迟", "成本", "提升", "下降", "准确率"),
    "module_structure": ("目录结构", "模块", "包结构", "组件"),
    "next_actions": ("下一步", "行动", "计划", "待办"),
    "overview": ("简介", "概述", "项目目标", "项目介绍"),
    "personal_contribution": ("我负责", "本人负责", "个人贡献", "主导", "独立", "我实现"),
    "pitfalls": ("易错", "坑", "注意", "反例"),
    "practice": ("练习", "实践", "实验", "作业"),
    "principle": ("原理", "机制", "为什么", "底层"),
    "projects": ("项目", "项目经历", "系统", "平台"),
    "question": ("问题", "题目", "Q:", "问："),
    "questions": ("问题", "题目", "考题", "八股"),
    "reflection": ("反思", "总结", "复盘", "改进"),
    "result": ("结果", "效果", "收益", "产出", "上线", "通过", "拒绝", "offer"),
    "review_points": ("复习", "回顾", "待复习", "遗忘"),
    "risks": ("风险", "缺陷", "限制", "隐患"),
    "role": ("岗位", "职位", "方向", "角色"),
    "round": ("轮次", "一面", "二面", "三面", "HR"),
    "scenario": ("场景", "应用", "适用", "使用场景"),
    "seniority": ("初级", "中级", "高级", "专家", "级别"),
    "skills": ("技能", "技术栈", "Java", "Python", "Redis", "MySQL"),
    "soft_skills": ("沟通", "协作", "抗压", "owner", "推动"),
    "solution": ("方案", "解决方案", "设计", "实现"),
    "startup": ("启动", "运行", "安装", "部署", "README"),
    "stage": ("阶段", "轮次", "流程", "状态"),
    "summary": ("总结", "小结", "结论", "理解"),
    "target_role": ("目标岗位", "求职方向", "目标角色"),
    "tech_stack": ("技术栈", "框架", "数据库", "中间件", "Redis", "MySQL", "Spring"),
    "technical_details": ("技术细节", "实现细节", "源码", "算法", "SQL", "缓存", "并发"),
    "title": ("标题", "岗位名称", "职位名称"),
    "topic": ("主题", "标题", "Topic", "知识点"),
    "tradeoffs": ("取舍", "为什么", "替代方案", "对比", "不选"),
    "usage": ("使用", "应用", "落地", "怎么用"),
    "weaknesses": ("薄弱", "不足", "不会", "失败原因"),
    "work_experience": ("工作经历", "实习", "任职", "公司"),
}

TECHNICAL_TERMS = (
    "Java", "Python", "Go", "TypeScript", "React", "Vue", "Spring", "Django",
    "FastAPI", "MySQL", "PostgreSQL", "Redis", "MongoDB", "Kafka", "RabbitMQ",
    "Docker", "Kubernetes", "Qdrant", "LangChain", "RAG", "embedding", "向量",
    "索引", "缓存", "队列", "并发", "事务", "限流", "分库分表", "一致性", "微服务",
)
EMPTY_PHRASES = ("参与", "负责相关", "熟悉", "了解", "优化了系统", "提升用户体验")
METRIC_PATTERN = re.compile(
    r"(\d+(\.\d+)?\s*(%|％|ms|s|秒|分钟|小时|QPS|TPS|w|万|k|K|MB|GB|倍))|"
    r"((提升|下降|减少|增加|降低|优化|节省)\s*\d+)",
    re.IGNORECASE,
)


def list_document_type_specs() -> list[dict[str, Any]]:
    return [_spec_to_dict(spec) for spec in DOCUMENT_TYPE_SPECS.values()]


def normalize_document_type(value: str | None) -> str:
    if not value:
        return "unknown"
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "八股笔记": "interview_notes",
        "公司题库": "company_question_bank",
        "代码项目说明": "code_project_readme",
        "简历": "resume",
        "技术方案": "technical_design",
        "架构设计": "technical_design",
        "面试记录": "interview_record",
        "项目复盘": "project_review",
        "学习笔记": "learning_note",
        "offer": "offer_result",
    }
    normalized = aliases.get(normalized, normalized)
    return normalized if normalized in DOCUMENT_TYPE_SPECS else "unknown"


def infer_document_type(text: str, file_name: str | None = None) -> str:
    haystack = f"{file_name or ''}\n{text[:4000]}".lower()
    scores: dict[str, int] = {}
    for code, spec in DOCUMENT_TYPE_SPECS.items():
        score = 0
        label = spec.label.lower()
        if label and label in haystack:
            score += 5
        for field in (*spec.required_fields, *spec.recommended_fields):
            score += _field_presence_score(text, field)
        scores[code] = score
    best_code, best_score = max(scores.items(), key=lambda item: item[1])
    return best_code if best_score >= 4 else "learning_note"


def assess_document_quality(
    text: str,
    document_type: str | None = None,
    title: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    text = _normalize_text(text)
    scan_text = re.sub(r"\s+", " ", text)
    doc_type = normalize_document_type(document_type)
    inferred_type = infer_document_type(text, file_name=file_name) if doc_type == "unknown" else doc_type
    spec = DOCUMENT_TYPE_SPECS[inferred_type]
    present_required = [field for field in spec.required_fields if _field_present(text, field)]
    missing_required = [field for field in spec.required_fields if field not in present_required]
    present_recommended = [field for field in spec.recommended_fields if _field_present(text, field)]

    char_count = len(scan_text)
    heading_count = len(re.findall(r"(?m)^#{1,6}\s+", text))
    paragraph_count = len([part for part in re.split(r"\n\s*\n", text) if part.strip()])
    metrics_count = len(METRIC_PATTERN.findall(scan_text))
    technical_terms = _matched_terms(scan_text, TECHNICAL_TERMS)
    empty_phrase_count = sum(scan_text.count(phrase) for phrase in EMPTY_PHRASES)

    completeness = 25 * len(present_required) / max(1, len(spec.required_fields))
    information = min(15, char_count / 1200 * 15)
    structure = min(15, (heading_count * 3) + (paragraph_count * 1.2))
    technical = min(15, len(technical_terms) * 2.5)
    metrics = min(15, metrics_count * 5)
    contribution = 10 if _field_present(text, "personal_contribution") or inferred_type in {"jd", "company_question_bank", "interview_notes", "learning_note"} else 3
    followup = 5 if _field_present(text, "questions") or _field_present(text, "tradeoffs") or paragraph_count >= 4 else 2
    score = round(min(100, completeness + information + structure + technical + metrics + contribution + followup))

    issues: list[dict[str, str]] = []
    suggestions: list[str] = []

    if char_count < 300:
        _add_issue(issues, "too_little_content", "high", "资料内容过少，RAG 检索和面试追问的稳定性会比较弱。")
        suggestions.append("补充背景、过程、结论和可复用细节，让资料至少形成 3-5 个完整段落。")
    elif char_count < 800:
        _add_issue(issues, "thin_content", "medium", "资料信息量偏少，可能只能回答非常局部的问题。")
        suggestions.append("继续补充上下文、案例、关键决策或复盘结论。")

    for field in missing_required:
        _add_issue(
            issues,
            f"missing_required_{field}",
            "high",
            f"缺少必要字段：{field}。",
        )
    if missing_required:
        suggestions.append("按资料模板补齐必要字段后再用于核心推荐或面试模拟。")

    if inferred_type in {"resume", "project_review", "technical_design"} and metrics_count == 0:
        _add_issue(issues, "missing_metrics", "high", "缺少量化指标，例如 QPS、耗时、转化率、成本下降或准确率。")
        suggestions.append("补充项目上线效果、性能变化、业务收益或规模数据。")

    if inferred_type in {"project_review", "technical_design", "code_project_readme"} and len(technical_terms) < 3:
        _add_issue(issues, "lack_technical_details", "high", "技术细节不足，难以支撑面试深挖。")
        suggestions.append("补充架构、核心模块、关键实现、技术难点和方案取舍。")

    if inferred_type in {"resume", "project_review"} and not _field_present(text, "personal_contribution"):
        _add_issue(issues, "weak_personal_contribution", "medium", "个人贡献描述不够明确。")
        suggestions.append("说明你负责的模块、做出的决策、具体实现和协作边界。")

    if inferred_type == "project_review" and not _field_present(text, "result"):
        _add_issue(issues, "missing_project_result", "high", "项目复盘缺少结果或收益描述。")
        suggestions.append("补充项目最终结果、上线状态、业务收益和失败反思。")

    if empty_phrase_count >= 4:
        _add_issue(issues, "too_many_empty_phrases", "medium", "空泛表述较多，可信度和可追问性偏弱。")
        suggestions.append("把“参与/熟悉/优化”等表达改成具体动作、对象和结果。")

    level = _quality_level(score)
    if not suggestions:
        suggestions.append("资料质量较好，可以直接用于 RAG、岗位匹配和面试模拟。")

    return {
        "type": inferred_type,
        "type_label": spec.label,
        "title": title or _guess_title(text, file_name),
        "quality_score": score,
        "level": level,
        "summary": {
            "char_count": char_count,
            "paragraph_count": paragraph_count,
            "heading_count": heading_count,
            "metrics_count": metrics_count,
            "technical_terms": technical_terms,
            "present_required_fields": present_required,
            "missing_required_fields": missing_required,
            "present_recommended_fields": present_recommended,
        },
        "issues": issues,
        "suggestions": suggestions,
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


_assess_document_quality = assess_document_quality


def assess_document_chunks(db: Session, document: Document) -> dict[str, Any]:
    chunks = (
        db.query(DocumentChunk)
        .filter(DocumentChunk.document_id == document.id)
        .order_by(DocumentChunk.chunk_index.asc())
        .all()
    )
    text = "\n\n".join(chunk.content for chunk in chunks)
    return assess_document_quality(
        text,
        document_type=document.doc_type,
        title=document.title,
        file_name=document.original_file_name or document.file_name,
    )


def save_quality_report(
    db: Session,
    document: Document,
    report: dict[str, Any],
) -> DocumentQualityReport:
    document.doc_type = report["type"]
    document.title = report["title"]
    document.quality_score = report["quality_score"]
    document.quality_level = report["level"]
    document.structured_summary_json = json.dumps(report["summary"], ensure_ascii=False)

    quality = DocumentQualityReport(
        document_id=document.id,
        doc_type=report["type"],
        quality_score=report["quality_score"],
        level=report["level"],
        summary_json=json.dumps(report["summary"], ensure_ascii=False),
        issues_json=json.dumps(report["issues"], ensure_ascii=False),
        suggestions_json=json.dumps(report["suggestions"], ensure_ascii=False),
    )
    db.add(quality)
    db.flush()
    return quality


def report_to_dict(report: DocumentQualityReport) -> dict[str, Any]:
    return {
        "id": report.id,
        "document_id": report.document_id,
        "type": report.doc_type,
        "quality_score": report.quality_score,
        "level": report.level,
        "summary": _loads(report.summary_json, {}),
        "issues": _loads(report.issues_json, []),
        "suggestions": _loads(report.suggestions_json, []),
        "created_at": report.created_at.isoformat() if report.created_at else None,
    }


def _spec_to_dict(spec: DocumentTypeSpec) -> dict[str, Any]:
    return {
        "code": spec.code,
        "label": spec.label,
        "primary_use": spec.primary_use,
        "required_fields": list(spec.required_fields),
        "recommended_fields": list(spec.recommended_fields),
        "chunk_strategy": spec.chunk_strategy,
        "prompt_template": spec.prompt_template,
        "generated_tasks": list(spec.generated_tasks),
    }


def _normalize_text(text: str) -> str:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[\t\f\v ]+", " ", line).strip() for line in normalized.split("\n")]
    return "\n".join(lines).strip()


def _field_present(text: str, field: str) -> bool:
    return _field_presence_score(text, field) > 0


def _field_presence_score(text: str, field: str) -> int:
    keywords = FIELD_KEYWORDS.get(field, (field,))
    score = 0
    for keyword in keywords:
        if keyword and keyword.lower() in text.lower():
            score += 1
    return score


def _matched_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    lower = text.lower()
    matched = []
    for term in terms:
        if term.lower() in lower:
            matched.append(term)
    return matched[:12]


def _add_issue(issues: list[dict[str, str]], code: str, severity: str, message: str) -> None:
    if not any(issue["code"] == code for issue in issues):
        issues.append({"code": code, "severity": severity, "message": message})


def _quality_level(score: int) -> str:
    if score >= 90:
        return "excellent"
    if score >= 75:
        return "usable"
    if score >= 60:
        return "needs_improvement"
    return "low_quality"


def _guess_title(text: str, file_name: str | None) -> str:
    heading = re.search(r"(?m)^#{1,6}\s+(.+)$", text or "")
    if heading:
        return heading.group(1).strip()[:120]
    if file_name:
        return file_name
    return "Untitled document"


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default
