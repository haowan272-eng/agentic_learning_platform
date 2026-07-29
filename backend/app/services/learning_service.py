from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import (
    AgentTask,
    LearningAssessment,
    LearningEvent,
    LearningPractice,
    LearningProfile,
    LearningReviewItem,
    LearningWeakness,
    User,
)
from app.services.learning_assets import (
    LONG_TERM_ASSET_TYPES,
    TECH_STACK_TERMS,
    _dumps,
    _infer_target_role,
    _loads,
    _persist_learning_long_term_assets,
    build_long_term_asset_snapshot,
    list_long_term_assets,
    long_term_asset_to_dict,
    upsert_long_term_asset,
)
from app.services.learning_interview import (
    INTERVIEW_CAPABILITY_DIMENSIONS,
    INTERVIEW_CAPABILITY_TITLES,
    INTERVIEW_SCORING_RUBRIC,
    INTERVIEW_SCORING_RUBRIC_TITLES,
    _WEAKNESS_SIGNAL_RE,
    _dimension_by_title,
    _dimension_segment,
    _weakness_severity,
    _weakness_topic,
    build_interview_capability_query,
    build_interview_scorecard,
    interview_scoring_rubric_for_prompt,
    interview_scoring_rubric_payload,
    is_interview_capability_request,
)


def resolve_user_id(db: Session, username: str) -> int:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise ValueError("用户不存在")
    return int(user.id)


def get_or_create_profile(db: Session, user_id: int) -> LearningProfile:
    profile = db.query(LearningProfile).filter(LearningProfile.user_id == user_id).first()
    if profile:
        return profile
    profile = LearningProfile(user_id=user_id, current_level="unknown", weekly_minutes=300, readiness_score=0.0)
    db.add(profile)
    db.commit()
    db.refresh(profile)
    return profile


def upsert_profile(db: Session, user_id: int, payload: dict[str, Any]) -> LearningProfile:
    profile = get_or_create_profile(db, user_id)
    profile.target_role = payload.get("target_role")
    profile.goal = payload.get("goal")
    profile.current_level = payload.get("current_level") or "unknown"
    profile.weekly_minutes = int(payload.get("weekly_minutes") or 300)
    profile.preference_json = _dumps(payload.get("preferences") or {})
    db.commit()
    db.refresh(profile)
    if profile.target_role:
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="target_role",
            asset_key="primary",
            title=profile.target_role,
            summary=profile.goal,
            payload={
                "target_role": profile.target_role,
                "goal": profile.goal,
                "current_level": profile.current_level,
                "weekly_minutes": profile.weekly_minutes,
                "preferences": _loads(profile.preference_json, {}),
            },
            evidence={"source": "learning_profile"},
            source="user",
        )
    record_learning_event(db, user_id, "profile.updated", payload=payload)
    return profile


def profile_to_dict(profile: LearningProfile) -> dict[str, Any]:
    return {
        "id": profile.id,
        "target_role": profile.target_role,
        "goal": profile.goal,
        "current_level": profile.current_level,
        "weekly_minutes": profile.weekly_minutes,
        "preferences": _loads(profile.preference_json, {}),
        "diagnostic_summary": profile.diagnostic_summary,
        "readiness_score": profile.readiness_score,
        "updated_at": profile.updated_at,
    }


def weakness_to_dict(item: LearningWeakness) -> dict[str, Any]:
    return {
        "id": item.id,
        "topic": item.topic,
        "category": item.category,
        "severity": item.severity,
        "confidence": item.confidence,
        "evidence": _loads(item.evidence_json, {}),
        "status": item.status,
        "updated_at": item.updated_at,
    }


def practice_to_dict(item: LearningPractice) -> dict[str, Any]:
    return {
        "id": item.id,
        "task_id": item.task_id,
        "kb_id": item.kb_id,
        "topic": item.topic,
        "question": item.question,
        "expected_answer": item.expected_answer,
        "difficulty": item.difficulty,
        "source": _loads(item.source_json, {}),
        "status": item.status,
        "created_at": item.created_at,
    }


def assessment_to_dict(item: LearningAssessment) -> dict[str, Any]:
    return {
        "id": item.id,
        "practice_id": item.practice_id,
        "task_id": item.task_id,
        "feedback": item.feedback,
        "score": item.score,
        "rubric": _loads(item.rubric_json, {}),
        "created_at": item.created_at,
    }


def review_to_dict(item: LearningReviewItem) -> dict[str, Any]:
    return {
        "id": item.id,
        "weakness_id": item.weakness_id,
        "topic": item.topic,
        "prompt": item.prompt,
        "due_at": item.due_at,
        "interval_days": item.interval_days,
        "status": item.status,
    }


def record_learning_event(db: Session, user_id: int | None, event_type: str, *, task_id: str | None = None, payload: dict[str, Any] | None = None) -> None:
    if not user_id:
        return
    db.add(LearningEvent(user_id=user_id, event_type=event_type, task_id=task_id, payload_json=_dumps(payload or {})))
    db.commit()


def upsert_weakness(
    db: Session,
    *,
    user_id: int,
    topic: str,
    category: str = "knowledge",
    severity: float = 0.5,
    confidence: float = 0.6,
    evidence: dict[str, Any] | None = None,
    task_id: str | None = None,
) -> LearningWeakness:
    topic = topic.strip()[:160] or "待明确薄弱点"
    item = db.query(LearningWeakness).filter(LearningWeakness.user_id == user_id, LearningWeakness.topic == topic).first()
    if not item:
        item = LearningWeakness(user_id=user_id, topic=topic)
        db.add(item)
    item.category = category
    item.severity = max(float(item.severity or 0), float(severity))
    item.confidence = max(float(item.confidence or 0), float(confidence))
    item.evidence_json = _dumps(evidence or {})
    item.last_seen_task_id = task_id
    item.status = "open"
    db.commit()
    db.refresh(item)
    return item


def create_practice(
    db: Session,
    *,
    user_id: int,
    topic: str,
    question: str,
    expected_answer: str | None = None,
    difficulty: str = "medium",
    task_id: str | None = None,
    kb_id: int | None = None,
    source: dict[str, Any] | None = None,
) -> LearningPractice:
    item = LearningPractice(
        user_id=user_id,
        task_id=task_id,
        kb_id=kb_id,
        topic=topic[:160],
        question=question,
        expected_answer=expected_answer,
        difficulty=difficulty,
        source_json=_dumps(source or {}),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def create_assessment(db: Session, user_id: int, payload: dict[str, Any]) -> LearningAssessment:
    rubric = payload.get("rubric") or {}
    if not rubric.get("items"):
        rubric = {
            **build_interview_scorecard(
                text=f"{payload.get('answer') or ''}\n{payload.get('feedback') or ''}",
                citation_count=0,
                weakness_count=1 if _WEAKNESS_SIGNAL_RE.search(str(payload.get("feedback") or "")) else 0,
            ),
            "source": "default_interview_scoring_rubric",
        }
    item = LearningAssessment(
        user_id=user_id,
        practice_id=payload.get("practice_id"),
        task_id=payload.get("task_id"),
        answer=payload.get("answer"),
        feedback=payload["feedback"],
        score=float(payload["score"]),
        rubric_json=_dumps(rubric),
    )
    db.add(item)
    if item.practice_id:
        practice = db.query(LearningPractice).filter(LearningPractice.id == item.practice_id, LearningPractice.user_id == user_id).first()
        if practice:
            practice.status = "assessed"
    db.commit()
    db.refresh(item)
    record_learning_event(db, user_id, "practice.assessed", task_id=item.task_id, payload={"score": item.score, "rubric": rubric})
    return item


def schedule_review(
    db: Session,
    *,
    user_id: int,
    topic: str,
    prompt: str,
    weakness_id: int | None = None,
    interval_days: int = 2,
) -> LearningReviewItem:
    item = LearningReviewItem(
        user_id=user_id,
        weakness_id=weakness_id,
        topic=topic[:160],
        prompt=prompt,
        due_at=datetime.now(timezone.utc) + timedelta(days=interval_days),
        interval_days=interval_days,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item


def record_rag_interview_diagnostic(
    db: Session,
    *,
    user_id: int,
    query: str,
    answer: str,
    citations: list[Any] | None = None,
    kb_id: int | None = None,
    document_id: int | None = None,
    conversation_id: int | None = None,
) -> None:
    citation_payload: list[dict[str, Any]] = []
    for item in citations or []:
        if hasattr(item, "model_dump"):
            citation_payload.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            citation_payload.append(dict(item))
        else:
            citation_payload.append({"value": str(item)})
    profile = get_or_create_profile(db, user_id)
    profile.goal = profile.goal or query[:1000]
    profile.target_role = profile.target_role or _infer_target_role(query)
    profile.current_level = "diagnosed"
    profile.diagnostic_summary = answer[:1200]

    weakness_candidates: list[tuple[dict[str, str], str, float]] = []
    for index, dimension in enumerate(INTERVIEW_CAPABILITY_DIMENSIONS):
        segment = _dimension_segment(answer, dimension["title"])
        severity = _weakness_severity(segment or answer, index)
        if severity:
            weakness_candidates.append((dimension, segment or answer, severity))

    if not weakness_candidates and "参考资料未明确说明" in answer:
        for index, dimension in enumerate(INTERVIEW_CAPABILITY_DIMENSIONS[:3]):
            weakness_candidates.append((dimension, _dimension_segment(answer, dimension["title"]) or answer, 0.6 - index * 0.04))

    scorecard = build_interview_scorecard(
        text=answer,
        citation_count=len(citation_payload),
        weakness_count=len(weakness_candidates),
    )
    profile.readiness_score = min(
        1.0,
        max(
            float(profile.readiness_score or 0.0),
            float(scorecard["total_score"]) / 100,
        ),
    )
    db.commit()

    created_topics: list[str] = []
    for index, (dimension, segment, severity) in enumerate(weakness_candidates[:6]):
        topic = _weakness_topic(dimension["title"], segment)
        weakness = upsert_weakness(
            db,
            user_id=user_id,
            topic=topic,
            category=dimension["key"],
            severity=severity,
            confidence=0.78 if citation_payload else 0.58,
            evidence={
                "source": "rag_interview_diagnostic",
                "capability": dimension["title"],
                "focus": dimension["focus"],
                "document_id": document_id,
                "kb_id": kb_id,
                "conversation_id": conversation_id,
                "citation_count": len(citation_payload),
                "citations": citation_payload[:3],
                "segment": segment[:800],
                "scoring_rubric": interview_scoring_rubric_payload(),
                "scorecard": scorecard,
            },
        )
        create_practice(
            db,
            user_id=user_id,
            kb_id=kb_id,
            topic=topic,
            question=dimension["practice"],
            expected_answer=(
                f"回答需要覆盖：{dimension['focus']}；必须引用简历或 JD 中的具体证据，资料缺口要主动说明。"
                "完成后按准确性、完整性、结构性、深度、可信度、面试适配度自评 0-5 分。"
            ),
            difficulty="hard" if severity >= 0.72 else "medium",
            source={
                "agent": "rag_first_answer",
                "weakness_id": weakness.id,
                "document_id": document_id,
                "conversation_id": conversation_id,
                "scoring_rubric": interview_scoring_rubric_payload(),
            },
        )
        schedule_review(
            db,
            user_id=user_id,
            weakness_id=weakness.id,
            topic=topic,
            prompt=f"复习「{dimension['title']}」：先补齐 {dimension['focus']}，再准备 1 个可引用的简历证据。",
            interval_days=1 + min(index, 2),
        )
        created_topics.append(topic)

    _persist_learning_long_term_assets(
        db,
        user_id=user_id,
        query=query,
        answer=answer,
        topics=created_topics or [item["title"] for item in INTERVIEW_CAPABILITY_DIMENSIONS[:3]],
        scorecard=scorecard,
        citations=citation_payload,
        source="rag_interview_diagnostic",
        kb_id=kb_id,
        document_id=document_id,
        conversation_id=conversation_id,
    )
    record_learning_event(
        db,
        user_id,
        "rag.interview_diagnostic_recorded",
        payload={
            "query": query[:500],
            "document_id": document_id,
            "kb_id": kb_id,
            "conversation_id": conversation_id,
            "topics": created_topics,
            "capabilities": [item["title"] for item in INTERVIEW_CAPABILITY_DIMENSIONS],
            "scorecard": scorecard,
            "scoring_rubric": interview_scoring_rubric_payload(),
        },
    )


def record_agent_learning_outputs(db: Session, state: dict[str, Any], final_answer: str) -> None:
    user_id = state.get("user_id")
    if not user_id:
        return
    text = f"{state.get('user_input') or ''}\n{final_answer}"
    topics = [
        dimension["title"]
        for dimension in INTERVIEW_CAPABILITY_DIMENSIONS
        if dimension["title"] in text or dimension["key"].lower() in text.lower()
    ]
    for keyword in ["知识盲区", "rag", "agent", "索引", "数据库", "缓存", "面试追问"]:
        if keyword.lower() in text.lower() and keyword not in topics:
            topics.append(keyword)
    if not topics:
        topics = [dimension["title"] for dimension in INTERVIEW_CAPABILITY_DIMENSIONS[:3]]

    scorecard = build_interview_scorecard(
        text=final_answer,
        citation_count=len(state.get("citations") or []),
        weakness_count=len(topics),
    )
    profile = get_or_create_profile(db, int(user_id))
    profile.goal = profile.goal or str(state.get("user_input") or "")[:1000]
    profile.target_role = profile.target_role or _infer_target_role(f"{state.get('user_input') or ''}\n{final_answer}")
    profile.current_level = "diagnosed"
    profile.diagnostic_summary = final_answer[:1200]
    profile.readiness_score = min(1.0, max(float(profile.readiness_score or 0.0), float(scorecard["total_score"]) / 100))
    db.commit()

    for index, topic in enumerate(topics[:3]):
        dimension = _dimension_by_title(topic)
        weakness = upsert_weakness(
            db,
            user_id=int(user_id),
            topic=topic,
            category=dimension["key"] if topic in INTERVIEW_CAPABILITY_TITLES else ("interview" if "面试" in topic or "表达" in topic else "knowledge"),
            severity=max(0.35, 0.72 - index * 0.12),
            confidence=0.7,
            evidence={
                "source": "agent_runtime",
                "capability": dimension["title"] if topic in INTERVIEW_CAPABILITY_TITLES else topic,
                "focus": dimension["focus"] if topic in INTERVIEW_CAPABILITY_TITLES else None,
                "citation_count": len(state.get("citations") or []),
                "scorecard": scorecard,
                "scoring_rubric": interview_scoring_rubric_payload(),
            },
            task_id=state.get("task_id"),
        )
        create_practice(
            db,
            user_id=int(user_id),
            task_id=state.get("task_id"),
            kb_id=state.get("kb_id"),
            topic=topic,
            question=dimension["practice"] if topic in INTERVIEW_CAPABILITY_TITLES else f"请围绕「{topic}」做一次 3 分钟面试回答，并说明你的例子、取舍和风险。",
            expected_answer=(
                f"回答应包含{dimension['focus']}，并补充项目证据、可量化结果和可能追问；"
                "按准确性、完整性、结构性、深度、可信度、面试适配度自评。"
            ) if topic in INTERVIEW_CAPABILITY_TITLES else "回答应包含核心概念、项目证据、可量化结果和可能追问，并按统一面试评分 Rubric 自评。",
            difficulty="medium" if index else "hard",
            source={"agent": "practice_agent", "weakness_id": weakness.id, "scoring_rubric": interview_scoring_rubric_payload()},
        )
        schedule_review(
            db,
            user_id=int(user_id),
            weakness_id=weakness.id,
            topic=topic,
            prompt=f"复习「{topic}」：先用 5 句话解释，再补一个来自你项目资料的证据。",
            interval_days=1 + index,
        )
    citation_payload: list[dict[str, Any]] = []
    for item in state.get("citations") or []:
        if hasattr(item, "model_dump"):
            citation_payload.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            citation_payload.append(dict(item))
        else:
            citation_payload.append({"value": str(item)})
    _persist_learning_long_term_assets(
        db,
        user_id=int(user_id),
        query=str(state.get("user_input") or ""),
        answer=final_answer,
        topics=topics[:3],
        scorecard=scorecard,
        citations=citation_payload,
        source="agent_runtime",
        task_id=state.get("task_id"),
        kb_id=state.get("kb_id"),
        document_id=state.get("document_id"),
        conversation_id=state.get("conversation_id"),
    )
    record_learning_event(db, int(user_id), "agent.learning_outputs_recorded", task_id=state.get("task_id"), payload={"topics": topics[:3], "scorecard": scorecard})


def build_dashboard(db: Session, user_id: int) -> dict[str, Any]:
    since = datetime.now(timezone.utc) - timedelta(days=14)
    active_days = {
        row[0].date().isoformat()
        for row in db.query(LearningEvent.created_at).filter(LearningEvent.user_id == user_id, LearningEvent.created_at >= since).all()
        if row[0]
    }
    completed_tasks = db.query(AgentTask).filter(
        AgentTask.user_id == user_id,
        AgentTask.status == "completed",
        AgentTask.updated_at >= since,
    ).count()
    assessments = db.query(LearningAssessment).filter(LearningAssessment.user_id == user_id, LearningAssessment.created_at >= since).all()
    practice_accuracy = sum(float(item.score or 0) for item in assessments) / len(assessments) if assessments else 0.0
    open_weaknesses = db.query(LearningWeakness).filter(LearningWeakness.user_id == user_id, LearningWeakness.status == "open").count()
    practices_14d = db.query(LearningPractice).filter(LearningPractice.user_id == user_id, LearningPractice.created_at >= since).count()
    cited_tasks = db.query(AgentTask).filter(
        AgentTask.user_id == user_id,
        AgentTask.status == "completed",
        AgentTask.updated_at >= since,
        AgentTask.final_answer.isnot(None),
    ).count()
    material_hit_rate = min(1.0, cited_tasks / max(completed_tasks, 1)) if completed_tasks else 0.0
    due_reviews = db.query(LearningReviewItem).filter(
        LearningReviewItem.user_id == user_id,
        LearningReviewItem.status == "due",
        LearningReviewItem.due_at <= datetime.now(timezone.utc) + timedelta(days=3),
    ).count()
    top_weaknesses = db.query(LearningWeakness).filter(LearningWeakness.user_id == user_id).order_by(LearningWeakness.severity.desc()).limit(5).all()
    recent_practices = db.query(LearningPractice).filter(LearningPractice.user_id == user_id).order_by(LearningPractice.created_at.desc()).limit(5).all()
    trend_rows = db.query(LearningWeakness.category, func.count(LearningWeakness.id)).filter(
        LearningWeakness.user_id == user_id,
    ).group_by(LearningWeakness.category).all()
    return {
        "active_days_14d": len(active_days),
        "tasks_completed_14d": completed_tasks,
        "practice_accuracy": round(practice_accuracy, 3),
        "open_weaknesses": open_weaknesses,
        "weakness_trend": [{"category": category, "count": count} for category, count in trend_rows],
        "material_hit_rate": round(material_hit_rate, 3),
        "agent_saved_minutes": completed_tasks * 18 + practices_14d * 6,
        "due_reviews": due_reviews,
        "recent_practices": [practice_to_dict(item) for item in recent_practices],
        "top_weaknesses": [weakness_to_dict(item) for item in top_weaknesses],
    }
