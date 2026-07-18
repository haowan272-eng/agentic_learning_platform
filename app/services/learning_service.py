from __future__ import annotations

import json
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


def _loads(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except Exception:
        return fallback


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


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
    item = LearningAssessment(
        user_id=user_id,
        practice_id=payload.get("practice_id"),
        task_id=payload.get("task_id"),
        answer=payload.get("answer"),
        feedback=payload["feedback"],
        score=float(payload["score"]),
        rubric_json=_dumps(payload.get("rubric") or {}),
    )
    db.add(item)
    if item.practice_id:
        practice = db.query(LearningPractice).filter(LearningPractice.id == item.practice_id, LearningPractice.user_id == user_id).first()
        if practice:
            practice.status = "assessed"
    db.commit()
    db.refresh(item)
    record_learning_event(db, user_id, "practice.assessed", task_id=item.task_id, payload={"score": item.score})
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


def record_agent_learning_outputs(db: Session, state: dict[str, Any], final_answer: str) -> None:
    user_id = state.get("user_id")
    if not user_id:
        return
    text = f"{state.get('user_input') or ''}\n{final_answer}"
    lower = text.lower()
    topics = []
    for keyword in ["项目表达", "知识盲区", "系统设计", "rag", "agent", "索引", "数据库", "缓存", "面试追问"]:
        if keyword.lower() in lower:
            topics.append(keyword)
    if not topics:
        topics = ["面试表达", "知识结构", "复盘深度"]

    profile = get_or_create_profile(db, int(user_id))
    profile.goal = profile.goal or str(state.get("user_input") or "")[:1000]
    profile.current_level = "diagnosed"
    profile.diagnostic_summary = final_answer[:1200]
    profile.readiness_score = min(1.0, max(float(profile.readiness_score or 0.0), 0.58 + min(len(state.get("citations") or []), 5) * 0.04))
    db.commit()

    for index, topic in enumerate(topics[:3]):
        weakness = upsert_weakness(
            db,
            user_id=int(user_id),
            topic=topic,
            category="interview" if "面试" in topic or "表达" in topic else "knowledge",
            severity=max(0.35, 0.72 - index * 0.12),
            confidence=0.7,
            evidence={"source": "agent_runtime", "citation_count": len(state.get("citations") or [])},
            task_id=state.get("task_id"),
        )
        create_practice(
            db,
            user_id=int(user_id),
            task_id=state.get("task_id"),
            kb_id=state.get("kb_id"),
            topic=topic,
            question=f"请围绕「{topic}」做一次 3 分钟面试回答，并说明你的例子、取舍和风险。",
            expected_answer="回答应包含核心概念、项目证据、可量化结果和可能追问。",
            difficulty="medium" if index else "hard",
            source={"agent": "practice_agent", "weakness_id": weakness.id},
        )
        schedule_review(
            db,
            user_id=int(user_id),
            weakness_id=weakness.id,
            topic=topic,
            prompt=f"复习「{topic}」：先用 5 句话解释，再补一个来自你项目资料的证据。",
            interval_days=1 + index,
        )
    record_learning_event(db, int(user_id), "agent.learning_outputs_recorded", task_id=state.get("task_id"), payload={"topics": topics[:3]})


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
