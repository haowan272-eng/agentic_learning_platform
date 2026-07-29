from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy.orm import Session

from app.models import LearningLongTermAsset
from app.services.learning_interview import interview_scoring_rubric_payload


LONG_TERM_ASSET_TYPES: dict[str, dict[str, str]] = {
    "target_role": {"title": "用户目标岗位", "owner": "learning_profile"},
    "tech_stack_profile": {"title": "技术栈画像", "owner": "learning_profile"},
    "project_experience": {"title": "项目经历库", "owner": "learning_profile"},
    "recurring_error": {"title": "高频错误", "owner": "learning_loop"},
    "weakness_map": {"title": "弱点地图", "owner": "learning_loop"},
    "interview_answer_version": {"title": "面试回答版本", "owner": "agent_output"},
    "mock_interview_record": {"title": "模拟面试记录", "owner": "agent_output"},
    "review_plan": {"title": "复习计划", "owner": "learning_loop"},
    "capability_trend": {"title": "能力趋势", "owner": "learning_loop"},
}

TECH_STACK_TERMS: tuple[str, ...] = (
    "Python",
    "Java",
    "JavaScript",
    "TypeScript",
    "Vue",
    "React",
    "FastAPI",
    "Django",
    "Flask",
    "Spring",
    "Redis",
    "MySQL",
    "PostgreSQL",
    "SQLite",
    "Docker",
    "Kubernetes",
    "RAG",
    "Agent",
    "LangGraph",
    "Qdrant",
    "Elasticsearch",
    "Kafka",
    "Celery",
    "LLM",
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


def _merge_dicts(base: Any, incoming: Any) -> dict[str, Any]:
    base_dict = dict(base) if isinstance(base, dict) else {}
    incoming_dict = dict(incoming) if isinstance(incoming, dict) else {}
    for key, value in incoming_dict.items():
        if isinstance(value, dict) and isinstance(base_dict.get(key), dict):
            base_dict[key] = _merge_dicts(base_dict[key], value)
        elif value is not None:
            base_dict[key] = value
    return base_dict


def _normalize_asset_type(asset_type: str) -> str:
    normalized = (asset_type or "").strip()
    if normalized not in LONG_TERM_ASSET_TYPES:
        allowed = ", ".join(sorted(LONG_TERM_ASSET_TYPES))
        raise ValueError(f"unknown long-term asset type: {asset_type}; allowed: {allowed}")
    return normalized


def _asset_key(value: str) -> str:
    normalized = " ".join((value or "").strip().lower().split())
    return normalized[:160] or "default"


def _detected_tech_stack(text: str) -> list[str]:
    lowered = (text or "").lower()
    found = []
    for term in TECH_STACK_TERMS:
        if term.lower() in lowered and term not in found:
            found.append(term)
    return found


def _infer_target_role(text: str) -> str:
    normalized = (text or "").lower()
    if any(term in normalized for term in ("后端", "backend", "fastapi", "spring", "java")):
        return "后端工程师"
    if any(term in normalized for term in ("前端", "frontend", "vue", "react")):
        return "前端工程师"
    if any(term in normalized for term in ("算法", "机器学习", "ai", "llm", "rag", "agent")):
        return "AI 应用工程师"
    if any(term in normalized for term in ("数据", "sql", "数仓")):
        return "数据工程师"
    if "面试" in normalized or "简历" in normalized or "jd" in normalized:
        return "面试能力提升"
    return "学习提升"


def long_term_asset_to_dict(item: LearningLongTermAsset) -> dict[str, Any]:
    return {
        "id": item.id,
        "asset_type": item.asset_type,
        "asset_key": item.asset_key,
        "title": item.title,
        "summary": item.summary,
        "payload": _loads(item.payload_json, {}),
        "evidence": _loads(item.evidence_json, {}),
        "confidence": item.confidence,
        "weight": item.weight,
        "source": item.source,
        "status": item.status,
        "task_id": item.task_id,
        "kb_id": item.kb_id,
        "document_id": item.document_id,
        "conversation_id": item.conversation_id,
        "version": item.version,
        "trend_value": item.trend_value,
        "observed_at": item.observed_at,
        "updated_at": item.updated_at,
    }


def upsert_long_term_asset(
    db: Session,
    *,
    user_id: int,
    asset_type: str,
    title: str,
    asset_key: str | None = None,
    summary: str | None = None,
    payload: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
    confidence: float = 0.6,
    weight: float = 1.0,
    source: str = "agent",
    status: str = "active",
    task_id: str | None = None,
    kb_id: int | None = None,
    document_id: int | None = None,
    conversation_id: int | None = None,
    trend_value: float | None = None,
    observed_at: datetime | None = None,
    commit: bool = True,
) -> LearningLongTermAsset:
    asset_type = _normalize_asset_type(asset_type)
    key = _asset_key(asset_key or title)
    item = (
        db.query(LearningLongTermAsset)
        .filter(
            LearningLongTermAsset.user_id == user_id,
            LearningLongTermAsset.asset_type == asset_type,
            LearningLongTermAsset.asset_key == key,
        )
        .first()
    )
    if not item:
        item = LearningLongTermAsset(user_id=user_id, asset_type=asset_type, asset_key=key, title=title[:255])
        db.add(item)
    else:
        item.version = int(item.version or 1) + 1
    item.title = title[:255]
    if summary is not None:
        item.summary = summary
    item.payload_json = _dumps(_merge_dicts(_loads(item.payload_json, {}), payload or {}))
    item.evidence_json = _dumps(_merge_dicts(_loads(item.evidence_json, {}), evidence or {}))
    item.confidence = max(float(item.confidence or 0.0), float(confidence))
    item.weight = float(item.weight or 0.0) + float(weight)
    item.source = source or item.source or "agent"
    item.status = status or item.status or "active"
    item.task_id = task_id or item.task_id
    item.kb_id = kb_id if kb_id is not None else item.kb_id
    item.document_id = document_id if document_id is not None else item.document_id
    item.conversation_id = conversation_id if conversation_id is not None else item.conversation_id
    item.trend_value = trend_value if trend_value is not None else item.trend_value
    item.observed_at = observed_at or item.observed_at or datetime.now(timezone.utc)
    if commit:
        db.commit()
        db.refresh(item)
    else:
        db.flush()
    return item


def list_long_term_assets(
    db: Session,
    user_id: int,
    *,
    asset_type: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    query = db.query(LearningLongTermAsset).filter(LearningLongTermAsset.user_id == user_id)
    if asset_type:
        query = query.filter(LearningLongTermAsset.asset_type == _normalize_asset_type(asset_type))
    rows = (
        query.order_by(
            LearningLongTermAsset.asset_type.asc(),
            LearningLongTermAsset.weight.desc(),
            LearningLongTermAsset.updated_at.desc(),
        )
        .limit(max(1, min(limit, 200)))
        .all()
    )
    return [long_term_asset_to_dict(item) for item in rows]


def build_long_term_asset_snapshot(db: Session, user_id: int) -> dict[str, list[dict[str, Any]]]:
    snapshot = {asset_type: [] for asset_type in LONG_TERM_ASSET_TYPES}
    for item in list_long_term_assets(db, user_id, limit=200):
        snapshot[item["asset_type"]].append(item)
    return snapshot


def _persist_learning_long_term_assets(
    db: Session,
    *,
    user_id: int,
    query: str,
    answer: str,
    topics: list[str],
    scorecard: dict[str, Any],
    citations: list[dict[str, Any]] | None = None,
    source: str,
    task_id: str | None = None,
    kb_id: int | None = None,
    document_id: int | None = None,
    conversation_id: int | None = None,
) -> None:
    text = f"{query}\n{answer}"
    target_role = _infer_target_role(text)
    tech_stack = _detected_tech_stack(text)
    citation_payload = citations or []
    score_value = float(scorecard.get("total_score") or 0.0) / 100 if scorecard else None

    upsert_long_term_asset(
        db,
        user_id=user_id,
        asset_type="target_role",
        asset_key="primary",
        title=target_role,
        summary=f"当前长期目标岗位：{target_role}",
        payload={"target_role": target_role, "goal": query[:1000]},
        evidence={"source": source, "query": query[:500]},
        source=source,
        task_id=task_id,
        kb_id=kb_id,
        document_id=document_id,
        conversation_id=conversation_id,
    )
    if tech_stack:
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="tech_stack_profile",
            asset_key="current",
            title="技术栈画像",
            summary="、".join(tech_stack[:12]),
            payload={"technologies": tech_stack, "last_signal": text[:800]},
            evidence={"source": source, "citation_count": len(citation_payload)},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
        )
    if citation_payload or "项目" in text:
        project_key = f"document:{document_id}" if document_id is not None else "current"
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="project_experience",
            asset_key=project_key,
            title="项目经历库",
            summary=(answer or query)[:500],
            payload={"query": query[:500], "evidence_count": len(citation_payload)},
            evidence={"source": source, "citations": citation_payload[:5]},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
        )
    if topics:
        weakness_payload = {
            "topics": topics[:12],
            "scorecard": scorecard,
            "rubric": interview_scoring_rubric_payload(),
        }
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="weakness_map",
            asset_key="current",
            title="弱点地图",
            summary="；".join(topics[:6]),
            payload=weakness_payload,
            evidence={"source": source, "citation_count": len(citation_payload)},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
        )
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="recurring_error",
            asset_key="current",
            title="高频错误",
            summary="；".join(topics[:6]),
            payload={"errors": topics[:12], "signals": ["weakness", "insufficient_evidence"]},
            evidence={"source": source, "answer_excerpt": answer[:800]},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
        )
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="review_plan",
            asset_key="current",
            title="复习计划",
            summary="围绕最高优先级弱点进行 1-3 天滚动复习。",
            payload={"review_topics": topics[:6], "interval_days": [1, 2, 3]},
            evidence={"source": source},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
        )
    if answer:
        digest = hashlib.sha1(answer[:500].encode("utf-8")).hexdigest()[:16]
        answer_key = f"task:{task_id}" if task_id else f"conversation:{conversation_id}" if conversation_id else f"answer:{digest}"
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="interview_answer_version",
            asset_key=answer_key,
            title="面试回答版本",
            summary=answer[:500],
            payload={"answer": answer[:8000], "query": query[:1000], "scorecard": scorecard},
            evidence={"source": source, "citations": citation_payload[:5]},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
            trend_value=score_value,
        )
    if "模拟面试" in text or "mock interview" in text.lower():
        record_key = f"task:{task_id}" if task_id else f"conversation:{conversation_id}" if conversation_id else "latest"
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="mock_interview_record",
            asset_key=record_key,
            title="模拟面试记录",
            summary=answer[:500],
            payload={"query": query[:1000], "answer": answer[:8000], "topics": topics[:8]},
            evidence={"source": source, "scorecard": scorecard},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
            trend_value=score_value,
        )
    if score_value is not None:
        upsert_long_term_asset(
            db,
            user_id=user_id,
            asset_type="capability_trend",
            asset_key="readiness",
            title="能力趋势",
            summary=f"最近一次能力分：{round(score_value * 100, 1)}/100",
            payload={"readiness_score": score_value, "scorecard": scorecard, "topics": topics[:12]},
            evidence={"source": source, "citation_count": len(citation_payload)},
            source=source,
            task_id=task_id,
            kb_id=kb_id,
            document_id=document_id,
            conversation_id=conversation_id,
            trend_value=score_value,
        )


