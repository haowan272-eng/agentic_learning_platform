from __future__ import annotations

import json

from app.models import (
    LearningAssessment,
    LearningLongTermAsset,
    LearningPractice,
    LearningProfile,
    LearningReviewItem,
    LearningWeakness,
)
from app.services.learning_service import (
    INTERVIEW_CAPABILITY_DIMENSIONS,
    INTERVIEW_SCORING_RUBRIC,
    build_interview_capability_query,
    create_assessment,
    is_interview_capability_request,
    record_rag_interview_diagnostic,
)


def test_interview_capability_request_detection_and_prompt_shape():
    query = "请基于我的简历和 JD 做面试能力诊断"

    assert is_interview_capability_request(query)
    prompt = build_interview_capability_query(query)

    for dimension in INTERVIEW_CAPABILITY_DIMENSIONS:
        assert dimension["title"] in prompt
        assert dimension["focus"] in prompt
    for rubric in INTERVIEW_SCORING_RUBRIC:
        assert rubric["title"] in prompt
        assert rubric["criterion"] in prompt
    assert "评分基线" in prompt
    assert "Weakness 标记" in prompt
    assert "第一次回答" in prompt


def test_rag_interview_diagnostic_records_learning_loop(db_session, auth_user):
    user, _headers = auth_user
    answer = """
## 知识理解
Redis 原理有项目证据，但缓存一致性对比不够清晰。

## 项目表达
背景和方案可见，但结果量化不足。

## 追问应对
性能问题和故障边界未明确。

## 系统设计
架构有描述，但扩展和可靠性证据不足。

## 沟通表达
表达结构需要补齐。

## 岗位匹配
JD 匹配点没有体现。
"""

    record_rag_interview_diagnostic(
        db_session,
        user_id=user.id,
        query="请基于我的简历和 JD 做面试能力诊断",
        answer=answer,
        citations=[{"source_id": 1, "document_id": 7, "quote": "Redis 项目经验"}],
        kb_id=3,
        document_id=7,
        conversation_id=11,
    )

    profile = db_session.query(LearningProfile).filter(LearningProfile.user_id == user.id).one()
    weaknesses = db_session.query(LearningWeakness).filter(LearningWeakness.user_id == user.id).all()
    practices = db_session.query(LearningPractice).filter(LearningPractice.user_id == user.id).all()
    reviews = db_session.query(LearningReviewItem).filter(LearningReviewItem.user_id == user.id).all()

    assert profile.current_level == "diagnosed"
    assert "知识理解" in (profile.diagnostic_summary or "")
    assert {item.category for item in weaknesses} >= {
        "knowledge_understanding",
        "project_storytelling",
        "followup_handling",
        "system_design",
        "communication",
        "jd_match",
    }
    assert len(practices) == len(weaknesses)
    assert len(reviews) == len(weaknesses)
    assert all(json.loads(practice.source_json or "{}").get("agent") == "rag_first_answer" for practice in practices)
    assert all(json.loads(practice.source_json or "{}").get("scoring_rubric") for practice in practices)
    assert all(json.loads(weakness.evidence_json or "{}").get("scorecard") for weakness in weaknesses)


def test_learning_assessment_backfills_explainable_rubric(db_session, auth_user):
    user, _headers = auth_user

    assessment = create_assessment(
        db_session,
        user.id,
        {
            "feedback": "回答结构不够清晰，项目证据不足。",
            "answer": "我做过 Redis 缓存。",
            "score": 0.42,
        },
    )

    row = db_session.query(LearningAssessment).filter(LearningAssessment.id == assessment.id).one()
    rubric = json.loads(row.rubric_json or "{}")
    assert rubric["source"] == "default_interview_scoring_rubric"
    assert {item["title"] for item in rubric["items"]} >= {item["title"] for item in INTERVIEW_SCORING_RUBRIC}
    assert rubric["total_score"] > 0


def test_rag_interview_diagnostic_persists_long_term_assets(db_session, auth_user):
    user, _headers = auth_user

    record_rag_interview_diagnostic(
        db_session,
        user_id=user.id,
        query="Backend interview diagnostic for a Redis RAG Agent project",
        answer="Redis RAG Agent project evidence is useful, but project storytelling and followup handling are weak.",
        citations=[{"document_id": 9, "quote": "Redis RAG Agent project"}],
        kb_id=2,
        document_id=9,
        conversation_id=12,
    )

    rows = db_session.query(LearningLongTermAsset).filter(LearningLongTermAsset.user_id == user.id).all()
    asset_types = {row.asset_type for row in rows}

    assert {
        "target_role",
        "tech_stack_profile",
        "project_experience",
        "weakness_map",
        "interview_answer_version",
        "review_plan",
        "capability_trend",
    } <= asset_types
    tech_stack = next(row for row in rows if row.asset_type == "tech_stack_profile")
    assert {"Redis", "RAG", "Agent"} <= set(json.loads(tech_stack.payload_json or "{}")["technologies"])
    trend = next(row for row in rows if row.asset_type == "capability_trend")
    assert trend.trend_value is not None


def test_learning_assets_api_allows_manual_upsert_and_snapshot(client, auth_user):
    _user, headers = auth_user
    payload = {
        "asset_type": "target_role",
        "asset_key": "primary",
        "title": "Backend Engineer",
        "summary": "Prepare for backend interviews.",
        "payload": {"target_role": "Backend Engineer"},
        "confidence": 0.9,
    }

    response = client.put("/learning/assets", json=payload, headers=headers)
    assert response.status_code == 200
    assert response.json()["asset_type"] == "target_role"
    assert response.json()["source"] == "user"

    response = client.get("/learning/assets?asset_type=target_role", headers=headers)
    assert response.status_code == 200
    assert response.json()[0]["title"] == "Backend Engineer"

    response = client.get("/learning/assets/snapshot", headers=headers)
    assert response.status_code == 200
    assert response.json()["target_role"][0]["payload"]["target_role"] == "Backend Engineer"
