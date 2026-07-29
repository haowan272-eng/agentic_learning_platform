from app.services.document_quality import (
    assess_document_quality,
    list_document_type_specs,
    normalize_document_type,
)


def test_document_type_specs_include_business_materials():
    codes = {spec["code"] for spec in list_document_type_specs()}

    assert {
        "resume",
        "jd",
        "project_review",
        "interview_notes",
        "interview_record",
        "company_question_bank",
        "code_project_readme",
        "learning_note",
        "offer_result",
        "technical_design",
    }.issubset(codes)
    assert normalize_document_type("项目复盘") == "project_review"


def test_project_review_quality_flags_missing_metrics_and_technical_details():
    report = assess_document_quality(
        """
        # 推荐系统项目复盘

        背景：业务希望改进推荐体验。
        目标：提升用户使用体验。
        方案：我负责相关模块开发。
        结果：已经上线。
        """,
        document_type="project_review",
        file_name="project.md",
    )

    issue_codes = {issue["code"] for issue in report["issues"]}
    assert report["type"] == "project_review"
    assert "missing_metrics" in issue_codes
    assert "lack_technical_details" in issue_codes
    assert report["quality_score"] < 75


def test_recheck_document_quality_persists_report(client, db_session, auth_user, factory):
    from app.models.document_quality import DocumentQualityReport
    from app.rag.chunk_models import DocumentChunk

    user, headers = auth_user
    document = factory.document(
        db_session,
        user.id,
        doc_type="project_review",
        title="订单系统项目复盘",
    )
    db_session.add(
        DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content=(
                "# 订单系统项目复盘\n\n"
                "## 背景\n订单链路在大促期间延迟高，库存扣减和通知写入互相阻塞，用户支付后经常等待确认。\n\n"
                "## 目标\n目标是降低接口耗时，提升峰值 QPS，并减少库存通知失败带来的人工处理成本。\n\n"
                "## 方案\n我负责 Redis 缓存、MySQL 索引和 Kafka 队列削峰，拆分同步链路和异步通知链路。\n\n"
                "## 技术细节\n核心实现包括 Redis 热点库存缓存、MySQL 组合索引、Kafka 消费幂等、事务一致性校验和限流保护。\n\n"
                "## 取舍\n没有直接扩大数据库规格，因为成本高且不能解决突发流量；选择队列削峰后再补偿校验。\n\n"
                "## 结果\n平均耗时下降 35%，峰值 QPS 提升 2 倍，库存通知失败率下降 60%，每周人工处理减少 4 小时。"
            ),
        )
    )
    db_session.commit()

    response = client.post(f"/document/{document.id}/quality/recheck", headers=headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["doc_type"] == "project_review"
    assert payload["quality_score"] >= 75
    issue_codes = {issue["code"] for issue in payload["report"]["issues"]}
    assert "missing_metrics" not in issue_codes
    assert "lack_technical_details" not in issue_codes
    assert db_session.query(DocumentQualityReport).filter_by(document_id=document.id).count() == 1
