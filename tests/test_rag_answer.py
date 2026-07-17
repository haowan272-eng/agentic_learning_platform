"""LangChain RAG回答与引用测试。"""
from unittest.mock import MagicMock


def _result():
    return {
        "chunk_id": 11,
        "document_id": 3,
        "kb_id": 2,
        "chunk_index": 0,
        "content": "退款申请应在购买后七天内提交。",
        "parent_content": "退款申请应在购买后七天内提交，并提供订单号。",
        "filename": "退款规则.pdf",
        "page_start": 4,
        "page_end": 4,
        "heading_path": "售后 > 退款",
        "source_type": "pdf",
        "location": "page:4",
        "score": 0.91,
    }


def test_build_evidence_uses_stable_source_numbers():
    from app.rag.answering import build_evidence

    context, citations = build_evidence([_result()])
    assert context.startswith("[1] 来源信息")
    assert citations[0].source_id == 1
    assert citations[0].filename == "退款规则.pdf"
    assert citations[0].quote == "退款申请应在购买后七天内提交。"


def test_invalid_model_citations_are_removed():
    from app.rag.answering import build_evidence, validate_answer_citations

    _, records = build_evidence([_result()])
    answer, used = validate_answer_citations("规则见[1]，伪造来源[99]。", records)
    assert answer == "规则见[1]，伪造来源。"
    assert [record.source_id for record in used] == [1]


def test_answer_without_results_does_not_call_llm(
    client, auth_user, monkeypatch, db_session
):
    _, headers = auth_user
    embedder = MagicMock()
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: embedder)
    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", lambda *args, **kwargs: [])
    answerer = MagicMock()
    monkeypatch.setattr("app.services.rag_service.get_rag_answerer", lambda: answerer)

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "不存在的问题", "top_k": 3},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["citations"] == []
    assert response.json()["conversation_id"] is not None
    answerer.answer.assert_not_called()


def test_answer_returns_deterministic_citation(client, auth_user, monkeypatch):
    _, headers = auth_user
    embedder = MagicMock()
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: embedder)
    monkeypatch.setattr(
        "app.rag.chain.Retriever.retrieve",
        lambda *args, **kwargs: [_result()],
    )
    answerer = MagicMock()
    answerer.answer.return_value = "应在七天内提交退款申请[1]。"
    monkeypatch.setattr("app.services.rag_service.get_rag_answerer", lambda: answerer)

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "多久内可以退款？", "top_k": 3, "save_history": False},
        headers=headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "应在七天内提交退款申请[1]。"
    assert data["citations"][0]["source_id"] == 1
    assert data["citations"][0]["page_start"] == 4
    assert data["citations"][0]["quote"] == "退款申请应在购买后七天内提交。"


def test_existing_conversation_rewrites_query_before_retrieval(
    client,
    auth_user,
    monkeypatch,
):
    _, headers = auth_user
    embedder = MagicMock()
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: embedder)
    monkeypatch.setattr("app.services.rag_service.RAG_QUERY_REWRITE_ENABLED", True)
    rewriter = MagicMock()
    rewriter.rewrite.return_value = "rewritten refund deadline query"
    monkeypatch.setattr("app.services.rag_service.get_query_rewriter", lambda: rewriter)
    captured = {}

    def retrieve(*args, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", retrieve)

    first = client.post(
        "/embedding/rag/answer",
        json={"query": "refund policy", "top_k": 3},
        headers=headers,
    )
    assert first.status_code == 200
    rewriter.rewrite.assert_not_called()

    second = client.post(
        "/embedding/rag/answer",
        json={
            "query": "what about the deadline?",
            "conversation_id": first.json()["conversation_id"],
            "top_k": 3,
        },
        headers=headers,
    )

    assert second.status_code == 200
    assert captured["query"] == "rewritten refund deadline query"
    assert second.json()["rewritten_query"] == "rewritten refund deadline query"
    rewriter.rewrite.assert_called_once()


def test_stream_answer_emits_tokens_and_validated_final(
    client, auth_user, db_session, monkeypatch
):
    _, headers = auth_user
    embedder = MagicMock()
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: embedder)
    monkeypatch.setattr(
        "app.rag.chain.Retriever.retrieve",
        lambda *args, **kwargs: [_result()],
    )
    answerer = MagicMock()
    answerer.stream.return_value = iter(["七天内", "提交[1]。"])
    monkeypatch.setattr("app.services.rag_service.get_rag_answerer", lambda: answerer)
    session_proxy = MagicMock(wraps=db_session)
    session_proxy.close = MagicMock()
    monkeypatch.setattr("app.services.rag_service.SessionLocal", lambda: session_proxy)

    response = client.post(
        "/embedding/rag/answer/stream",
        json={"query": "多久内可以退款？", "top_k": 3},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: token" in response.text
    assert '"delta":"七天内"' in response.text
    assert "event: final" in response.text
    assert '"answer":"七天内提交[1]。"' in response.text
    assert "event: done" in response.text


def test_memory_enhances_retrieval_and_llm_fallback_is_persisted(
    client, auth_user, db_session, monkeypatch
):
    from app.models import UserMemory

    user, headers = auth_user
    db_session.add(UserMemory(
        user_id=user.id,
        keyword="偏好中文回答",
        category="preference",
        weight=2.0,
    ))
    db_session.flush()
    embedder = MagicMock()
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: embedder)
    captured = {}

    def retrieve(*args, **kwargs):
        captured.update(kwargs)
        return [_result()]

    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", retrieve)
    answerer = MagicMock()
    answerer.answer.side_effect = TimeoutError("llm timeout")
    monkeypatch.setattr("app.services.rag_service.get_rag_answerer", lambda: answerer)

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "退款规则是什么？"},
        headers=headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["degraded"] is True
    assert data["memory_used"][0]["keyword"] == "偏好中文回答"
    assert "偏好中文回答" in captured["query"]
    assert data["citations"][0]["source_id"] == 1

    messages = client.get(
        f"/conversations/{data['conversation_id']}/messages",
        headers=headers,
    ).json()
    assert messages[-1]["degraded"] is True
    assert "偏好中文回答" in messages[-1]["memory_json"]


def test_answer_rejects_other_users_conversation(
    client, auth_user, auth_user2, monkeypatch
):
    _, headers_a = auth_user
    _, headers_b = auth_user2
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: MagicMock())
    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", lambda *args, **kwargs: [])
    conversation_id = client.post(
        "/embedding/rag/answer",
        json={"query": "私有问题"},
        headers=headers_a,
    ).json()["conversation_id"]

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "测试", "conversation_id": conversation_id},
        headers=headers_b,
    )
    assert response.status_code == 404


def test_use_memory_false_does_not_create_long_term_memory(
    client, auth_user, db_session, monkeypatch
):
    from app.models import RagMessage, UserMemory

    _, headers = auth_user
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: MagicMock())
    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", lambda *args, **kwargs: [])

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "我喜欢安静的地方", "use_memory": False},
        headers=headers,
    )

    assert response.status_code == 200
    assert db_session.query(UserMemory).count() == 0
    user_message = db_session.query(RagMessage).filter(RagMessage.role == "user").one()
    assert user_message.memory_extracted is True
