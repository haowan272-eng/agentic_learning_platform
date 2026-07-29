from unittest.mock import MagicMock


def _detail(response):
    return response.json()["detail"]


def test_unknown_route_uses_error_envelope(client):
    response = client.get("/missing-route", headers={"X-Request-ID": "req-404"})

    assert response.status_code == 404
    detail = _detail(response)
    assert detail["code"] == "NOT_FOUND"
    assert detail["message"] == "Not Found"
    assert detail["request_id"] == "req-404"
    assert detail["retryable"] is False


def test_invalid_token_uses_auth_error_code(client):
    response = client.get(
        "/kb",
        headers={"Authorization": "Bearer invalid-token", "X-Request-ID": "req-auth"},
    )

    assert response.status_code == 401
    detail = _detail(response)
    assert detail["code"] == "AUTH_TOKEN_INVALID"
    assert detail["message"] == "Invalid token"
    assert detail["request_id"] == "req-auth"


def test_validation_error_uses_field_details(client, auth_user):
    _, headers = auth_user
    response = client.post(
        "/embedding/rag/answer",
        json={"query": "", "top_k": 999},
        headers=headers,
    )

    assert response.status_code == 422
    detail = _detail(response)
    assert detail["code"] == "VALIDATION_ERROR"
    assert detail["details"]["errors"]


def test_rag_retrieval_failure_uses_retryable_error_code(client, auth_user, monkeypatch):
    _, headers = auth_user
    monkeypatch.setattr("app.services.rag_service.get_embedder", MagicMock(side_effect=TimeoutError("qdrant timeout")))

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "refund policy", "top_k": 3},
        headers=headers,
    )

    assert response.status_code == 503
    detail = _detail(response)
    assert detail["code"] == "RAG_RETRIEVAL_UNAVAILABLE"
    assert detail["message"] == "Knowledge retrieval is temporarily unavailable"
    assert detail["retryable"] is True
    assert "qdrant timeout" in detail["details"]["error"]


def test_rag_generation_fallback_reports_warning(client, auth_user, monkeypatch):
    _, headers = auth_user
    result = {
        "chunk_id": 11,
        "document_id": 3,
        "kb_id": 2,
        "chunk_index": 0,
        "content": "Refund requests must be filed within seven days.",
        "parent_content": "Refund requests must be filed within seven days with an order id.",
        "filename": "refund-policy.pdf",
        "page_start": 4,
        "page_end": 4,
        "heading_path": "Support > Refunds",
        "source_type": "pdf",
        "location": "page:4",
        "score": 0.91,
    }
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: MagicMock())
    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", lambda *args, **kwargs: [result])
    answerer = MagicMock()
    answerer.answer.side_effect = TimeoutError("llm timeout")
    monkeypatch.setattr("app.services.rag_service.get_rag_answerer", lambda: answerer)

    response = client.post(
        "/embedding/rag/answer",
        json={"query": "When can I request a refund?"},
        headers=headers,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["degraded"] is True
    assert body["warnings"][0]["code"] == "RAG_GENERATION_DEGRADED"
