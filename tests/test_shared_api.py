"""共享语料API权限测试。"""
from unittest.mock import MagicMock, patch


def test_document_list_hides_other_users_personal_document(
    client, auth_user, auth_user2, db_session, factory
):
    _, headers_a = auth_user
    user_b, _ = auth_user2
    document = factory.document(db_session, user_b.id, original_file_name="team-guide.txt")

    response = client.get("/document/list", headers=headers_a)

    assert response.status_code == 200
    assert document.id not in {item["id"] for item in response.json()}


def test_document_list_contains_member_kb_document(
    client, auth_user, auth_user2, db_session, factory
):
    from app.models import KnowledgeBaseMember

    owner, _ = auth_user
    user_b, headers_b = auth_user2
    kb = factory.knowledge_base(db_session, owner.id)
    db_session.add(KnowledgeBaseMember(kb_id=kb.id, user_id=user_b.id, role="viewer"))
    document = factory.document(
        db_session,
        owner.id,
        kb_id=kb.id,
        original_file_name="team-guide.txt",
    )
    db_session.flush()

    response = client.get("/document/list", headers=headers_b)

    assert response.status_code == 200
    assert document.id in {item["id"] for item in response.json()}


def test_uploader_can_delete_unclassified_document(
    client, auth_user, db_session, factory
):
    user, headers = auth_user
    document = factory.document(db_session, user.id, source_retained=False, file_path=None)
    with patch("app.rag.vectorstore.get_qdrant_store") as get_store:
        get_store.return_value.delete_by_document_id.return_value = -1
        response = client.delete(f"/document/{document.id}", headers=headers)
    assert response.status_code == 200


def test_other_user_cannot_delete_unclassified_document(
    client, auth_user, auth_user2, db_session, factory
):
    user_a, _ = auth_user
    _, headers_b = auth_user2
    document = factory.document(db_session, user_a.id)
    response = client.delete(f"/document/{document.id}", headers=headers_b)
    assert response.status_code == 403


def test_public_search_side_route_is_removed(client, auth_user):
    _, headers = auth_user
    response = client.post(
        "/embedding/search",
        json={"query": "共享资料", "top_k": 5},
        headers=headers,
    )
    assert response.status_code == 404


def test_kb_is_hidden_from_non_member(
    client, auth_user, auth_user2, db_session, factory
):
    owner, _ = auth_user
    _, viewer_headers = auth_user2
    kb = factory.knowledge_base(db_session, owner.id)

    response = client.get("/kb", headers=viewer_headers)

    assert response.status_code == 200
    assert kb.id not in {item["id"] for item in response.json()}


def test_conversations_are_private_between_users(client, auth_user, auth_user2, monkeypatch):
    _, headers_a = auth_user
    _, headers_b = auth_user2
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: MagicMock())
    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", lambda *args, **kwargs: [])
    created = client.post(
        "/embedding/rag/answer",
        json={"query": "A的私有问题"},
        headers=headers_a,
    )
    assert created.status_code == 200
    conversation_id = created.json()["conversation_id"]

    response = client.get(
        f"/conversations/{conversation_id}/messages",
        headers=headers_b,
    )
    assert response.status_code == 404


def test_answer_route_is_the_only_conversation_writer(client, auth_user, monkeypatch):
    _, headers = auth_user
    monkeypatch.setattr("app.services.rag_service.get_embedder", lambda: MagicMock())
    monkeypatch.setattr("app.rag.chain.Retriever.retrieve", lambda *args, **kwargs: [])
    added = client.post(
        "/embedding/rag/answer",
        json={"query": "共享知识库里有什么？"},
        headers=headers,
    )
    assert added.status_code == 200
    conversation_id = added.json()["conversation_id"]
    assert client.post(
        f"/conversations/{conversation_id}/messages",
        json={"role": "user", "content": "绕过主链路"},
        headers=headers,
    ).status_code == 405
    messages = client.get(
        f"/conversations/{conversation_id}/messages",
        headers=headers,
    ).json()
    assert [item["role"] for item in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "共享知识库里有什么？"
