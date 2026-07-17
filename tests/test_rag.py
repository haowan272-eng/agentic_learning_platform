"""共享知识库核心回归测试。"""
from unittest.mock import patch

import pytest


def test_document_model_is_not_bound_to_conversation():
    from app.models import Document

    assert not hasattr(Document, "conversation_id")


def test_worker_is_document_only():
    import worker

    assert worker.celery_app.conf.task_acks_late is True
    assert worker.celery_app.conf.worker_prefetch_multiplier == 1
    assert not hasattr(worker, "process_chat_task")


def test_bm25_index_contains_documents_from_different_uploaders(
    db_session, auth_user, auth_user2, factory
):
    from app.rag.chunk_models import DocumentChunk
    from app.rag.embeddings import Embedder

    user_a, _ = auth_user
    user_b, _ = auth_user2
    doc_a = factory.document(db_session, user_a.id)
    doc_b = factory.document(db_session, user_b.id)
    db_session.add_all([
        DocumentChunk(document_id=doc_a.id, content="杭州西湖", chunk_index=0),
        DocumentChunk(document_id=doc_b.id, content="北京故宫", chunk_index=0),
    ])
    db_session.flush()

    embedder = Embedder("unused")
    assert embedder.build_bm25(db_session, user_id=None) == 2
    state = embedder._bm25_indexes[(None, None, None)]
    assert {item["document_id"] for item in state["chunks"]} == {doc_a.id, doc_b.id}


def test_bm25_cache_key_includes_kb_id(db_session, auth_user, factory):
    from app.rag.chunk_models import DocumentChunk
    from app.rag.embeddings import Embedder

    user, _ = auth_user
    kb = factory.knowledge_base(db_session, user.id)
    document = factory.document(db_session, user.id, kb_id=kb.id)
    db_session.add(DocumentChunk(document_id=document.id, content="共享资料", chunk_index=0))
    db_session.flush()

    embedder = Embedder("unused")
    embedder.build_bm25(db_session, user_id=None, kb_id=kb.id)
    assert (None, None, kb.id) in embedder._bm25_indexes


def test_chunk_metadata_does_not_store_parent_context():
    from app.rag.chunker import chunk_markdown

    chunks = chunk_markdown("# A\n\none\n\n two\n\n three", "md", chunk_tokens=2, overlap_tokens=0)

    assert chunks
    assert "parent_content" not in chunks[0].metadata()


def test_attach_parent_contexts_expands_neighbors(db_session, auth_user, factory):
    from app.rag.chunk_models import DocumentChunk
    from app.rag.parent_context import attach_parent_contexts

    user, _ = auth_user
    document = factory.document(db_session, user.id)
    db_session.add_all([
        DocumentChunk(document_id=document.id, content="before", chunk_index=0),
        DocumentChunk(document_id=document.id, content="hit", chunk_index=1),
        DocumentChunk(document_id=document.id, content="after", chunk_index=2),
    ])
    db_session.flush()

    results = [{
        "document_id": document.id,
        "chunk_index": 1,
        "content": "hit",
    }]

    attach_parent_contexts(db_session, results, radius=1)

    assert results[0]["parent_content"] == "before\n\nhit\n\nafter"


def test_attach_parent_contexts_does_not_cross_documents(db_session, auth_user, factory):
    from app.rag.chunk_models import DocumentChunk
    from app.rag.parent_context import attach_parent_contexts

    user, _ = auth_user
    doc_a = factory.document(db_session, user.id)
    doc_b = factory.document(db_session, user.id)
    db_session.add_all([
        DocumentChunk(document_id=doc_a.id, content="a-hit", chunk_index=0),
        DocumentChunk(document_id=doc_b.id, content="b-neighbor", chunk_index=1),
    ])
    db_session.flush()

    results = [{
        "document_id": doc_a.id,
        "chunk_index": 0,
        "content": "a-hit",
    }]

    attach_parent_contexts(db_session, results, radius=1)

    assert results[0]["parent_content"] == "a-hit"


def test_qdrant_search_can_filter_kb_without_uploader(mock_qdrant):
    from app.rag.vectorstore import QdrantStore

    store = QdrantStore(collection_name="test", dim=1024)
    store._client = mock_qdrant
    store.search([0.0] * 1024, kb_id=7, user_id=None)
    query_filter = mock_qdrant.query_points.call_args.kwargs["query_filter"]
    keys = {condition.key for condition in query_filter.must}
    assert keys == {"kb_id"}


def test_document_queue_dispatches_celery_task():
    from app.queue import enqueue_document_index_task

    with patch(
        "app.tasks.document_index.index_document_task.apply_async"
    ) as apply_async, patch("app.queue.store_doc_index_progress") as save_progress:
        apply_async.return_value.id = "task-1"
        task_id = enqueue_document_index_task(document_id=1, user_id=2, kb_id=3)

    assert task_id == "task-1"
    assert apply_async.call_args.kwargs["kwargs"] == {
        "document_id": 1,
        "user_id": 2,
        "kb_id": 3,
    }
    save_progress.assert_called_once_with(1, "queued", task_id="task-1", attempt=0)


def test_document_celery_task_uses_late_ack_and_worker_loss_redelivery():
    from app.tasks.document_index import index_document_task

    assert index_document_task.acks_late is True
    assert index_document_task.reject_on_worker_lost is True


def test_document_celery_task_retries_when_document_lock_is_busy(monkeypatch):
    from app.services.document_index_service import DocumentIndexLockBusy
    from app.tasks import document_index

    progress_calls = []

    monkeypatch.setattr(
        document_index,
        "run_document_index",
        lambda *_args: (_ for _ in ()).throw(DocumentIndexLockBusy("locked")),
    )
    monkeypatch.setattr(
        document_index,
        "store_doc_index_progress",
        lambda *args, **kwargs: progress_calls.append((args, kwargs)),
    )
    monkeypatch.setattr(
        document_index,
        "_update_failed_document",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not fail document")),
    )
    monkeypatch.setattr(
        document_index.index_document_task,
        "retry",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError(f"retry:{kwargs['countdown']}")),
    )

    document_index.index_document_task.push_request(id="task-1", retries=0)
    try:
        with pytest.raises(RuntimeError, match="retry:5"):
            document_index.index_document_task.run(document_id=7, user_id=3)
    finally:
        document_index.index_document_task.pop_request()

    assert progress_calls == [
        ((7, "waiting_lock", "locked"), {"task_id": "task-1", "attempt": 1})
    ]


def test_invalid_pdf_signature_is_rejected():
    from app.api.document import _has_valid_signature

    assert _has_valid_signature(".pdf", b"not-pdf") is False
    assert _has_valid_signature(".pdf", b"%PDF-1.7") is True


def test_qdrant_dimension_mismatch_fails_early(mock_qdrant):
    from app.rag.vectorstore import QdrantStore

    mock_qdrant.get_collection.return_value.config.params.vectors.size = 512
    store = QdrantStore(collection_name="old", dim=1024)
    store._client = mock_qdrant
    with pytest.raises(RuntimeError, match="向量维度为 512"):
        store.ensure_collection()
