"""多模态异步流水线的并发、顺序与超时测试。"""
import asyncio
import threading
import time


def test_upload_endpoint_returns_202():
    from app.api.document import router

    route = next(
        route
        for route in router.routes
        if getattr(route, "path", None) == "/document/upload"
    )
    assert route.status_code == 202


def test_legacy_doc_signature_is_supported():
    from app.api.document import _has_valid_signature

    ole_header = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 24

    assert _has_valid_signature(".doc", ole_header)
    assert not _has_valid_signature(".doc", b"PK\x03\x04" + b"\x00" * 28)


def test_legacy_doc_fallback_uses_antiword(tmp_path, monkeypatch):
    from app.rag.chunker import _fallback_text_reader

    doc_path = tmp_path / "legacy.doc"
    doc_path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1")

    class Result:
        stdout = "  legacy   content  "

    def fake_run(command, **kwargs):
        assert command == ["antiword", str(doc_path)]
        assert kwargs["capture_output"] is True
        return Result()

    monkeypatch.setattr("app.rag.chunker.subprocess.run", fake_run)

    assert _fallback_text_reader(str(doc_path)) == "legacy content"


def test_async_image_batch_preserves_order_and_deduplicates(tmp_path, monkeypatch):
    from app.rag.chunker import _ImageContext, _extract_images_concurrently_async

    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    third = tmp_path / "third.png"
    first.write_bytes(b"same-image")
    duplicate.write_bytes(b"same-image")
    third.write_bytes(b"other-image")
    calls = []

    async def fake_extract(path, _context):
        calls.append(path)
        await asyncio.sleep(0.01 if "first" in path else 0)
        return path

    monkeypatch.setattr("app.rag.chunker._extract_image_text_async", fake_extract)
    paths = [str(first), str(duplicate), str(third)]
    results = asyncio.run(_extract_images_concurrently_async(paths, _ImageContext()))

    assert results == [str(first), str(first), str(third)]
    assert calls == [str(first), str(third)]


def test_pdf_hash_dedup_runs_off_event_loop(tmp_path, monkeypatch):
    from app.rag import chunker

    first = tmp_path / "first.png"
    duplicate = tmp_path / "duplicate.png"
    other = tmp_path / "other.png"
    first.write_bytes(b"same")
    duplicate.write_bytes(b"same")
    other.write_bytes(b"other")
    events = []

    def slow_digest(path):
        import hashlib

        events.append("hash-start")
        time.sleep(0.03)
        events.append("hash-end")
        return hashlib.sha256(open(path, "rb").read()).hexdigest()

    monkeypatch.setattr(chunker, "_image_digest", slow_digest)
    records = [
        {"images": [(1, str(first)), (2, str(duplicate))]},
        {"images": [(1, str(other))]},
    ]

    async def run():
        task = asyncio.create_task(chunker._deduplicate_page_images_async(records))
        await asyncio.sleep(0.005)
        events.append("event-loop-tick")
        return await task

    digests = asyncio.run(run())

    assert events.index("event-loop-tick") < events.index("hash-end")
    assert len(digests) == 2
    assert records[0]["images"][0][1] == str(first)
    assert len(records[0]["images"]) == 1


def test_ocr_runs_off_event_loop_with_bounded_concurrency(tmp_path, monkeypatch):
    from app.rag.chunker import _ImageContext, _extract_images_concurrently_async

    paths = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        path.write_bytes(f"image-{index}".encode())
        paths.append(str(path))

    active = 0
    maximum = 0
    lock = threading.Lock()

    def fake_ocr(path):
        nonlocal active, maximum
        with lock:
            active += 1
            maximum = max(maximum, active)
        time.sleep(0.03)
        with lock:
            active -= 1
        return path

    monkeypatch.setattr("app.rag.chunker._ocr_image", fake_ocr)
    context = _ImageContext(captioner=None, ocr_semaphore=asyncio.Semaphore(2))
    results = asyncio.run(_extract_images_concurrently_async(paths, context))

    assert results == paths
    assert maximum == 2


def test_visual_calls_are_locally_bounded(tmp_path, monkeypatch):
    from app.rag.chunker import _ImageContext, _extract_images_concurrently_async

    paths = []
    for index in range(4):
        path = tmp_path / f"visual-{index}.png"
        path.write_bytes(f"visual-{index}".encode())
        paths.append(str(path))

    active = 0
    maximum = 0

    class FakeCaptioner:
        async def __call__(self, path):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            await asyncio.sleep(0.02)
            active -= 1
            return f"描述:{path}"

    monkeypatch.setattr("app.rag.chunker._ocr_image", lambda _path: "")
    context = _ImageContext(
        captioner=FakeCaptioner(),
        vision_semaphore=asyncio.Semaphore(1),
    )
    results = asyncio.run(_extract_images_concurrently_async(paths, context))

    assert len(results) == 4
    assert maximum == 1


def test_global_vision_slot_releases_lease(monkeypatch):
    from app.rag import distributed_limit

    released = []
    monkeypatch.setattr(distributed_limit, "_try_acquire", lambda _token: True)
    monkeypatch.setattr(distributed_limit, "_release", lambda token: released.append(token))

    async def use_slot():
        async with distributed_limit.vision_global_slot():
            return "ok"

    assert asyncio.run(use_slot()) == "ok"
    assert len(released) == 1


def test_document_lock_heartbeat_only_refreshes_owned_lock(monkeypatch):
    import app.queue as queue

    class FakeRedis:
        def __init__(self):
            self.args = None

        def eval(self, *args):
            self.args = args
            return 1

    client = FakeRedis()
    monkeypatch.setattr(queue, "_get_client", lambda: client)

    assert queue.refresh_doc_index_lock(7, "task-token") is True
    assert "lock:doc_index:7" in client.args
    assert "task-token" in client.args


def test_document_index_lock_busy_is_retried_not_acknowledged(monkeypatch):
    from app.services import document_index_service

    class FakeSession:
        def rollback(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(document_index_service, "SessionLocal", lambda: FakeSession())
    monkeypatch.setattr(document_index_service, "acquire_doc_index_lock", lambda *_args: False)

    try:
        document_index_service.run_document_index(7, 3, "task-token")
        raise AssertionError("expected lock busy")
    except document_index_service.DocumentIndexLockBusy as exc:
        assert "Document 7 is locked" in str(exc)


def test_worker_async_boundary_enforces_total_timeout(monkeypatch):
    from app.services import document_index_service

    async def slow_chunk(*_args, **_kwargs):
        await asyncio.sleep(0.05)
        return []

    monkeypatch.setattr("app.rag.chunker.chunk_document_async", slow_chunk)
    monkeypatch.setattr(
        document_index_service, "RAG_DOCUMENT_TIMEOUT_SECONDS", 0.01
    )

    try:
        document_index_service._run_document_chunking(1, object(), 420, 60)
        raise AssertionError("expected timeout")
    except TimeoutError as exc:
        assert "文档解析超过" in str(exc)
