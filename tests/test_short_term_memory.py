"""Redis短期记忆、PostgreSQL回源与长期记忆检查点测试。"""
import json

from redis.exceptions import ConnectionError


class FakeRedis:
    def __init__(self):
        self.data = {}
        self.ttls = {}

    def pipeline(self, transaction=True):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def execute(self):
        return True

    def rpush(self, key, *values):
        self.data.setdefault(key, []).extend(values)
        return len(self.data[key])

    def ltrim(self, key, start, end):
        values = self.data.get(key, [])
        start = len(values) + start if start < 0 else start
        end = len(values) + end if end < 0 else end
        self.data[key] = values[max(0, start):end + 1]

    def lrange(self, key, start, end):
        values = self.data.get(key, [])
        start = len(values) + start if start < 0 else start
        end = len(values) + end if end < 0 else end
        return values[max(0, start):end + 1]

    def expire(self, key, ttl):
        self.ttls[key] = ttl
        return True

    def delete(self, key):
        self.data.pop(key, None)
        return 1


class BrokenRedis(FakeRedis):
    def lrange(self, key, start, end):
        raise ConnectionError("redis unavailable")


def _conversation_with_messages(db, user_id, contents):
    from app.models import RagConversation, RagMessage

    conversation = RagConversation(user_id=user_id, title="短期记忆")
    db.add(conversation)
    db.flush()
    rows = []
    for role, content in contents:
        row = RagMessage(
            conversation_id=conversation.id,
            role=role,
            content=content,
            memory_extracted=role != "user",
        )
        db.add(row)
        rows.append(row)
    db.flush()
    return conversation, rows


def test_short_term_cache_hit_returns_recent_messages(db_session, auth_user, monkeypatch):
    from app.services import short_term_memory as service

    user, _ = auth_user
    conversation, rows = _conversation_with_messages(
        db_session,
        user.id,
        [("user", "问题"), ("assistant", "回答")],
    )
    redis = FakeRedis()
    monkeypatch.setattr(service, "get_redis", lambda: redis)
    for row in rows:
        service.append_short_term_message(
            user.id, conversation.id, row.id, row.role, row.content, row.created_at
        )

    messages = service.load_short_term_messages(db_session, user.id, conversation.id)

    assert [row["content"] for row in messages] == ["问题", "回答"]
    assert redis.ttls[service.short_term_key(user.id, conversation.id)] > 0


def test_cache_miss_rehydrates_from_postgres(db_session, auth_user, monkeypatch):
    from app.services import short_term_memory as service

    user, _ = auth_user
    conversation, _ = _conversation_with_messages(
        db_session,
        user.id,
        [("user", "数据库消息")],
    )
    redis = FakeRedis()
    monkeypatch.setattr(service, "get_redis", lambda: redis)

    messages = service.load_short_term_messages(db_session, user.id, conversation.id)

    assert messages[0]["content"] == "数据库消息"
    key = service.short_term_key(user.id, conversation.id)
    assert json.loads(redis.data[key][0])["content"] == "数据库消息"


def test_redis_failure_falls_back_to_postgres(db_session, auth_user, monkeypatch):
    from app.services import short_term_memory as service

    user, _ = auth_user
    conversation, _ = _conversation_with_messages(
        db_session,
        user.id,
        [("user", "可靠消息")],
    )
    monkeypatch.setattr(service, "get_redis", lambda: BrokenRedis())

    messages = service.load_short_term_messages(db_session, user.id, conversation.id)

    assert messages[0]["content"] == "可靠消息"


def test_short_term_keys_are_isolated_by_user():
    from app.services.short_term_memory import short_term_key

    assert short_term_key(1, 7) != short_term_key(2, 7)


def test_long_term_extraction_checkpoint_prevents_double_count(
    db_session, auth_user, monkeypatch
):
    from app.models import UserMemory
    from app.services import memory_service

    user, _ = auth_user
    conversation, rows = _conversation_with_messages(
        db_session,
        user.id,
        [("user", "我喜欢安静的地方")],
    )
    monkeypatch.setattr(
        memory_service,
        "load_short_term_messages",
        lambda *_args, **_kwargs: [{
            "message_id": rows[0].id,
            "role": "user",
            "content": "我喜欢安静的地方",
            "created_at": "",
        }],
    )
    monkeypatch.setattr(
        memory_service,
        "extract_from_conversation",
        lambda *_args, **_kwargs: [{
            "keyword": "安静的地方",
            "category": "preference",
            "weight": 1.0,
        }],
    )

    first = memory_service.remember_short_term_window(
        db_session, user.id, conversation.id
    )
    second = memory_service.remember_short_term_window(
        db_session, user.id, conversation.id
    )
    memory = db_session.query(UserMemory).filter(UserMemory.user_id == user.id).one()

    assert first
    assert second == []
    assert memory.weight == 1.0
    assert rows[0].memory_extracted is True
