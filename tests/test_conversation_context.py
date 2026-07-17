"""长对话压缩与结构化任务笔记测试。"""


def test_long_conversation_is_compacted_and_keeps_recent_messages(
    db_session, auth_user, monkeypatch
):
    from app.models import RagConversation, RagMessage
    from app.services import conversation_context as context_module

    user, _ = auth_user
    conversation = RagConversation(user_id=user.id, title="长任务")
    db_session.add(conversation)
    db_session.flush()
    for index in range(6):
        db_session.add(RagMessage(
            conversation_id=conversation.id,
            role="user" if index % 2 == 0 else "assistant",
            content=f"消息{index}",
        ))
    db_session.flush()
    monkeypatch.setattr(context_module, "RAG_COMPACTION_THRESHOLD", 4)
    monkeypatch.setattr(context_module, "RAG_HISTORY_MESSAGES", 2)

    result = context_module.build_conversation_context(db_session, conversation)

    assert result.compacted is True
    assert "消息0" in conversation.summary
    assert "消息4" in result.history
    assert "消息5" in result.history
    assert "消息0" not in result.history.split("【最近对话要点】")[-1]
    assert conversation.summary_until_message_id is not None
    assert "消息0" in result.task_state


def test_context_compaction_is_incremental(db_session, auth_user, monkeypatch):
    from app.models import RagConversation, RagMessage
    from app.services import conversation_context as context_module

    user, _ = auth_user
    conversation = RagConversation(user_id=user.id, title="增量压缩")
    db_session.add(conversation)
    db_session.flush()
    monkeypatch.setattr(context_module, "RAG_COMPACTION_THRESHOLD", 3)
    monkeypatch.setattr(context_module, "RAG_HISTORY_MESSAGES", 1)
    for index in range(3):
        db_session.add(RagMessage(
            conversation_id=conversation.id,
            role="user",
            content=f"阶段{index}",
        ))
    db_session.flush()
    first = context_module.build_conversation_context(db_session, conversation)
    checkpoint = conversation.summary_until_message_id
    second = context_module.build_conversation_context(db_session, conversation)

    assert first.compacted is True
    assert second.compacted is False
    assert conversation.summary_until_message_id == checkpoint


def test_compaction_trigger_counts_raw_message_tokens(db_session, auth_user, monkeypatch):
    from app.models import RagConversation, RagMessage
    from app.services import conversation_context as context_module

    user, _ = auth_user
    conversation = RagConversation(user_id=user.id, title="原文 token 触发")
    db_session.add(conversation)
    db_session.flush()
    db_session.add(RagMessage(
        conversation_id=conversation.id,
        role="assistant",
        content="长回答" + ("内容" * 600),
    ))
    db_session.flush()
    monkeypatch.setattr(context_module, "RAG_COMPACTION_THRESHOLD", 99)
    monkeypatch.setattr(context_module, "RAG_HISTORY_MESSAGES", 1)
    monkeypatch.setattr(context_module, "RAG_MEMORY_MAX_TOKENS", 800)

    result = context_module.build_conversation_context(db_session, conversation)

    assert result.compacted is True
    assert conversation.summary_until_message_id is not None
