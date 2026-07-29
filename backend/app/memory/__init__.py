"""Agent 记忆网关。

将 RAG 记忆理念转化为 Agent Runtime 上下文：
- Redis 热窗口：短期事件缓存
- PostgreSQL 事件溯源：MemoryEvent 持久化
- 长期学习者画像：UserMemory 聚合与衰减
- 会话摘要：SessionSummary 压缩归档
"""

from .context import build_agent_context
from .consolidator import consolidate_memory_events
from .events import write_memory_event
from .profile import load_user_profile
from .service import (
    build_context_for_state,
    consolidate_task_memory,
    summarize_task_session,
)
from .summarizer import summarize_session

__all__ = [
    "build_agent_context",
    "build_context_for_state",
    "consolidate_memory_events",
    "consolidate_task_memory",
    "load_user_profile",
    "summarize_session",
    "summarize_task_session",
    "write_memory_event",
]
