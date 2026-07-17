"""Agent memory gateway.

This package turns the original RAG memory ideas into Agent Runtime context:
Redis hot windows, PostgreSQL event sourcing, long-term learner profile, and
session summaries.
"""

from .context import build_agent_context
from .consolidator import consolidate_memory_events
from .events import write_memory_event
from .profile import load_user_profile
from .service import (
    build_context_for_state,
    consolidate_task_memory,
    read_profile_for_user,
    summarize_task_session,
)
from .summarizer import summarize_session

__all__ = [
    "build_agent_context",
    "build_context_for_state",
    "consolidate_memory_events",
    "consolidate_task_memory",
    "load_user_profile",
    "read_profile_for_user",
    "summarize_session",
    "summarize_task_session",
    "write_memory_event",
]
