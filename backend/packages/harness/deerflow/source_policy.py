from __future__ import annotations

from typing import Any, Literal


SourcePolicy = Literal["auto", "local_only"]

_LOCAL_ONLY_PHRASES = (
    "仅参考本地知识库",
    "只参考本地知识库",
    "仅使用本地知识库",
    "只使用本地知识库",
    "只查本地资料",
    "仅查本地资料",
    "only use local knowledge base",
    "local knowledge base only",
)


def resolve_source_policy(state: dict[str, Any]) -> SourcePolicy:
    """Prefer an explicit API policy, while preserving natural-language intent."""
    if state.get("source_policy") == "local_only":
        return "local_only"
    user_input = str(state.get("user_input") or "").lower()
    if any(phrase in user_input for phrase in _LOCAL_ONLY_PHRASES):
        return "local_only"
    return "auto"
