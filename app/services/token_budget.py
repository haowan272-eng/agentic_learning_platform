"""Token budget helpers for prompt context assembly."""
from __future__ import annotations

from functools import lru_cache
import os
import re
from typing import Any


@lru_cache(maxsize=1)
def _transformers_tokenizer() -> Any | None:
    model = os.getenv("RAG_TOKENIZER_MODEL", "").strip()
    if not model:
        return None
    local_only = os.getenv("RAG_TOKENIZER_LOCAL_ONLY", "true").lower() != "false"
    try:
        from transformers import AutoTokenizer  # type: ignore

        return AutoTokenizer.from_pretrained(model, local_files_only=local_only)
    except Exception:
        return None


@lru_cache(maxsize=1)
def _tiktoken_encoder() -> Any | None:
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _encode(text: str) -> list[int] | None:
    tokenizer = _transformers_tokenizer()
    if tokenizer is not None:
        try:
            return list(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass
    encoder = _tiktoken_encoder()
    if encoder is not None:
        return encoder.encode(text)
    return None


def _decode(tokens: list[int]) -> str | None:
    tokenizer = _transformers_tokenizer()
    if tokenizer is not None:
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception:
            pass
    encoder = _tiktoken_encoder()
    if encoder is not None:
        return encoder.decode(tokens)
    return None


def count_tokens(text: str) -> int:
    """Return an approximate token count, using a local tokenizer when configured."""
    if not text:
        return 0
    tokens = _encode(text)
    if tokens is not None:
        return len(tokens)

    cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
    other = re.sub(r"[\u4e00-\u9fff]", " ", text)
    other_tokens = sum(
        max(1, (len(token) + 3) // 4)
        for token in re.findall(r"\w+|[^\w\s]", other)
    )
    return cjk + other_tokens


def truncate_tokens(text: str, max_tokens: int, keep: str = "end") -> str:
    """Trim text to a token budget while preserving either the start or end."""
    if max_tokens <= 0 or not text:
        return ""
    tokens = _encode(text)
    if tokens is not None:
        if len(tokens) <= max_tokens:
            return text
        selected = tokens[-max_tokens:] if keep == "end" else tokens[:max_tokens]
        decoded = _decode(selected)
        if decoded is not None:
            return decoded.strip()

    if count_tokens(text) <= max_tokens:
        return text
    ratio = max_tokens / max(1, count_tokens(text))
    char_budget = max(1, int(len(text) * ratio))
    trimmed = text[-char_budget:] if keep == "end" else text[:char_budget]
    while trimmed and count_tokens(trimmed) > max_tokens:
        shrink_by = max(1, len(trimmed) // 10)
        trimmed = trimmed[shrink_by:] if keep == "end" else trimmed[:-shrink_by]
    return trimmed.strip()


def fit_recent_lines(lines: list[str], max_tokens: int) -> list[str]:
    """Keep the newest lines that fit within the budget."""
    if max_tokens <= 0:
        return []
    selected: list[str] = []
    used = 0
    for line in reversed(lines):
        line_tokens = count_tokens(line)
        if selected and used + line_tokens > max_tokens:
            break
        if line_tokens > max_tokens:
            selected.append(truncate_tokens(line, max_tokens, keep="end"))
            break
        selected.append(line)
        used += line_tokens
    return list(reversed(selected))
