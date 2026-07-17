"""长对话上下文压缩、结构化笔记与最近消息窗口"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re

from sqlalchemy.orm import Session

from app.core.config import (
    RAG_COMPACTION_THRESHOLD,
    RAG_HISTORY_MESSAGES,
    RAG_MEMORY_MAX_TOKENS,
    RAG_MEMORY_RECENT_MAX_TOKENS,
    RAG_MEMORY_SUMMARY_MAX_TOKENS,
    RAG_MEMORY_TASK_STATE_MAX_TOKENS,
)
from app.models import RagConversation, RagMessage
from app.services.short_term_memory import load_short_term_messages
from app.services.token_budget import count_tokens, fit_recent_lines, truncate_tokens


SUMMARY_SECTION_BUDGETS = {
    "constraints": 240,
    "decisions": 260,
    "config_params": 260,
    "paths": 160,
    "project_context": 160,
    "run_results": 200,
    "todos": 160,
    "raw_fallback": 120,
}

SUMMARY_SECTION_TITLES = {
    "project_context": "项目上下文",
    "paths": "关键路径",
    "constraints": "约束/禁忌",
    "decisions": "已确认决策",
    "config_params": "参数配置",
    "run_results": "运行结果",
    "todos": "待办事项",
    "raw_fallback": "保底原文",
}

PROJECT_KEYWORDS = (
    "RAG",
    "RAGAS",
    "铜川",
    "可研",
    "参考文献",
    "Docker",
    "VLM",
    "OCR",
    "记忆",
    "query rewrite",
    "rerank",
    "BM25",
    "Dense Retrieval",
    "向量",
    "检索",
)

SHORT_COMMANDS = {
    "执行",
    "继续",
    "好",
    "好的",
    "可以",
    "行",
    "嗯",
    "不要",
    "修改",
}

PATH_RE = re.compile(
    r"(?:[A-Za-z]:\\[^\s，。；;：:'\"`]+|(?:evaluation|scripts|app|tests)\\[^\s，。；;：:'\"`]+|(?:evaluation|scripts|app|tests)/[^\s，。；;：:'\"`]+)"
)
ENV_RE = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*[^\s，。；;]+")
METRIC_RE = re.compile(
    r"\b(answer_relevancy|llm_context_precision_with_reference|context_precision|context_recall|faithfulness)\b\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)"
)
PARAM_RE = re.compile(
    r"(top[-_ ]?k\s*(?:=|为|设置为)\s*\d+|rerank(?: candidates)?\s*(?:=|为|设置为)?\s*\d+|父块[^，。；;\n]{0,40}|前后各\s*\d+\s*个|命中子块\s*\+\s*前\s*\d+\s*个\s*\+\s*后\s*\d+\s*个)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ConversationContext:
    history: str
    task_state: str
    compacted: bool


def _value(message, name: str):
    return message.get(name) if isinstance(message, dict) else getattr(message, name)


def _role_line(message, max_tokens: int = 400) -> str:
    labels = {"user": "用户", "assistant": "助手", "system": "系统"}
    role = _value(message, "role")
    content = " ".join((_value(message, "content") or "").split())
    content = truncate_tokens(content, max_tokens, keep="end")
    return f"- {labels.get(role, role)}：{content}"


def _raw_role_line(message) -> str:
    labels = {"user": "用户", "assistant": "助手", "system": "系统"}
    role = _value(message, "role")
    content = " ".join((_value(message, "content") or "").split())
    return f"- {labels.get(role, role)}：{content}"


def _clip(text: str, max_tokens: int, keep: str = "end") -> str:
    return truncate_tokens(" ".join((text or "").split()), max_tokens, keep=keep)


def _message_text(message, max_tokens: int = 220) -> str:
    content = " ".join((_value(message, "content") or "").split())
    return truncate_tokens(content, max_tokens, keep="end")


def _add_unique(items: list[str], item: str, max_items: int = 12) -> None:
    item = " ".join((item or "").split()).strip(" -")
    if not item or item in items:
        return
    items.append(item)
    if len(items) > max_items:
        del items[0 : len(items) - max_items]


def _is_short_command(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    return compact in SHORT_COMMANDS or len(compact) <= 2


def _section_lines(items: list[str], max_tokens: int) -> list[str]:
    selected: list[str] = []
    used = 0
    for item in reversed(items):
        line = f"- {item}"
        line_tokens = count_tokens(line)
        if selected and used + line_tokens > max_tokens:
            break
        if line_tokens > max_tokens:
            selected.append("- " + truncate_tokens(item, max_tokens, keep="end"))
            break
        selected.append(line)
        used += line_tokens
    return list(reversed(selected))


def _extract_summary_sections(messages: list) -> dict[str, list[str]]:
    sections = {key: [] for key in SUMMARY_SECTION_BUDGETS}
    raw_candidates: list[str] = []
    latest_metrics: dict[str, str] = {}
    latest_configs: dict[str, str] = {}

    for message in messages:
        role = _value(message, "role")
        text = _message_text(message, max_tokens=260)
        if not text:
            continue

        for path in PATH_RE.findall(text):
            _add_unique(sections["paths"], path, max_items=10)

        for match in ENV_RE.findall(text):
            key = match.split("=", 1)[0].strip()
            latest_configs[key] = match.strip()

        for match in PARAM_RE.findall(text):
            _add_unique(sections["config_params"], match.strip(), max_items=12)

        for metric, value in METRIC_RE.findall(text):
            latest_metrics[metric] = value

        if any(keyword.lower() in text.lower() for keyword in PROJECT_KEYWORDS):
            _add_unique(
                sections["project_context"],
                truncate_tokens(text, 90, keep="end"),
                max_items=8,
            )

        if role == "user":
            if any(word in text for word in ("不可以", "不能", "不要", "禁止", "别", "不准")):
                _add_unique(sections["constraints"], text, max_items=10)
            if any(word in text for word in ("改成", "设置为", "就", "按", "用", "换成", "打开", "关闭")):
                if not _is_short_command(text):
                    _add_unique(sections["decisions"], text, max_items=10)
            if text.startswith(("帮我", "你先", "现在", "继续", "先")) and not _is_short_command(text):
                _add_unique(sections["todos"], text, max_items=8)

        if not _is_short_command(text):
            label = "用户" if role == "user" else "助手" if role == "assistant" else role
            raw_candidates.append(f"[{label}] {truncate_tokens(text, 120, keep='end')}")

    for key, value in latest_configs.items():
        _add_unique(sections["config_params"], value, max_items=12)
    if latest_metrics:
        metrics_text = "，".join(f"{key}={value}" for key, value in latest_metrics.items())
        _add_unique(sections["run_results"], f"最近 RAGAS/评测指标：{metrics_text}", max_items=6)

    for raw in raw_candidates[-4:]:
        _add_unique(sections["raw_fallback"], raw, max_items=4)

    return sections


def _render_summary_sections(sections: dict[str, list[str]]) -> str:
    ordered_keys = (
        "project_context",
        "paths",
        "constraints",
        "decisions",
        "config_params",
        "run_results",
        "todos",
        "raw_fallback",
    )
    rendered: list[str] = []
    for key in ordered_keys:
        lines = _section_lines(sections.get(key, []), SUMMARY_SECTION_BUDGETS[key])
        if lines:
            rendered.append(f"【{SUMMARY_SECTION_TITLES[key]}】\n" + "\n".join(lines))
    return "\n\n".join(rendered)


def _append_with_budget(parts: list[str], text: str, max_tokens: int) -> list[str]:
    if not text:
        return parts
    candidate = parts + [text]
    if count_tokens("\n\n".join(candidate)) <= max_tokens:
        return candidate
    remaining = max_tokens - count_tokens("\n\n".join(parts))
    if remaining <= 0:
        return parts
    clipped = truncate_tokens(text, remaining, keep="end")
    return parts + ([clipped] if clipped else [])


def _task_state(messages: list[RagMessage]) -> dict:
    user_messages = [message for message in messages if message.role == "user"]
    if not user_messages:
        return {"goal": "", "recent_requests": []}
    return {
        "goal": _clip(user_messages[0].content, 120, keep="start"),
        "recent_requests": [
            _clip(message.content, 120, keep="end")
            for message in user_messages[-3:]
        ],
    }


def _merge_summary(previous: str, messages: list[RagMessage]) -> str:
    sections = []
    if previous:
        previous_budget = max(200, RAG_MEMORY_SUMMARY_MAX_TOKENS // 4)
        sections.append(truncate_tokens(previous.strip(), previous_budget, keep="end"))
    if messages:
        extracted = _render_summary_sections(_extract_summary_sections(messages))
        if extracted:
            sections.append(extracted)
        else:
            sections.append("【保底原文】\n" + "\n".join(_role_line(row) for row in messages[-4:]))
    bounded_sections: list[str] = []
    for section in sections:
        bounded_sections = _append_with_budget(
            bounded_sections,
            section,
            RAG_MEMORY_SUMMARY_MAX_TOKENS,
        )
    combined = "\n\n".join(bounded_sections)
    # Keep the most recent compacted notes; full messages remain in PostgreSQL.
    return truncate_tokens(combined, RAG_MEMORY_SUMMARY_MAX_TOKENS, keep="end")


def _bounded_history(summary: str | None, recent: list[dict]) -> str:
    summary_part = ""
    if summary:
        summary_text = truncate_tokens(summary, RAG_MEMORY_SUMMARY_MAX_TOKENS, keep="end")
        summary_part = "【历史压缩摘要】\n" + summary_text

    recent_part = ""
    if recent:
        recent_notes = _render_summary_sections(_extract_summary_sections(recent))
        if recent_notes:
            recent_part = "【最近对话要点】\n" + truncate_tokens(
                recent_notes,
                RAG_MEMORY_RECENT_MAX_TOKENS,
                keep="end",
            )
        else:
            recent_lines = [_role_line(row) for row in recent]
            recent_lines = fit_recent_lines(recent_lines, RAG_MEMORY_RECENT_MAX_TOKENS)
            if recent_lines:
                recent_part = "【最近原始消息】\n" + "\n".join(recent_lines)

    parts = [part for part in (summary_part, recent_part) if part]
    history = "\n\n".join(parts) or "（无历史对话）"
    return truncate_tokens(history, RAG_MEMORY_MAX_TOKENS, keep="end")


def _task_text(state: dict) -> str:
    text = (
        f"目标：{state['goal'] or '未明确'}\n"
        + "最近请求：\n"
        + "\n".join(f"- {item}" for item in state["recent_requests"])
    )
    return truncate_tokens(text, RAG_MEMORY_TASK_STATE_MAX_TOKENS, keep="end")


def build_conversation_context(
    db: Session,
    conversation: RagConversation | None,
) -> ConversationContext:
    if conversation is None:
        return ConversationContext("（无历史对话）", "（无任务状态）", False)

    uncompressed = (
        db.query(RagMessage)
        .filter(
            RagMessage.conversation_id == conversation.id,
            RagMessage.id > (conversation.summary_until_message_id or 0),
        )
        .order_by(RagMessage.id)
        .all()
    )
    state_messages = list(uncompressed)
    compacted = False
    uncompressed_text = "\n".join(_raw_role_line(row) for row in uncompressed)
    uncompressed_tokens = count_tokens(uncompressed_text)
    should_compact = (
        len(uncompressed) >= RAG_COMPACTION_THRESHOLD
        or uncompressed_tokens > RAG_MEMORY_MAX_TOKENS
        or len(uncompressed_text) > RAG_MEMORY_MAX_TOKENS
    )
    if should_compact:
        compact_count = max(0, len(uncompressed) - RAG_HISTORY_MESSAGES)
        if (
            compact_count == 0
            and (
                uncompressed_tokens > RAG_MEMORY_MAX_TOKENS
                or len(uncompressed_text) > RAG_MEMORY_MAX_TOKENS
            )
        ):
            compact_count = 1
        rows_to_compact = uncompressed[:compact_count]
        if rows_to_compact:
            conversation.summary = _merge_summary(conversation.summary or "", rows_to_compact)
            conversation.summary_until_message_id = rows_to_compact[-1].id
            uncompressed = uncompressed[compact_count:]
            compacted = True

    recent = []
    if RAG_HISTORY_MESSAGES:
        recent = load_short_term_messages(
            db,
            conversation.user_id,
            conversation.id,
            limit=RAG_HISTORY_MESSAGES,
        )
        checkpoint = conversation.summary_until_message_id or 0
        recent = [row for row in recent if row["message_id"] > checkpoint]
    state = _task_state(state_messages)
    try:
        previous_state = json.loads(conversation.task_state_json or "{}")
    except (TypeError, json.JSONDecodeError):
        previous_state = {}
    if previous_state.get("goal"):
        state["goal"] = previous_state["goal"]
    conversation.task_state_json = json.dumps(state, ensure_ascii=False)

    return ConversationContext(_bounded_history(conversation.summary, recent), _task_text(state), compacted)
