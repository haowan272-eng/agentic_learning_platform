"""从对话消息中提取用户长期记忆关键词。使用 LLM 提取，兼容任意领域。"""
import json
import logging
import re
from collections import Counter

from app.core.config import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_MODEL,
    RAG_MEMORY_EXTRACTION_MESSAGE_HEAD_CHARS,
    RAG_MEMORY_EXTRACTION_MESSAGE_TAIL_CHARS,
)

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """从用户消息中提取可作为长期记忆的关键词，返回 JSON 数组。
每条记忆包含：
- keyword: 关键词（不超过20字）
- category: 分类，取 project / topic / preference / constraint / other 之一

只提取对后续对话有帮助的持久信息：项目名、关注的技术/领域、约束条件、偏好等。
不要提取临时命令、一次性日志、报错堆栈、无意义数字或短确认词。
如果消息中没有任何值得长期记忆的信息，返回空数组 []。

用户消息：
{content}

只输出 JSON 数组，不要其他文字。"""


def _memory_slice(text: str) -> str:
    text = text or ""
    head = text[:RAG_MEMORY_EXTRACTION_MESSAGE_HEAD_CHARS]
    tail = text[-RAG_MEMORY_EXTRACTION_MESSAGE_TAIL_CHARS:] if RAG_MEMORY_EXTRACTION_MESSAGE_TAIL_CHARS else ""
    if len(text) <= len(head) + len(tail):
        return text
    return f"{head}\n...\n{tail}"


def _llm_extract(content: str) -> list[dict]:
    """调用 LLM 从单条消息提取关键词，失败则返回空列表。"""
    import urllib.request

    prompt = _EXTRACTION_PROMPT.format(content=content[:4000])
    payload = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 500,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
            },
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            raw = body["choices"][0]["message"]["content"].strip()
            return _parse_llm_response(raw)
    except Exception as exc:
        logger.warning("LLM 记忆提取失败: %s", exc)
        return []


def _parse_llm_response(raw: str) -> list[dict]:
    """解析 LLM 返回的 JSON，容错处理。"""
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [
                {"keyword": str(item.get("keyword", ""))[:128],
                 "category": str(item.get("category", "other"))[:32]}
                for item in items
                if isinstance(item, dict) and item.get("keyword")
            ]
    except json.JSONDecodeError:
        logger.warning("LLM 记忆提取返回非 JSON: %.200s", raw)
    return []


def extract_from_message(user_role_text: str) -> list[dict]:
    """从单条用户消息提取关键词。"""
    if not user_role_text or not DEEPSEEK_API_KEY:
        return []
    return _llm_extract(_memory_slice(user_role_text))


def extract_from_conversation(messages: list[dict]) -> list[dict]:
    """从整个对话的用户消息中批量提取，按频次排。"""
    user_texts = [msg.get("content", "") for msg in messages if msg.get("role") == "user"]
    if not user_texts:
        return []
    combined = "\n---\n".join(_memory_slice(text) for text in user_texts[-10:])
    return _deduplicate(_llm_extract(combined))


def _deduplicate(kw_list: list[dict]) -> list[dict]:
    counter: Counter = Counter()
    exemplar: dict[str, dict] = {}
    for item in kw_list:
        key = (item["keyword"], item["category"])
        counter[key] += 1
        exemplar[key] = item
    return [
        {**exemplar[key], "weight": round(float(cnt), 2)}
        for key, cnt in counter.most_common(50)
    ]


__all__ = ["extract_from_message", "extract_from_conversation"]
