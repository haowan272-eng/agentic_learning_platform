"""Query rewriting for rewrite-retrieve-read RAG."""
from __future__ import annotations

from typing import Optional

from app.core.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    RAG_ANSWER_MAX_RETRIES,
    RAG_ANSWER_TIMEOUT_SECONDS,
    RAG_QUERY_REWRITE_MAX_CHARS,
    RAG_QUERY_REWRITE_MODEL,
    RAG_QUERY_REWRITE_TEMPERATURE,
)


class LangChainQueryRewriter:
    """Rewrite a contextual user question into a standalone retrieval query."""

    def __init__(self):
        self._chain = None

    def _build_chain(self):
        if not DEEPSEEK_API_KEY:
            raise RuntimeError("DEEPSEEK_API_KEY is not configured; cannot rewrite retrieval query")
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """你是 RAG 检索 Query 改写器，只负责把用户问题改写成更适合检索的中文查询。
目标：提升 BM25 关键词匹配和向量语义检索召回，但不能改变问题意图。

规则：
1. 只输出一条改写后的检索 query，不要回答问题，不要解释，不要列多条。
2. 保留原问题中的专有名词、数字、时间、地点、文件对象、技术术语和否定条件。
3. 对“它、这个、这些资料、该项目、上述方法”等指代，只能根据最近对话补全明确对象。
4. 当前问题已经独立清楚时，尽量原样保留，只补 2-6 个有助于检索的关键词或同义词。
5. 不要加入没有依据的新实体、新结论、新金额、新日期或推测性背景。
6. 不要把问题改写成答案，也不要扩展成研究计划、总结提纲或多跳推理链。
7. 如果无法确定如何补全或增强，原样返回用户问题。
8. 输出长度控制在 15-80 个中文字符；综合问题可略长，但必须仍是单条检索 query。
9. 优先包含：核心实体 + 关键技术词 + 用户要问的关系/属性/结果。""",
            ),
            (
                "human",
                """【私人长期记忆】
{memory}

【最近对话】
{history}

【当前任务状态】
{task_state}

【用户当前问题】
{question}

请只输出一条独立、清晰、保守、适合知识库检索的 Query。""",
            ),
        ])
        model = ChatOpenAI(
            model=RAG_QUERY_REWRITE_MODEL,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            temperature=RAG_QUERY_REWRITE_TEMPERATURE,
            max_tokens=256,
            timeout=RAG_ANSWER_TIMEOUT_SECONDS,
            max_retries=RAG_ANSWER_MAX_RETRIES,
            streaming=False,
        )
        self._chain = prompt | model | StrOutputParser()
        return self._chain

    @property
    def chain(self):
        return self._chain or self._build_chain()

    def rewrite(
        self,
        question: str,
        history: str,
        memory: str,
        task_state: str,
    ) -> str:
        rewritten = self.chain.invoke({
            "question": question,
            "history": history or "（无历史对话）",
            "memory": memory or "（无长期记忆）",
            "task_state": task_state or "（无任务状态）",
        }).strip()
        rewritten = _clean_rewritten_query(rewritten)
        return rewritten or question


def _clean_rewritten_query(value: str) -> str:
    text = " ".join((value or "").strip().split())
    if len(text) > RAG_QUERY_REWRITE_MAX_CHARS:
        return text[:RAG_QUERY_REWRITE_MAX_CHARS].strip()
    return text


_query_rewriter: Optional[LangChainQueryRewriter] = None


def get_query_rewriter() -> LangChainQueryRewriter:
    global _query_rewriter
    if _query_rewriter is None:
        _query_rewriter = LangChainQueryRewriter()
    return _query_rewriter
