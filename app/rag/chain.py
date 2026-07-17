"""RAG 检索器。

BM25 + BGE + RRF + Cross-Encoder 混合检索，返回结果。
"""
from typing import Optional


class Retriever:
    """RAG 检索器：委托 Embedder.search() 执行混合检索。"""

    def __init__(self, embedder):
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[int] = None,
        user_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        personal_space_only: bool = False,
        bm25_weight: float = 0.4,
    ) -> list[dict]:
        """
        BM25 + BGE + RRF + Cross-Encoder 混合检索。

        embedder.search() 内部完成双路召回、RRF 融合和 Cross-Encoder 重排。
        """
        return self.embedder.search(
            query=query,
            top_k=top_k,
            document_id=document_id,
            user_id=user_id,
            kb_id=kb_id,
            personal_space_only=personal_space_only,
            bm25_weight=bm25_weight,
        )
