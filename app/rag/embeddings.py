"""Sentence-Transformers 封装：文本转向量、批量入库、相似度搜索。

命令行示例：
    python embedding/embedder.py                    # 为所有分块生成 embedding
    python embedding/embedder.py --document-id 1    # 为指定文档生成 embedding
    python embedding/embedder.py --search "查询文本" # 相似度搜索
"""

import json
import logging
import math
import os
import re
from typing import Optional

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import RAG_RERANK_CANDIDATES

from app.models.document import Document
from app.rag.chunk_models import DocumentChunk
from app.rag.rag_models import ChunkEmbedding

logger = logging.getLogger(__name__)

# 模型路径：本地优先，Docker 通过环境变量覆盖
_MODEL_ROOT = os.environ.get("MODEL_ROOT", os.path.expanduser("~/models"))
DEFAULT_MODEL = os.environ.get(
    "BGE_MODEL_PATH",
    os.path.join(_MODEL_ROOT, "bge-large-zh-v1.5"),
)



# ==================== Embedder ====================

class Embedder:
    """sentence-transformers 封装：编码、存储和检索。"""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None
        # BM25 indexes are isolated by user/document/kb.
        self._bm25_indexes: dict[tuple[Optional[int], Optional[int], Optional[int]], dict] = {}
        self._cross_encoder = None

    @property
    def model(self):
        """延迟加载模型，首次使用时下载。"""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading embedding model %s", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            logger.info("embedding model loaded; dim=%s", self.dim)
        return self._model

    @property
    def dim(self) -> int:
        return self.model.get_embedding_dimension()

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        将文本列表编码为归一化向量。

        返回：np.ndarray，shape 为 (len(texts), dim)，dtype 为 float32，已做 L2 归一化。
        """
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True)

    # ==================== BM25 索引管理 ====================

    @property
    def bm25_indexed(self) -> bool:
        """兼容旧调用：表示至少存在一个隔离后的 BM25 索引。"""
        return bool(self._bm25_indexes)

    @staticmethod
    def _bm25_key(
        user_id: Optional[int],
        document_id: Optional[int],
        kb_id: Optional[int] = None,
        personal_space_only: bool = False,
    ) -> tuple:
        if not personal_space_only:
            return user_id, document_id, kb_id
        return user_id, document_id, kb_id, personal_space_only

    def _chunk_query(
        self,
        db: Session,
        user_id: Optional[int],
        document_id: Optional[int],
        kb_id: Optional[int] = None,
        personal_space_only: bool = False,
    ):
        query = db.query(DocumentChunk)
        if user_id is not None or kb_id is not None or personal_space_only:
            query = query.join(Document, DocumentChunk.document_id == Document.id)
            if user_id is not None:
                query = query.filter(Document.user_id == user_id)
            if kb_id is not None:
                query = query.filter(Document.kb_id == kb_id)
            if personal_space_only:
                query = query.filter(Document.kb_id.is_(None))
        if document_id is not None:
            query = query.filter(DocumentChunk.document_id == document_id)
        return query

    def ensure_bm25(
        self,
        db: Session,
        user_id: Optional[int],
        document_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        personal_space_only: bool = False,
    ) -> int:
        """按用户、文档或知识库构建/刷新 BM25；用 count/max(id) 检测 Worker 产生的新分块。"""
        query = self._chunk_query(db, user_id, document_id, kb_id, personal_space_only)
        count, max_id = query.with_entities(func.count(DocumentChunk.id), func.max(DocumentChunk.id)).one()
        fingerprint = (int(count or 0), int(max_id or 0))
        state = self._bm25_indexes.get(self._bm25_key(user_id, document_id, kb_id, personal_space_only))
        if state is None or state["fingerprint"] != fingerprint:
            return self.build_bm25(
                db,
                user_id=user_id,
                document_id=document_id,
                kb_id=kb_id,
                personal_space_only=personal_space_only,
            )
        return fingerprint[0]

    def build_bm25(
        self,
        db: Session,
        user_id: Optional[int] = None,
        document_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        personal_space_only: bool = False,
    ) -> int:
        """构建 BM25 索引"""
        q = self._chunk_query(db, user_id, document_id, kb_id, personal_space_only)
        rows = q.order_by(DocumentChunk.id).all()
        doc_ids = {row.document_id for row in rows}
        docs = {
            doc.id: doc
            for doc in db.query(Document).filter(Document.id.in_(doc_ids)).all()
        } if doc_ids else {}
        def cached_chunk(row) -> dict:
            try:
                metadata = json.loads(row.metadata_json or "{}")
            except (TypeError, json.JSONDecodeError):
                metadata = {}
            doc = docs.get(row.document_id)
            return {
                "id": row.id,
                "document_id": row.document_id,
                "kb_id": doc.kb_id if doc else None,
                "content": row.content or "",
                "chunk_index": row.chunk_index,
                "modality": row.modality or "text",
                "page_start": row.page_start,
                "page_end": row.page_end,
                "filename": (doc.original_file_name or doc.file_name or "未知文档") if doc else "未知文档",
                "heading_path": metadata.get("heading_path"),
                "parent_content": metadata.get("parent_content"),
                "source_type": metadata.get("source_type"),
                "location": metadata.get("location"),
            }
        chunks = [
            cached_chunk(r)
            for r in rows
        ]
        scorer = BM25Scorer()
        scorer.index(chunks)
        self._bm25_indexes[self._bm25_key(user_id, document_id, kb_id, personal_space_only)] = {
            "scorer": scorer,
            "chunks": chunks,
            "fingerprint": (len(chunks), max((c["id"] for c in chunks), default=0)),
        }
        return len(chunks)

    def rebuild_bm25(
        self,
        db: Session,
        user_id: Optional[int] = None,
        document_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        personal_space_only: bool = False,
    ) -> int:
        """重建 BM25 索引（文档增删后调用）"""
        self._bm25_indexes.pop(self._bm25_key(user_id, document_id, kb_id, personal_space_only), None)
        return self.build_bm25(
            db,
            user_id=user_id,
            document_id=document_id,
            kb_id=kb_id,
            personal_space_only=personal_space_only,
        )

    # ==================== 批量生成 & 存储 ====================

    def generate_for_chunks(
        self,
        db: Session,
        document_id: Optional[int] = None,
        user_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        batch_size: int = 32,
        strict_vectorstore: bool = False,
        pipeline_hash: Optional[str] = None,
    ) -> int:
        """
        读取分块，编码并写入 chunk_embeddings 表。

        固定写入共享集合，并通过 upsert + stale 清理保证文档重建幂等。

        参数：
            db: 数据库会话。
            document_id: 若为 None，处理所有文档的全部分块。
            user_id: 可选，限定上传用户。
            kb_id: 可选，限定知识库。
            batch_size: 编码时的批次大小。
            strict_vectorstore: True 时 Qdrant 写入失败将抛出异常，使后台任务明确标记失败。
            pipeline_hash: 旧版本兼容参数，当前不会改变目标集合。

        返回：本次新增或更新的 embedding 数量。
        """
        # 1. 查询分块
        q = db.query(DocumentChunk)
        if user_id is not None or kb_id is not None:
            q = q.join(Document, DocumentChunk.document_id == Document.id)
            if user_id is not None:
                q = q.filter(Document.user_id == user_id)
            if kb_id is not None:
                q = q.filter(Document.kb_id == kb_id)
        if document_id is not None:
            q = q.filter(DocumentChunk.document_id == document_id)  # 指定文档

        chunks = q.order_by(DocumentChunk.id).all()

        if not chunks:
            logger.info("no chunks found for embedding")
            return 0

        logger.info("found %s chunks to embed", len(chunks))

        # 2. 提取文本
        # 标题路径、图片 OCR/描述等信息放入 embedding_content；旧数据自动回退到 content。
        texts = [c.embedding_content or c.content for c in chunks]

        # 3. 编码
        logger.info("encoding chunks")
        vectors = self.encode(texts, batch_size=batch_size)
        logger.info("encoded %s vectors; dim=%s", len(vectors), vectors.shape[1])
        from app.core.config import QDRANT_DIM
        if int(vectors.shape[1]) != QDRANT_DIM:
            raise RuntimeError(
                f"Embedding 模型输出 {vectors.shape[1]} 维，与 QDRANT_DIM={QDRANT_DIM} 不一致。"
                "请修改 .env 后重启 API 和 Worker。"
            )

        # 4. 记录元数据到 chunk_embeddings；只记录模型、维度和时间，不存向量字节。
        chunk_ids = [chunk.id for chunk in chunks]
        existing_by_chunk = {
            row.chunk_id: row
            for row in db.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id.in_(chunk_ids)).all()
        }
        vector_dim = int(vectors.shape[1])
        count = 0
        for chunk in chunks:
            existing = existing_by_chunk.get(chunk.id)
            if existing:
                existing.model_name = self.model_name
                existing.dim = vector_dim
            else:
                db.add(
                    ChunkEmbedding(
                        chunk_id=chunk.id,
                        dim=vector_dim,
                        model_name=self.model_name,
                    )
                )
            count += 1

        # 5. Sync Qdrant: upsert first, then remove stale points.
        try:
            from app.rag.vectorstore import get_qdrant_store

            qdrant = get_qdrant_store()

            # Query document metadata for Qdrant payload and source display.
            doc_ids = set(c.document_id for c in chunks)
            docs_map = {}
            for d in db.query(Document).filter(Document.id.in_(doc_ids)).all():
                docs_map[d.id] = {
                    "file_name": d.original_file_name or d.file_name or "未知文档",
                    "user_id": d.user_id if hasattr(d, "user_id") else None,
                    "kb_id": d.kb_id if hasattr(d, "kb_id") else None,
                }

            points = []
            for chunk, vec in zip(chunks, vectors):
                doc_meta = docs_map.get(chunk.document_id, {})
                try:
                    chunk_meta = json.loads(chunk.metadata_json or "{}")
                except (TypeError, json.JSONDecodeError):
                    chunk_meta = {}
                points.append({
                    "id": chunk.id,
                    "vector": vec.tolist(),
                    "payload": {
                        "chunk_id": chunk.id,
                        "document_id": chunk.document_id,
                        "user_id": doc_meta.get("user_id"),
                        "kb_id": doc_meta.get("kb_id"),
                        "filename": doc_meta.get("file_name", "未知文档"),
                        "chunk_index": chunk.chunk_index,
                        "content": chunk.content,
                        "modality": chunk.modality or "text",
                        "page_start": chunk.page_start,
                        "page_end": chunk.page_end,
                        "heading_path": chunk_meta.get("heading_path"),
                        "source_type": chunk_meta.get("source_type"),
                        "location": chunk_meta.get("location"),
                    },
                })

            qdrant.upsert_points(points)
            target_collection = qdrant.collection_name
            logger.info("qdrant upserted %s vectors to collection %s", len(points), target_collection)

            # 清理同一文档中本次未产出的过期分块；删除后重新分块可能导致 ID 变化。
            if document_id is not None:
                new_chunk_ids = {c.id for c in chunks}
                existing_ids = qdrant.list_point_ids(
                    document_id=document_id,
                    collection_name=target_collection,
                )
                stale_ids = existing_ids - new_chunk_ids
                if stale_ids:
                    qdrant.delete_points_by_ids(
                        list(stale_ids),
                        collection_name=target_collection,
                    )
                    logger.info("qdrant removed %s stale points for document %s", len(stale_ids), document_id)
        except ImportError as e:
            db.rollback()
            if strict_vectorstore:
                raise RuntimeError("qdrant-client 未安装，无法完成文档索引") from e
            logger.warning("qdrant-client not installed; skipping qdrant upsert")
            return 0
        except Exception as e:
            db.rollback()
            if strict_vectorstore:
                raise RuntimeError(f"Qdrant 写入失败: {e}") from e
            logger.warning("qdrant upsert failed", exc_info=True)
            return 0

        db.commit()
        logger.info("saved %s chunk embedding metadata records", count)

        return count

    # ==================== Cross-Encoder 重排 ====================

    _RERANKER_MODEL = os.environ.get(
        "RERANKER_MODEL_PATH",
        os.path.join(_MODEL_ROOT, "bge-reranker-v2-m3"),
    )

    def _load_reranker(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            logger.info("loading reranker %s", self._RERANKER_MODEL)
            self._cross_encoder = CrossEncoder(self._RERANKER_MODEL)
        return self._cross_encoder

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Cross-Encoder 逐对精排"""
        if len(candidates) <= top_k:
            return candidates
        try:
            model = self._load_reranker()
            pairs = [(query, c.get("content", "")) for c in candidates]
            scores = model.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["score"] = round(float(s), 4)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_k]
        except Exception as exc:
            logger.warning("rerank failed; using RRF order", exc_info=True)
            return candidates[:top_k]

    # ==================== 相似度搜索 ====================

    def search(
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
        BM25 + BGE + RRF + Cross-Encoder 返回 top_k 结果。

        bm25_weight: 0.0 表示偏语义，0.5 表示等权，1.0 表示偏关键词。
        """
        from app.rag.vectorstore import get_qdrant_store

        qdrant = get_qdrant_store()
        recall = top_k * 4

        # BM25 reads only the isolated user/document/kb index.
        bm25_state = self._bm25_indexes.get(self._bm25_key(user_id, document_id, kb_id, personal_space_only))
        bm25_ranked = bm25_state["scorer"].search(query, top_k=recall) if bm25_state else []

        # Dense retrieval; keep BM25 fallback when vector service fails.
        try:
            q_vec = self.encode([query])[0]
            dense_raw = qdrant.search(
                query_vector=q_vec.tolist(),
                top_k=recall,
                user_id=user_id,
                document_id=document_id,
                kb_id=kb_id,
                kb_id_is_null=personal_space_only,
            )
        except Exception as exc:
            if not bm25_state:
                raise
            logger.warning("dense search failed; using BM25 only", exc_info=True)
            dense_raw = []
        dense_ranked = [(r["chunk_id"], r.get("score", 0.0)) for r in dense_raw]

        # RRF
        effective_bm25_weight = bm25_weight if bm25_state else 0.0
        rrf = _rrf_fusion(bm25_ranked, dense_ranked, bm25_weight=effective_bm25_weight)

        # 收集候选。
        dense_map = {r["chunk_id"]: r for r in dense_raw}
        chunk_map = {c["id"]: c for c in (bm25_state["chunks"] if bm25_state else [])}
        candidates: list[dict] = []
        for chunk_id, _ in rrf[: max(top_k, RAG_RERANK_CANDIDATES)]:
            chunk = chunk_map.get(chunk_id)
            dense = dense_map.get(chunk_id, {})
            if chunk:
                candidates.append({
                    "chunk_id": chunk_id,
                    "document_id": chunk["document_id"],
                    "kb_id": dense.get("kb_id") if dense else chunk.get("kb_id"),
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    "modality": chunk.get("modality", "text"),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "score": round(dense.get("score", 0.0), 4),
                    "heading_path": dense.get("heading_path") or chunk.get("heading_path"),
                    "parent_content": dense.get("parent_content") or chunk.get("parent_content"),
                    "filename": dense.get("filename") or chunk.get("filename", ""),
                    "source_type": dense.get("source_type") or chunk.get("source_type"),
                    "location": dense.get("location") or chunk.get("location"),
                })
            else:
                candidates.append(dense)

        # Cross-Encoder 重排
        return self._rerank(query, candidates, top_k)


# ==================== 单例工厂 ====================

_embedder: Optional[Embedder] = None


def get_embedder() -> Embedder:
    """获取 Embedder 单例（模型只加载一次）"""
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


# ==================== 命令行入口 ====================

def main():
    import argparse

    from app.core.database import SessionLocal

    parser = argparse.ArgumentParser(description="Chunk 到 Embedding 工具")  # 1. 加载分块
    parser.add_argument(
        "--doc-id",
        type=int,
        default=None,
        help="只处理指定文档 ID 的分块",
    )
    parser.add_argument(
        "--search",
        type=str,
        default=None,
        help="搜索相似分块（不生成 embedding",
    )
    parser.add_argument( #
        "--top-k",
        type=int,
        default=5,
        help="搜索返回条数",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"模型名称 (默认: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    embedder = Embedder(model_name=args.model)

    if args.search:
        # 搜索模式：使用 Qdrant HNSW 索引。
        results = embedder.search(args.search, top_k=args.top_k)
        logger.info("search query: %s", args.search)
        for i, r in enumerate(results, 1):
            content = (r.get('content') or '')[:120]
            logger.info("result #%s score=%.4f chunk_id=%s doc_id=%s content=%s", i, r["score"], r["chunk_id"], r["document_id"], content)
    else:
        # 生成模式
        db = SessionLocal()
        try:
            embedder.generate_for_chunks(db, document_id=args.doc_id)
        finally:
            db.close()


# ==================== BM25 分词与索引 ====================

def _tokenize(text: str) -> list[str]:
    """
    CJK 单字切分，ASCII 按空白/标点切分，保留连字符复合词。

    实验室专有名词场景：CJK 单字切分可避免分词工具误拆冷门术语导致漏召回；
    ASCII 连字符复合词（SDS-PAGE、C-反应蛋白）整词保留。
    """
    if not text:
        return []
    text = text.lower().strip()
    # 保护连字符复合词
    text = text.replace('-', '@')
    tokens: list[str] = []
    for part in text.split():
        part = part.replace('@', '-')
        # CJK 单字切，非CJK整体保留
        for st in re.findall(r'[一-鿿㐀-䶿]|[^一-鿿㐀-䶿]+', part):
            st = st.strip()
            if st:
                tokens.append(st)
    return tokens


class BM25Scorer:
    """
    内存 BM25 索引。

    score(D,Q) = Σ IDF(qi) × f(qi,D)×(k1+1) / (f(qi,D) + k1×(1−b+b×|D|/avgDL))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: list[dict] = []
        self._doc_freq: dict[str, int] = {}
        self._doc_len: list[int] = []
        self._term_freqs: list[dict[str, int]] = []
        self._avgdl: float = 0.0
        self._indexed: bool = False

    def index(self, docs: list[dict]) -> None:
        self._docs = docs
        self._doc_freq.clear()
        self._doc_len.clear()
        self._term_freqs.clear()
        for doc in docs:
            tokens = _tokenize(doc.get("content", ""))
            tf: dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self._term_freqs.append(tf)
            self._doc_len.append(len(tokens))
            for t in set(tokens):
                self._doc_freq[t] = self._doc_freq.get(t, 0) + 1
        self._avgdl = sum(self._doc_len) / max(1, len(self._doc_len))
        self._indexed = True

    def search(self, query: str, top_k: int = 20) -> list[tuple[int, float]]:
        """返回 [(chunk_id, bm25_score), ...] 按分数降序"""
        if not self._indexed:
            return []
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []
        N = len(self._docs)
        results: list[tuple[int, float]] = []
        for idx, tf in enumerate(self._term_freqs):
            score = 0.0
            dl = self._doc_len[idx]
            for token in query_tokens:
                df = self._doc_freq.get(token, 0)
                if df == 0:
                    continue
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                f = tf.get(token, 0)
                num = f * (self.k1 + 1.0)
                den = f + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                score += idf * num / den
            if score > 0:
                results.append((self._docs[idx]["id"], score))
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

def _rrf_fusion(
    bm25_ranked: list[tuple[int, float]],
    dense_ranked: list[tuple[int, float]],
    k: int = 60,
    bm25_weight: float = 0.4,
) -> list[tuple[int, float]]:
    """
    RRF 融合 BM25 与 BGE 排名，输出统一排序。

    RRF(D) = bm25_weight/(k+rank_bm25) + (1−bm25_weight)/(k+rank_dense)
    """
    scores: dict[int, float] = {}
    for rank, (chunk_id, _) in enumerate(bm25_ranked):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + bm25_weight / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(dense_ranked):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 - bm25_weight) / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from app.logging_config import setup_logging
    setup_logging()
    main()
