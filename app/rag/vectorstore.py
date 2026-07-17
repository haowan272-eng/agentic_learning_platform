"""单集合 Qdrant 向量存储。

所有共享文档写入同一集合，通过 payload 中的 document_id/kb_id 过滤。
分块策略指纹只用于 PostgreSQL 审计，不再触发全局别名切换。
"""
from __future__ import annotations

import hashlib
from typing import Optional

from app.core.config import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_DIM,
    QDRANT_PREFER_GRPC,
    QDRANT_QUANTIZATION,
    QDRANT_URL,
)


class QdrantStore:
    def __init__(
        self,
        url: str = QDRANT_URL,
        collection_name: str = QDRANT_COLLECTION_NAME,
        dim: int = QDRANT_DIM,
    ):
        self.url = url
        self.collection_name = collection_name
        self.dim = dim
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from qdrant_client import QdrantClient

            # httpx + WSL 端口转发有兼容性问题，强制走 gRPC (6334)
            self._client = QdrantClient(
                url=self.url,
                api_key=QDRANT_API_KEY,
                prefer_grpc=QDRANT_PREFER_GRPC,
                timeout=30,
                trust_env=False,
            )
        return self._client

    @staticmethod
    def compute_pipeline_hash(
        chunk_size: int,
        chunk_overlap: int,
        model_name: str,
        dim: int,
    ) -> str:
        payload = f"{chunk_size}:{chunk_overlap}:{model_name}:{dim}"
        return hashlib.md5(payload.encode()).hexdigest()[:12]

    def ensure_collection(self, pipeline_hash: Optional[str] = None) -> str:
        """幂等创建固定集合；pipeline_hash 仅为兼容旧调用，不影响集合名"""
        name = self.collection_name
        if not self.client.collection_exists(name):
            from qdrant_client.models import Distance, VectorParams

            kwargs = {
                "collection_name": name,
                "vectors_config": VectorParams(size=self.dim, distance=Distance.COSINE),
            }
            if QDRANT_QUANTIZATION == "int8":
                from qdrant_client.models import (
                    ScalarQuantization,
                    ScalarQuantizationConfig,
                    ScalarType,
                )

                kwargs["quantization_config"] = ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    )
                )
            self.client.create_collection(**kwargs)

        info = self.client.get_collection(name)
        vectors = info.config.params.vectors
        actual_dim = getattr(vectors, "size", None)
        if actual_dim is None and isinstance(vectors, dict):
            sizes = {getattr(item, "size", None) for item in vectors.values()}
            sizes.discard(None)
            actual_dim = sizes.pop() if len(sizes) == 1 else None
        if actual_dim is not None and int(actual_dim) != self.dim:
            raise RuntimeError(
                f"Qdrant 集合 '{name}' 的向量维度为 {actual_dim}，"
                f"当前模型要求 {self.dim}。请更换 QDRANT_COLLECTION_NAME 或重新索引。"
            )
        return name

    def get_active_collection(self) -> Optional[str]:
        try:
            return self.collection_name if self.client.collection_exists(self.collection_name) else None
        except Exception:
            return None

    def upsert_points(
        self,
        points: list[dict],
        pipeline_hash: Optional[str] = None,
    ) -> None:
        from qdrant_client.models import PointStruct

        target = self.ensure_collection()
        for start in range(0, len(points), 128):
            batch = [
                PointStruct(id=p["id"], vector=p["vector"], payload=p["payload"])
                for p in points[start:start + 128]
            ]
            self.client.upsert(collection_name=target, points=batch)

    @staticmethod
    def _filter(
        document_id: Optional[int] = None,
        user_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        kb_id_is_null: bool = False,
    ):
        from qdrant_client.models import FieldCondition, Filter, IsNullCondition, MatchValue

        values = {
            "document_id": document_id,
            "user_id": user_id,
            "kb_id": kb_id,
        }
        conditions = [
            FieldCondition(key=key, match=MatchValue(value=value))
            for key, value in values.items()
            if value is not None
        ]
        if kb_id_is_null:
            conditions.append(IsNullCondition(is_null={"key": "kb_id"}))
        return Filter(must=conditions) if conditions else None

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        document_id: Optional[int] = None,
        user_id: Optional[int] = None,
        kb_id: Optional[int] = None,
        kb_id_is_null: bool = False,
        score_threshold: Optional[float] = None,
    ) -> list[dict]:
        target = self.ensure_collection()
        results = self.client.query_points(
            collection_name=target,
            query=query_vector,
            limit=top_k,
            query_filter=self._filter(document_id, user_id, kb_id, kb_id_is_null),
            score_threshold=score_threshold,
        )
        return [
            {
                "chunk_id": point.payload.get("chunk_id"),
                "document_id": point.payload.get("document_id"),
                "user_id": point.payload.get("user_id"),
                "kb_id": point.payload.get("kb_id"),
                "filename": point.payload.get("filename", "未知文档"),
                "chunk_index": point.payload.get("chunk_index"),
                "content": point.payload.get("content", ""),
                "modality": point.payload.get("modality", "text"),
                "page_start": point.payload.get("page_start"),
                "page_end": point.payload.get("page_end"),
                "heading_path": point.payload.get("heading_path"),
                "parent_content": point.payload.get("parent_content"),
                "source_type": point.payload.get("source_type"),
                "location": point.payload.get("location"),
                "score": round(point.score, 4) if point.score else 0.0,
            }
            for point in results.points
        ]

    def delete_by_document_id(self, document_id: int, strict: bool = False) -> int:
        target = self.get_active_collection()
        if target is None:
            if strict:
                raise RuntimeError(f"Qdrant 集合不存在，无法删除文档 {document_id}")
            return 0
        try:
            self.client.delete(
                collection_name=target,
                points_selector=self._filter(document_id=document_id),
            )
            return -1
        except Exception as exc:
            if strict:
                raise RuntimeError(f"删除文档 {document_id} 的旧向量失败: {exc}") from exc
            return 0

    def delete_points_by_ids(
        self,
        point_ids: list[int],
        collection_name: Optional[str] = None,
    ) -> int:
        if not point_ids:
            return 0
        target = collection_name or self.get_active_collection()
        if not target:
            return 0
        from qdrant_client.models import PointIdsList

        self.client.delete(
            collection_name=target,
            points_selector=PointIdsList(points=point_ids),
        )
        return -1

    def list_point_ids(
        self,
        document_id: Optional[int] = None,
        collection_name: Optional[str] = None,
    ) -> set[int]:
        target = collection_name or self.get_active_collection()
        if not target:
            return set()
        result: set[int] = set()
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=target,
                scroll_filter=self._filter(document_id=document_id),
                limit=1000,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            result.update(point.id for point in points)
            if offset is None:
                return result

    def count_points(
        self,
        user_id: Optional[int] = None,
        kb_id: Optional[int] = None,
    ) -> int:
        try:
            target = self.ensure_collection()
            return self.client.count(
                collection_name=target,
                count_filter=self._filter(user_id=user_id, kb_id=kb_id),
                exact=True,
            ).count
        except Exception:
            return 0


_qdrant_store: Optional[QdrantStore] = None


def get_qdrant_store() -> QdrantStore:
    global _qdrant_store
    if _qdrant_store is None:
        _qdrant_store = QdrantStore()
    return _qdrant_store
