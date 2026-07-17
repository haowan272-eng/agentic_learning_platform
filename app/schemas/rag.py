"""唯一RAG回答接口的数据契约"""
from typing import Optional

from pydantic import BaseModel, Field


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: Optional[int] = None
    kb_id: Optional[int] = None
    bm25_weight: float = Field(default=0.4, ge=0.0, le=1.0)
    conversation_id: Optional[int] = None
    use_memory: bool = True
    rewrite_query: bool = True


class CitationResult(BaseModel):
    source_id: int
    chunk_id: int
    document_id: Optional[int] = None
    kb_id: Optional[int] = None
    filename: str
    chunk_index: Optional[int] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    heading_path: Optional[str] = None
    source_type: Optional[str] = None
    location: Optional[str] = None
    score: float
    quote: str


class MemoryResult(BaseModel):
    keyword: str
    category: str
    weight: float


class RetrievedSourceResult(CitationResult):
    context: str


class AnswerResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    conversation_id: int
    citations: list[CitationResult]
    retrieved_contexts: list[str]
    retrieved_sources: list[RetrievedSourceResult]
    retrieved_count: int
    memory_used: list[MemoryResult]
    degraded: bool = False
    context_compacted: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)
