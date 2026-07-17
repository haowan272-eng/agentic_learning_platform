"""私有RAG对话API模型"""
from typing import Optional

from pydantic import BaseModel


class ConversationResponse(BaseModel):
    id: int
    title: str
    kb_id: Optional[int] = None
    created_at: str
    updated_at: str


class MessageResponse(BaseModel):
    id: int
    role: str
    content: str
    citations_json: Optional[str] = None
    memory_json: Optional[str] = None
    degraded: bool = False
    created_at: str
