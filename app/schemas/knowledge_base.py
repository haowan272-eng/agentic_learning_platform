"""知识库相关的 Pydantic 请求/响应模型。"""
from typing import Optional
from pydantic import BaseModel, Field


class KnowledgeBaseCreate(BaseModel):
    """创建知识库请"""
    name: str = Field(..., min_length=1, max_length=255, description="知识库名")
    description: Optional[str] = Field(None, max_length=2000, description="描述")
    chunk_config: Optional[str] = Field(None, description="分块配置 JSON")


class KnowledgeBaseUpdate(BaseModel):
    """更新知识库请求（全部可选）"""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=2000)
    chunk_config: Optional[str] = None


class KnowledgeBaseResponse(BaseModel):
    """知识库详"""
    id: int
    name: str
    description: Optional[str] = None
    chunk_config: Optional[str] = None
    created_by: int
    created_at: str
    updated_at: str
    member_count: int = 0
    role: Optional[str] = None


class AddMemberRequest(BaseModel):
    """添加成员请求"""
    username: str = Field(..., description="用户")
    role: str = Field(..., pattern="^(viewer|editor|admin)$", description="角色: viewer/editor/admin")


class UpdateMemberRequest(BaseModel):
    """更新成员角色请求"""
    role: str = Field(..., pattern="^(viewer|editor|admin)$")


class MemberResponse(BaseModel):
    """成员信息"""
    user_id: int
    username: str
    role: str
    created_at: str
