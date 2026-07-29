"""学习领域 ORM 模型。

围绕面试能力提升场景建模：
- LearningProfile:    用户学习画像（目标岗位、能力等级）
- LearningWeakness:   薄弱点识别与跟踪
- LearningPractice:   练习任务生成
- LearningAssessment: 练习评估与评分
- LearningReviewItem: 间隔复习调度
- LearningEvent:      学习行为事件记录
- LearningLongTermAsset: 长期学习资产（技术栈画像、项目经历、面试回答版本等）
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from app.core.database import Base


class LearningProfile(Base):
    """用户学习画像——每个用户一条记录，记录目标岗位、能力等级和可用时间。"""
    __tablename__ = "learning_profiles"
    __table_args__ = (UniqueConstraint("user_id", name="uq_learning_profile_user"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    target_role = Column(String(128), nullable=True, index=True)
    goal = Column(Text, nullable=True)
    current_level = Column(String(64), nullable=False, default="unknown", index=True)
    weekly_minutes = Column(Integer, nullable=False, default=300)
    preference_json = Column(Text, nullable=True)
    diagnostic_summary = Column(Text, nullable=True)
    readiness_score = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LearningWeakness(Base):
    """学习薄弱点——按用户和主题去重，严重程度随出现频次递增。"""
    __tablename__ = "learning_weaknesses"
    __table_args__ = (UniqueConstraint("user_id", "topic", name="uq_learning_weakness_user_topic"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    topic = Column(String(160), nullable=False, index=True)
    category = Column(String(64), nullable=False, default="knowledge", index=True)
    severity = Column(Float, nullable=False, default=0.5, index=True)
    confidence = Column(Float, nullable=False, default=0.5)
    evidence_json = Column(Text, nullable=True)
    last_seen_task_id = Column(String(64), nullable=True, index=True)
    status = Column(String(32), nullable=False, default="open", index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LearningPractice(Base):
    """练习任务——根据薄弱点自动生成面试回答练习，关联知识库资料。"""
    __tablename__ = "learning_practices"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    task_id = Column(String(64), nullable=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True)
    topic = Column(String(160), nullable=False, index=True)
    question = Column(Text, nullable=False)
    expected_answer = Column(Text, nullable=True)
    difficulty = Column(String(32), nullable=False, default="medium", index=True)
    source_json = Column(Text, nullable=True)
    status = Column(String(32), nullable=False, default="assigned", index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LearningAssessment(Base):
    """练习评估——记录用户练习答案、反馈和评分，使用面试评分 Rubric。"""
    __tablename__ = "learning_assessments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    practice_id = Column(Integer, ForeignKey("learning_practices.id", ondelete="SET NULL"), nullable=True, index=True)
    task_id = Column(String(64), nullable=True, index=True)
    answer = Column(Text, nullable=True)
    feedback = Column(Text, nullable=False)
    score = Column(Float, nullable=False, default=0.0, index=True)
    rubric_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class LearningReviewItem(Base):
    """间隔复习条目——基于薄弱点生成复习计划，按 due_at 到期提醒。"""
    __tablename__ = "learning_review_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    weakness_id = Column(Integer, ForeignKey("learning_weaknesses.id", ondelete="CASCADE"), nullable=True, index=True)
    topic = Column(String(160), nullable=False, index=True)
    prompt = Column(Text, nullable=False)
    due_at = Column(DateTime, nullable=False, index=True)
    interval_days = Column(Integer, nullable=False, default=1)
    status = Column(String(32), nullable=False, default="due", index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class LearningEvent(Base):
    """学习行为事件——记录所有学习相关操作，用于 dashboard 活跃天数等统计。"""
    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    task_id = Column(String(64), nullable=True, index=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)


class LearningLongTermAsset(Base):
    """长期学习资产——存储技术栈画像、项目经历库、面试回答版本、能力趋势等持久资产。"""
    __tablename__ = "learning_long_term_assets"
    __table_args__ = (
        UniqueConstraint("user_id", "asset_type", "asset_key", name="uq_learning_asset_user_type_key"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_type = Column(String(64), nullable=False, index=True)
    asset_key = Column(String(160), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    summary = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)
    confidence = Column(Float, nullable=False, default=0.6)
    weight = Column(Float, nullable=False, default=1.0)
    source = Column(String(64), nullable=False, default="agent", index=True)
    status = Column(String(32), nullable=False, default="active", index=True)
    task_id = Column(String(64), nullable=True, index=True)
    kb_id = Column(Integer, ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True)
    conversation_id = Column(Integer, ForeignKey("rag_conversations.id", ondelete="SET NULL"), nullable=True, index=True)
    version = Column(Integer, nullable=False, default=1)
    trend_value = Column(Float, nullable=True)
    observed_at = Column(DateTime, nullable=True, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
