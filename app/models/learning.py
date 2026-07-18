from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from app.core.database import Base


class LearningProfile(Base):
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
    __tablename__ = "learning_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    task_id = Column(String(64), nullable=True, index=True)
    payload_json = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
