"""Add learning domain tables for coaching and dashboard metrics.

Revision ID: 20260718_0013
Create Date: 2026-07-18
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260718_0013"
down_revision: Union[str, None] = "20260717_0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "learning_profiles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("target_role", sa.String(128), nullable=True, index=True),
        sa.Column("goal", sa.Text(), nullable=True),
        sa.Column("current_level", sa.String(64), nullable=False, default="unknown", index=True),
        sa.Column("weekly_minutes", sa.Integer(), nullable=False, default=300),
        sa.Column("preference_json", sa.Text(), nullable=True),
        sa.Column("diagnostic_summary", sa.Text(), nullable=True),
        sa.Column("readiness_score", sa.Float(), nullable=False, default=0.0),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("user_id", name="uq_learning_profile_user"),
    )
    op.create_table(
        "learning_weaknesses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("topic", sa.String(160), nullable=False, index=True),
        sa.Column("category", sa.String(64), nullable=False, default="knowledge", index=True),
        sa.Column("severity", sa.Float(), nullable=False, default=0.5, index=True),
        sa.Column("confidence", sa.Float(), nullable=False, default=0.5),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("last_seen_task_id", sa.String(64), nullable=True, index=True),
        sa.Column("status", sa.String(32), nullable=False, default="open", index=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("user_id", "topic", name="uq_learning_weakness_user_topic"),
    )
    op.create_table(
        "learning_practices",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("task_id", sa.String(64), nullable=True, index=True),
        sa.Column("kb_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("topic", sa.String(160), nullable=False, index=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("expected_answer", sa.Text(), nullable=True),
        sa.Column("difficulty", sa.String(32), nullable=False, default="medium", index=True),
        sa.Column("source_json", sa.Text(), nullable=True),
        sa.Column("status", sa.String(32), nullable=False, default="assigned", index=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_table(
        "learning_assessments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("practice_id", sa.Integer(), sa.ForeignKey("learning_practices.id", ondelete="SET NULL"), nullable=True, index=True),
        sa.Column("task_id", sa.String(64), nullable=True, index=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("feedback", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, default=0.0, index=True),
        sa.Column("rubric_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "learning_review_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("weakness_id", sa.Integer(), sa.ForeignKey("learning_weaknesses.id", ondelete="CASCADE"), nullable=True, index=True),
        sa.Column("topic", sa.String(160), nullable=False, index=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("due_at", sa.DateTime(), nullable=False, index=True),
        sa.Column("interval_days", sa.Integer(), nullable=False, default=1),
        sa.Column("status", sa.String(32), nullable=False, default="due", index=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_table(
        "learning_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
        sa.Column("event_type", sa.String(64), nullable=False, index=True),
        sa.Column("task_id", sa.String(64), nullable=True, index=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), index=True),
    )


def downgrade() -> None:
    op.drop_table("learning_events")
    op.drop_table("learning_review_items")
    op.drop_table("learning_assessments")
    op.drop_table("learning_practices")
    op.drop_table("learning_weaknesses")
    op.drop_table("learning_profiles")
