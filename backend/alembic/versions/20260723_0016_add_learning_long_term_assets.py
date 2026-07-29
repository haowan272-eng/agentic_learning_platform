"""Add learning long-term assets.

Revision ID: 20260723_0016
Revises: 20260723_0015
Create Date: 2026-07-23
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0016"
down_revision: Union[str, None] = "20260723_0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("learning_long_term_assets"):
        return

    op.create_table(
        "learning_long_term_assets",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("asset_type", sa.String(length=64), nullable=False),
        sa.Column("asset_key", sa.String(length=160), nullable=False),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("evidence_json", sa.Text(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.6"),
        sa.Column("weight", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("source", sa.String(length=64), nullable=False, server_default="agent"),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="active"),
        sa.Column("task_id", sa.String(length=64), nullable=True),
        sa.Column("kb_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id", ondelete="SET NULL"), nullable=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id", ondelete="SET NULL"), nullable=True),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("rag_conversations.id", ondelete="SET NULL"), nullable=True),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("trend_value", sa.Float(), nullable=True),
        sa.Column("observed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        sa.UniqueConstraint("user_id", "asset_type", "asset_key", name="uq_learning_asset_user_type_key"),
    )

    inspector = sa.inspect(bind)
    for index_name, columns in [
        ("ix_learning_long_term_assets_user_id", ["user_id"]),
        ("ix_learning_long_term_assets_asset_type", ["asset_type"]),
        ("ix_learning_long_term_assets_asset_key", ["asset_key"]),
        ("ix_learning_long_term_assets_source", ["source"]),
        ("ix_learning_long_term_assets_status", ["status"]),
        ("ix_learning_long_term_assets_task_id", ["task_id"]),
        ("ix_learning_long_term_assets_kb_id", ["kb_id"]),
        ("ix_learning_long_term_assets_document_id", ["document_id"]),
        ("ix_learning_long_term_assets_conversation_id", ["conversation_id"]),
        ("ix_learning_long_term_assets_observed_at", ["observed_at"]),
    ]:
        if not _has_index(inspector, "learning_long_term_assets", index_name):
            op.create_index(index_name, "learning_long_term_assets", columns)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("learning_long_term_assets"):
        return
    for index_name in [
        "ix_learning_long_term_assets_observed_at",
        "ix_learning_long_term_assets_conversation_id",
        "ix_learning_long_term_assets_document_id",
        "ix_learning_long_term_assets_kb_id",
        "ix_learning_long_term_assets_task_id",
        "ix_learning_long_term_assets_status",
        "ix_learning_long_term_assets_source",
        "ix_learning_long_term_assets_asset_key",
        "ix_learning_long_term_assets_asset_type",
        "ix_learning_long_term_assets_user_id",
    ]:
        if _has_index(inspector, "learning_long_term_assets", index_name):
            op.drop_index(index_name, table_name="learning_long_term_assets")
    op.drop_table("learning_long_term_assets")
