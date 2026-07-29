"""Add knowledge base visibility.

Revision ID: 20260723_0014
Revises: 20260718_0013
Create Date: 2026-07-23
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0014"
down_revision: Union[str, None] = "20260718_0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("knowledge_bases"):
        return

    if not _has_column(inspector, "knowledge_bases", "visibility"):
        op.add_column(
            "knowledge_bases",
            sa.Column("visibility", sa.String(length=20), nullable=False, server_default="private"),
        )
        op.alter_column("knowledge_bases", "visibility", server_default=None)

    inspector = sa.inspect(bind)
    if not _has_index(inspector, "knowledge_bases", "ix_knowledge_bases_visibility"):
        op.create_index("ix_knowledge_bases_visibility", "knowledge_bases", ["visibility"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("knowledge_bases"):
        return
    if _has_index(inspector, "knowledge_bases", "ix_knowledge_bases_visibility"):
        op.drop_index("ix_knowledge_bases_visibility", table_name="knowledge_bases")
    inspector = sa.inspect(bind)
    if _has_column(inspector, "knowledge_bases", "visibility"):
        op.drop_column("knowledge_bases", "visibility")
