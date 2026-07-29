"""Drop the unused embedding LargeBinary column from chunk_embeddings.

Revision ID: 20260730_0018
Revises: 20260730_0017
Create Date: 2026-07-30
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260730_0018"
down_revision = "20260730_0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("chunk_embeddings")}
    if "embedding" in columns:
        op.drop_column("chunk_embeddings", "embedding")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("chunk_embeddings")}
    if "embedding" not in columns:
        op.add_column(
            "chunk_embeddings",
            sa.Column("embedding", sa.LargeBinary(), nullable=True),
        )
