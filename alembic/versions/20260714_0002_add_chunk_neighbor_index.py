"""Add chunk neighbor lookup index.

Revision ID: 20260714_0002
Revises: 20260707_0001
Create Date: 2026-07-14
"""
from __future__ import annotations

from alembic import op


revision = "20260714_0002"
down_revision = "20260707_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_documents_chunks_document_chunk_index",
        "documents_chunks",
        ["document_id", "chunk_index"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_chunks_document_chunk_index",
        table_name="documents_chunks",
    )
