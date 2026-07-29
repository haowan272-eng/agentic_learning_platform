"""Baseline current RAG schema.

Revision ID: 20260707_0001
Revises:
Create Date: 2026-07-07
"""
from __future__ import annotations

from alembic import op

from app.core.database import Base
import app.models  # noqa: F401

revision = "20260707_0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    connection = op.get_bind()
    Base.metadata.create_all(bind=connection)


def downgrade() -> None:
    connection = op.get_bind()
    Base.metadata.drop_all(bind=connection)
