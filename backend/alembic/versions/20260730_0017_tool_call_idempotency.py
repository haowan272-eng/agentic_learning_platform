"""Add replay-safe idempotency metadata to tool calls.

Revision ID: 20260730_0017
Revises: 20260723_0016
Create Date: 2026-07-30
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260730_0017"
down_revision = "20260723_0016"
branch_labels = None
depends_on = None


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {column["name"] for column in inspector.get_columns("tool_calls")}
    for column in (
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    ):
        if column.name not in columns:
            op.add_column("tool_calls", column)
    inspector = sa.inspect(bind)
    if "ix_tool_calls_idempotency_key" not in {index["name"] for index in inspector.get_indexes("tool_calls")}:
        op.create_index("ix_tool_calls_idempotency_key", "tool_calls", ["idempotency_key"], unique=True)
    if "ix_tool_calls_status" not in {index["name"] for index in inspector.get_indexes("tool_calls")}:
        op.create_index("ix_tool_calls_status", "tool_calls", ["status"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if _has_index(inspector, "tool_calls", "ix_tool_calls_status"):
        op.drop_index("ix_tool_calls_status", table_name="tool_calls")
    if _has_index(inspector, "tool_calls", "ix_tool_calls_idempotency_key"):
        op.drop_index("ix_tool_calls_idempotency_key", table_name="tool_calls")
    columns = {column["name"] for column in sa.inspect(bind).get_columns("tool_calls")}
    for name in ("finished_at", "started_at", "status", "idempotency_key"):
        if name in columns:
            op.drop_column("tool_calls", name)
