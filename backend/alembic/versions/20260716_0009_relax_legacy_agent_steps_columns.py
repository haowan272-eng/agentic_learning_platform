"""Relax legacy agent_steps columns for runtime step rows.

Revision ID: 20260716_0009
Revises: 20260716_0008
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0009"
down_revision = "20260716_0008"
branch_labels = None
depends_on = None


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    if not inspector.has_table(table_name):
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("agent_steps"):
        return

    # The database may contain a pre-Agent learning-step table with these
    # required columns. Runtime rows are keyed by task_id/run_id/step_id, so the
    # legacy columns must not be required for new Agent execution traces.
    for column_name, column_type in (
        ("conversation_id", sa.Integer()),
        ("user_id", sa.Integer()),
        ("step_name", sa.String(length=255)),
    ):
        if _has_column(inspector, "agent_steps", column_name):
            op.alter_column(
                "agent_steps",
                column_name,
                existing_type=column_type,
                nullable=True,
            )


def downgrade() -> None:
    pass
