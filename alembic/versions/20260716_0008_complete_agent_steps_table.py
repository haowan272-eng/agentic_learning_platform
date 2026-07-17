"""Complete the historical agent_steps table shape.

Revision ID: 20260716_0008
Revises: 20260716_0007
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0008"
down_revision = "20260716_0007"
branch_labels = None
depends_on = None


def _has_table(inspector: sa.Inspector, table_name: str) -> bool:
    return inspector.has_table(table_name)


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    if not _has_table(inspector, table_name):
        return False
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    if not _has_table(inspector, table_name):
        return False
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def _add_column_if_missing(inspector: sa.Inspector, table_name: str, column: sa.Column) -> None:
    if not _has_column(inspector, table_name, column.name):
        op.add_column(table_name, column)


def _create_index_if_missing(
    inspector: sa.Inspector,
    index_name: str,
    table_name: str,
    columns: list[str],
) -> None:
    if _has_table(inspector, table_name) and not _has_index(inspector, table_name, index_name):
        op.create_index(index_name, table_name, columns)


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not _has_table(inspector, "agent_steps"):
        return

    # Some databases already had an older learning-step table named
    # agent_steps. Keep those legacy columns and add the planned runtime
    # columns needed by the LangGraph execution trace.
    _add_column_if_missing(inspector, "agent_steps", sa.Column("task_id", sa.String(length=64), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("run_id", sa.String(length=64), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("step_id", sa.String(length=64), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("step_index", sa.Integer(), nullable=True, server_default="0"))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("agent_name", sa.String(length=64), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("skill_name", sa.String(length=128), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("tool_name", sa.String(length=128), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("objective", sa.Text(), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("input_json", sa.Text(), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("output_json", sa.Text(), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("error_json", sa.Text(), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("started_at", sa.DateTime(), nullable=True))
    _add_column_if_missing(inspector, "agent_steps", sa.Column("finished_at", sa.DateTime(), nullable=True))

    inspector = sa.inspect(bind)
    _create_index_if_missing(inspector, "ix_agent_steps_task_id", "agent_steps", ["task_id"])
    _create_index_if_missing(inspector, "ix_agent_steps_run_id", "agent_steps", ["run_id"])
    _create_index_if_missing(inspector, "ix_agent_steps_step_id", "agent_steps", ["step_id"])
    _create_index_if_missing(inspector, "ix_agent_steps_step_index", "agent_steps", ["step_index"])
    _create_index_if_missing(inspector, "ix_agent_steps_agent_name", "agent_steps", ["agent_name"])
    _create_index_if_missing(inspector, "ix_agent_steps_skill_name", "agent_steps", ["skill_name"])
    _create_index_if_missing(inspector, "ix_agent_steps_tool_name", "agent_steps", ["tool_name"])
    _create_index_if_missing(inspector, "ix_agent_steps_status", "agent_steps", ["status"])


def downgrade() -> None:
    pass
