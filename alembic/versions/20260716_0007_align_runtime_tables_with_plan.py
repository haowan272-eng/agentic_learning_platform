"""Align agent runtime tables with the complete project plan.

Revision ID: 20260716_0007
Revises: 20260716_0006
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0007"
down_revision = "20260716_0006"
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


def _add_column_if_missing(
    inspector: sa.Inspector,
    table_name: str,
    column: sa.Column,
) -> None:
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

    if _has_table(inspector, "agent_tasks"):
        _add_column_if_missing(
            inspector,
            "agent_tasks",
            sa.Column("parent_task_id", sa.String(length=64), nullable=True),
        )
        _add_column_if_missing(inspector, "agent_tasks", sa.Column("goal", sa.Text(), nullable=True))
        _add_column_if_missing(inspector, "agent_tasks", sa.Column("intent", sa.String(length=128), nullable=True))
        _add_column_if_missing(inspector, "agent_tasks", sa.Column("kb_scope_json", sa.Text(), nullable=True))
        _add_column_if_missing(
            inspector,
            "agent_tasks",
            sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        )
        _add_column_if_missing(inspector, "agent_tasks", sa.Column("completed_at", sa.DateTime(), nullable=True))
        _create_index_if_missing(inspector, "ix_agent_tasks_parent_task_id", "agent_tasks", ["parent_task_id"])
        _create_index_if_missing(inspector, "ix_agent_tasks_intent", "agent_tasks", ["intent"])

    inspector = sa.inspect(bind)
    if _has_table(inspector, "agent_steps"):
        _add_column_if_missing(inspector, "agent_steps", sa.Column("plan_id", sa.Integer(), nullable=True))
        _add_column_if_missing(inspector, "agent_steps", sa.Column("error_type", sa.String(length=128), nullable=True))
        _add_column_if_missing(inspector, "agent_steps", sa.Column("error_message", sa.Text(), nullable=True))
        _create_index_if_missing(inspector, "ix_agent_steps_plan_id", "agent_steps", ["plan_id"])
        _create_index_if_missing(inspector, "ix_agent_steps_error_type", "agent_steps", ["error_type"])

    inspector = sa.inspect(bind)
    if _has_table(inspector, "tool_calls"):
        _add_column_if_missing(inspector, "tool_calls", sa.Column("agent_name", sa.String(length=64), nullable=True))
        _add_column_if_missing(inspector, "tool_calls", sa.Column("skill_name", sa.String(length=128), nullable=True))
        _add_column_if_missing(inspector, "tool_calls", sa.Column("output_json", sa.Text(), nullable=True))
        _add_column_if_missing(
            inspector,
            "tool_calls",
            sa.Column("ok", sa.Integer(), nullable=False, server_default="1"),
        )
        _add_column_if_missing(inspector, "tool_calls", sa.Column("error_type", sa.String(length=128), nullable=True))
        _add_column_if_missing(inspector, "tool_calls", sa.Column("error_message", sa.Text(), nullable=True))
        _add_column_if_missing(
            inspector,
            "tool_calls",
            sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        )
        _add_column_if_missing(inspector, "tool_calls", sa.Column("latency_ms", sa.Integer(), nullable=True))
        _add_column_if_missing(inspector, "tool_calls", sa.Column("prompt_tokens", sa.Integer(), nullable=True))
        _add_column_if_missing(inspector, "tool_calls", sa.Column("completion_tokens", sa.Integer(), nullable=True))
        _create_index_if_missing(inspector, "ix_tool_calls_agent_name", "tool_calls", ["agent_name"])
        _create_index_if_missing(inspector, "ix_tool_calls_skill_name", "tool_calls", ["skill_name"])
        _create_index_if_missing(inspector, "ix_tool_calls_ok", "tool_calls", ["ok"])
        _create_index_if_missing(inspector, "ix_tool_calls_error_type", "tool_calls", ["error_type"])
        if _has_column(inspector, "tool_calls", "result_json"):
            op.execute(
                sa.text(
                    "UPDATE tool_calls "
                    "SET output_json = result_json "
                    "WHERE output_json IS NULL AND result_json IS NOT NULL"
                )
            )

    inspector = sa.inspect(bind)
    if _has_table(inspector, "verifications"):
        _add_column_if_missing(inspector, "verifications", sa.Column("target_type", sa.String(length=64), nullable=True))
        _add_column_if_missing(inspector, "verifications", sa.Column("target_id", sa.String(length=128), nullable=True))
        _add_column_if_missing(inspector, "verifications", sa.Column("evidence_json", sa.Text(), nullable=True))
        _create_index_if_missing(inspector, "ix_verifications_target_type", "verifications", ["target_type"])
        _create_index_if_missing(inspector, "ix_verifications_target_id", "verifications", ["target_id"])

    inspector = sa.inspect(bind)
    if _has_table(inspector, "user_memories"):
        _add_column_if_missing(inspector, "user_memories", sa.Column("memory_key", sa.String(length=128), nullable=True))
        _add_column_if_missing(inspector, "user_memories", sa.Column("value", sa.Text(), nullable=True))
        _add_column_if_missing(inspector, "user_memories", sa.Column("confidence", sa.Float(), nullable=True))
        _add_column_if_missing(inspector, "user_memories", sa.Column("source_event_id", sa.Integer(), nullable=True))
        _add_column_if_missing(inspector, "user_memories", sa.Column("source_task_id", sa.String(length=64), nullable=True))
        _create_index_if_missing(inspector, "ix_user_memories_memory_key", "user_memories", ["memory_key"])
        _create_index_if_missing(inspector, "ix_user_memories_source_event_id", "user_memories", ["source_event_id"])
        _create_index_if_missing(inspector, "ix_user_memories_source_task_id", "user_memories", ["source_task_id"])


def downgrade() -> None:
    # Keep downgrade conservative because this migration only adds nullable
    # compatibility fields and fills output_json from the legacy result_json.
    pass
