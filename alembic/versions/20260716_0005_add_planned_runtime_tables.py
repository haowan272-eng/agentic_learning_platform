"""Add planned Agent runtime and memory tables.

Revision ID: 20260716_0005
Revises: 20260716_0004
Create Date: 2026-07-16
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0005"
down_revision = "20260716_0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("agent_sessions"):
        return

    op.create_table(
        "agent_sessions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("title", sa.String(length=255), nullable=False),
        sa.Column("active_task_id", sa.String(length=64), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_sessions_session_id", "agent_sessions", ["session_id"], unique=True)
    op.create_index("ix_agent_sessions_user_id", "agent_sessions", ["user_id"])
    op.create_index("ix_agent_sessions_active_task_id", "agent_sessions", ["active_task_id"])

    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), sa.ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("current_node", sa.String(length=128), nullable=True),
        sa.Column("current_step_id", sa.String(length=128), nullable=True),
        sa.Column("checkpoint_json", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_agent_runs_task_id", "agent_runs", ["task_id"])
    op.create_index("ix_agent_runs_run_id", "agent_runs", ["run_id"], unique=True)
    op.create_index("ix_agent_runs_user_id", "agent_runs", ["user_id"])
    op.create_index("ix_agent_runs_status", "agent_runs", ["status"])

    op.create_table(
        "tools",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.String(length=64), nullable=False),
        sa.Column("input_schema_json", sa.Text(), nullable=True),
        sa.Column("output_schema_json", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Integer(), nullable=False),
        sa.Column("timeout_ms", sa.Integer(), nullable=False),
        sa.Column("max_retries", sa.Integer(), nullable=False),
        sa.Column("fallback_tool", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_tools_name", "tools", ["name"], unique=True)
    op.create_index("ix_tools_category", "tools", ["category"])
    op.create_index("ix_tools_enabled", "tools", ["enabled"])

    op.create_table(
        "tool_permissions",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("tool_name", sa.String(length=128), sa.ForeignKey("tools.name", ondelete="CASCADE"), nullable=False),
        sa.Column("agent_name", sa.String(length=128), nullable=False),
        sa.Column("allowed", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("tool_name", "agent_name", name="uq_tool_permission_agent"),
    )
    op.create_index("ix_tool_permissions_tool_name", "tool_permissions", ["tool_name"])
    op.create_index("ix_tool_permissions_agent_name", "tool_permissions", ["agent_name"])
    op.create_index("ix_tool_permissions_allowed", "tool_permissions", ["allowed"])

    op.create_table(
        "memory_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(length=64), nullable=True),
        sa.Column("task_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=128), nullable=False),
        sa.Column("category", sa.String(length=64), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("source", sa.String(length=64), nullable=False),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_memory_events_user_id", "memory_events", ["user_id"])
    op.create_index("ix_memory_events_session_id", "memory_events", ["session_id"])
    op.create_index("ix_memory_events_task_id", "memory_events", ["task_id"])
    op.create_index("ix_memory_events_event_type", "memory_events", ["event_type"])
    op.create_index("ix_memory_events_category", "memory_events", ["category"])
    op.create_index("ix_memory_events_source", "memory_events", ["source"])

    op.create_table(
        "session_summaries",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("summary_until_event_id", sa.Integer(), nullable=True),
        sa.Column("metadata_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_session_summaries_session_id", "session_summaries", ["session_id"])
    op.create_index("ix_session_summaries_user_id", "session_summaries", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_session_summaries_user_id", table_name="session_summaries")
    op.drop_index("ix_session_summaries_session_id", table_name="session_summaries")
    op.drop_table("session_summaries")

    op.drop_index("ix_memory_events_source", table_name="memory_events")
    op.drop_index("ix_memory_events_category", table_name="memory_events")
    op.drop_index("ix_memory_events_event_type", table_name="memory_events")
    op.drop_index("ix_memory_events_task_id", table_name="memory_events")
    op.drop_index("ix_memory_events_session_id", table_name="memory_events")
    op.drop_index("ix_memory_events_user_id", table_name="memory_events")
    op.drop_table("memory_events")

    op.drop_index("ix_tool_permissions_allowed", table_name="tool_permissions")
    op.drop_index("ix_tool_permissions_agent_name", table_name="tool_permissions")
    op.drop_index("ix_tool_permissions_tool_name", table_name="tool_permissions")
    op.drop_table("tool_permissions")

    op.drop_index("ix_tools_enabled", table_name="tools")
    op.drop_index("ix_tools_category", table_name="tools")
    op.drop_index("ix_tools_name", table_name="tools")
    op.drop_table("tools")

    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_user_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_run_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_task_id", table_name="agent_runs")
    op.drop_table("agent_runs")

    op.drop_index("ix_agent_sessions_active_task_id", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_user_id", table_name="agent_sessions")
    op.drop_index("ix_agent_sessions_session_id", table_name="agent_sessions")
    op.drop_table("agent_sessions")
