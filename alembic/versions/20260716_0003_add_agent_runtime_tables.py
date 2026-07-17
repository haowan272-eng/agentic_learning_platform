"""Add Agent Runtime persistence tables.

Revision ID: 20260716_0003
Revises: 20260714_0002
Create Date: 2026-07-16
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0003"
down_revision = "20260714_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("agent_tasks"):
        return

    op.create_table(
        "agent_tasks",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("username", sa.String(length=128), nullable=False),
        sa.Column("task_type", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("user_input", sa.Text(), nullable=False),
        sa.Column("final_answer", sa.Text(), nullable=True),
        sa.Column("kb_id", sa.Integer(), sa.ForeignKey("knowledge_bases.id"), nullable=True),
        sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=True),
        sa.Column("conversation_id", sa.Integer(), sa.ForeignKey("rag_conversations.id"), nullable=True),
        sa.Column("state_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_tasks_session_id", "agent_tasks", ["session_id"])
    op.create_index("ix_agent_tasks_task_id", "agent_tasks", ["task_id"], unique=True)
    op.create_index("ix_agent_tasks_run_id", "agent_tasks", ["run_id"])
    op.create_index("ix_agent_tasks_user_id", "agent_tasks", ["user_id"])
    op.create_index("ix_agent_tasks_username", "agent_tasks", ["username"])
    op.create_index("ix_agent_tasks_task_type", "agent_tasks", ["task_type"])
    op.create_index("ix_agent_tasks_status", "agent_tasks", ["status"])
    op.create_index("ix_agent_tasks_kb_id", "agent_tasks", ["kb_id"])
    op.create_index("ix_agent_tasks_document_id", "agent_tasks", ["document_id"])
    op.create_index("ix_agent_tasks_conversation_id", "agent_tasks", ["conversation_id"])

    op.create_table(
        "agent_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("session_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column("event_index", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=True),
        sa.Column("skill_name", sa.String(length=64), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=True),
        sa.Column("step_id", sa.String(length=64), nullable=True),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("task_id", "event_index", name="uq_agent_event_index"),
    )
    op.create_index("ix_agent_events_session_id", "agent_events", ["session_id"])
    op.create_index("ix_agent_events_task_id", "agent_events", ["task_id"])
    op.create_index("ix_agent_events_run_id", "agent_events", ["run_id"])
    op.create_index("ix_agent_events_event_type", "agent_events", ["event_type"])

    op.create_table(
        "agent_tool_calls",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("step_id", sa.String(length=64), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=False),
        sa.Column("input_json", sa.Text(), nullable=True),
        sa.Column("result_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_tool_calls_task_id", "agent_tool_calls", ["task_id"])
    op.create_index("ix_agent_tool_calls_run_id", "agent_tool_calls", ["run_id"])
    op.create_index("ix_agent_tool_calls_step_id", "agent_tool_calls", ["step_id"])
    op.create_index("ix_agent_tool_calls_tool_name", "agent_tool_calls", ["tool_name"])

    op.create_table(
        "agent_verifications",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("step_id", sa.String(length=64), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("issues_json", sa.Text(), nullable=True),
        sa.Column("payload_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_verifications_task_id", "agent_verifications", ["task_id"])
    op.create_index("ix_agent_verifications_run_id", "agent_verifications", ["run_id"])
    op.create_index("ix_agent_verifications_step_id", "agent_verifications", ["step_id"])
    op.create_index("ix_agent_verifications_status", "agent_verifications", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_verifications_status", table_name="agent_verifications")
    op.drop_index("ix_agent_verifications_step_id", table_name="agent_verifications")
    op.drop_index("ix_agent_verifications_run_id", table_name="agent_verifications")
    op.drop_index("ix_agent_verifications_task_id", table_name="agent_verifications")
    op.drop_table("agent_verifications")

    op.drop_index("ix_agent_tool_calls_tool_name", table_name="agent_tool_calls")
    op.drop_index("ix_agent_tool_calls_step_id", table_name="agent_tool_calls")
    op.drop_index("ix_agent_tool_calls_run_id", table_name="agent_tool_calls")
    op.drop_index("ix_agent_tool_calls_task_id", table_name="agent_tool_calls")
    op.drop_table("agent_tool_calls")

    op.drop_index("ix_agent_events_event_type", table_name="agent_events")
    op.drop_index("ix_agent_events_run_id", table_name="agent_events")
    op.drop_index("ix_agent_events_task_id", table_name="agent_events")
    op.drop_index("ix_agent_events_session_id", table_name="agent_events")
    op.drop_table("agent_events")

    op.drop_index("ix_agent_tasks_conversation_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_document_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_kb_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_status", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_task_type", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_username", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_user_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_run_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_task_id", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_session_id", table_name="agent_tasks")
    op.drop_table("agent_tasks")
