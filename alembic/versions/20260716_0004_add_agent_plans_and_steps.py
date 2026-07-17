"""Add Agent plan and step persistence tables.

Revision ID: 20260716_0004
Revises: 20260716_0003
Create Date: 2026-07-16
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0004"
down_revision = "20260716_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if inspector.has_table("agent_plans"):
        return

    op.create_table(
        "agent_plans",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), sa.ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("plan_version", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("goal", sa.Text(), nullable=False),
        sa.Column("intent", sa.String(length=128), nullable=False),
        sa.Column("plan_json", sa.Text(), nullable=False),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_index("ix_agent_plans_task_id", "agent_plans", ["task_id"])
    op.create_index("ix_agent_plans_run_id", "agent_plans", ["run_id"])
    op.create_index("ix_agent_plans_plan_version", "agent_plans", ["plan_version"])
    op.create_index("ix_agent_plans_source", "agent_plans", ["source"])
    op.create_index("ix_agent_plans_status", "agent_plans", ["status"])
    op.create_index("ix_agent_plans_intent", "agent_plans", ["intent"])

    op.create_table(
        "agent_steps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("task_id", sa.String(length=64), sa.ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("step_id", sa.String(length=64), nullable=False),
        sa.Column("step_index", sa.Integer(), nullable=False),
        sa.Column("agent_name", sa.String(length=64), nullable=True),
        sa.Column("skill_name", sa.String(length=128), nullable=True),
        sa.Column("tool_name", sa.String(length=128), nullable=True),
        sa.Column("objective", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("input_json", sa.Text(), nullable=True),
        sa.Column("output_json", sa.Text(), nullable=True),
        sa.Column("error_json", sa.Text(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.UniqueConstraint("run_id", "step_id", name="uq_agent_step_run_step"),
    )
    op.create_index("ix_agent_steps_task_id", "agent_steps", ["task_id"])
    op.create_index("ix_agent_steps_run_id", "agent_steps", ["run_id"])
    op.create_index("ix_agent_steps_step_id", "agent_steps", ["step_id"])
    op.create_index("ix_agent_steps_step_index", "agent_steps", ["step_index"])
    op.create_index("ix_agent_steps_agent_name", "agent_steps", ["agent_name"])
    op.create_index("ix_agent_steps_skill_name", "agent_steps", ["skill_name"])
    op.create_index("ix_agent_steps_tool_name", "agent_steps", ["tool_name"])
    op.create_index("ix_agent_steps_status", "agent_steps", ["status"])


def downgrade() -> None:
    op.drop_index("ix_agent_steps_status", table_name="agent_steps")
    op.drop_index("ix_agent_steps_tool_name", table_name="agent_steps")
    op.drop_index("ix_agent_steps_skill_name", table_name="agent_steps")
    op.drop_index("ix_agent_steps_agent_name", table_name="agent_steps")
    op.drop_index("ix_agent_steps_step_index", table_name="agent_steps")
    op.drop_index("ix_agent_steps_step_id", table_name="agent_steps")
    op.drop_index("ix_agent_steps_run_id", table_name="agent_steps")
    op.drop_index("ix_agent_steps_task_id", table_name="agent_steps")
    op.drop_table("agent_steps")

    op.drop_index("ix_agent_plans_intent", table_name="agent_plans")
    op.drop_index("ix_agent_plans_status", table_name="agent_plans")
    op.drop_index("ix_agent_plans_source", table_name="agent_plans")
    op.drop_index("ix_agent_plans_plan_version", table_name="agent_plans")
    op.drop_index("ix_agent_plans_run_id", table_name="agent_plans")
    op.drop_index("ix_agent_plans_task_id", table_name="agent_plans")
    op.drop_table("agent_plans")
