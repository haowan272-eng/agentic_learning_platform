"""Add Agent Runtime reliability controls and transactional outbox.

Revision ID: 20260716_0010
Revises: 20260716_0009
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0010"
down_revision = "20260716_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    task_columns = {column["name"] for column in inspector.get_columns("agent_tasks")}
    task_indexes = {index["name"] for index in inspector.get_indexes("agent_tasks")}

    for column in (
        sa.Column("budget_json", sa.Text(), nullable=True),
        sa.Column("idempotency_key", sa.String(length=128), nullable=True),
        sa.Column("cancel_requested", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("lease_owner", sa.String(length=128), nullable=True),
        sa.Column("lease_until", sa.DateTime(), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
    ):
        if column.name not in task_columns:
            op.add_column("agent_tasks", column)

    if "ix_agent_tasks_idempotency_key" not in task_indexes:
        op.create_index("ix_agent_tasks_idempotency_key", "agent_tasks", ["idempotency_key"], unique=True)
    if "ix_agent_tasks_cancel_requested" not in task_indexes:
        op.create_index("ix_agent_tasks_cancel_requested", "agent_tasks", ["cancel_requested"])
    if "ix_agent_tasks_lease_until" not in task_indexes:
        op.create_index("ix_agent_tasks_lease_until", "agent_tasks", ["lease_until"])

    inspector = sa.inspect(bind)
    if not inspector.has_table("agent_outbox"):
        op.create_table(
            "agent_outbox",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("task_id", sa.String(length=64), sa.ForeignKey("agent_tasks.task_id", ondelete="CASCADE"), nullable=False),
            sa.Column("event_type", sa.String(length=64), nullable=False, server_default="agent_task_requested"),
            sa.Column("payload_json", sa.Text(), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
            sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
            sa.Column("available_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.Column("dispatched_at", sa.DateTime(), nullable=True),
            sa.Column("last_error", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.text("CURRENT_TIMESTAMP")),
            sa.UniqueConstraint("task_id", "event_type", name="uq_agent_outbox_task_event"),
        )
        op.create_index("ix_agent_outbox_task_id", "agent_outbox", ["task_id"])
        op.create_index("ix_agent_outbox_status", "agent_outbox", ["status"])
        op.create_index("ix_agent_outbox_available_at", "agent_outbox", ["available_at"])


def downgrade() -> None:
    op.drop_table("agent_outbox")
    op.drop_index("ix_agent_tasks_lease_until", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_cancel_requested", table_name="agent_tasks")
    op.drop_index("ix_agent_tasks_idempotency_key", table_name="agent_tasks")
    for column in ("last_error", "lease_until", "lease_owner", "cancel_requested", "idempotency_key", "budget_json"):
        op.drop_column("agent_tasks", column)
