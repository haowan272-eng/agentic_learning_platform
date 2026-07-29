"""Align tool call and verification table names with the project plan.

Revision ID: 20260716_0006
Revises: 20260716_0005
Create Date: 2026-07-16
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0006"
down_revision = "20260716_0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "agent_tool_calls" in tables and "tool_calls" not in tables:
        op.rename_table("agent_tool_calls", "tool_calls")
    if "agent_verifications" in tables and "verifications" not in tables:
        op.rename_table("agent_verifications", "verifications")


def downgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    tables = set(inspector.get_table_names())
    if "tool_calls" in tables and "agent_tool_calls" not in tables:
        op.rename_table("tool_calls", "agent_tool_calls")
    if "verifications" in tables and "agent_verifications" not in tables:
        op.rename_table("verifications", "agent_verifications")
