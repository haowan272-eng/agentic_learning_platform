"""Store durable human approval payloads for LangGraph interrupts.

Revision ID: 20260716_0011
Revises: 20260716_0010
Create Date: 2026-07-16
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op


revision = "20260716_0011"
down_revision = "20260716_0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("agent_tasks", sa.Column("resume_payload_json", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_tasks", "resume_payload_json")
