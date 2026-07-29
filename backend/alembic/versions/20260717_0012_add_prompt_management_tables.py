"""Add prompt management tables — templates, evaluations, examples.

Revision ID: 20260717_0012
Create Date: 2026-07-17
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260717_0012"
down_revision: Union[str, None] = "20260716_0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("prompt_templates"):
        op.create_table(
            "prompt_templates",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("name", sa.String(128), nullable=False, index=True),
            sa.Column("role", sa.String(32), nullable=False, index=True),
            sa.Column("version", sa.Integer(), nullable=False, default=1, index=True),
            sa.Column("template_text", sa.Text(), nullable=False),
            sa.Column("variables_schema_json", sa.Text(), nullable=True),
            sa.Column("description", sa.Text(), nullable=True),
            sa.Column("is_active", sa.Integer(), nullable=False, default=0, index=True),
            sa.Column("is_default", sa.Integer(), nullable=False, default=0, index=True),
            sa.Column("deployment_status", sa.String(16), nullable=False, default="draft", index=True),
            sa.Column("created_by", sa.String(64), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
            sa.UniqueConstraint("name", "version", name="uq_prompt_name_version"),
        )

    if not inspector.has_table("prompt_evaluations"):
        op.create_table(
            "prompt_evaluations",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("template_id", sa.Integer(), sa.ForeignKey("prompt_templates.id", ondelete="CASCADE"), nullable=False, index=True),
            sa.Column("template_name", sa.String(128), nullable=False, index=True),
            sa.Column("template_version", sa.Integer(), nullable=False, index=True),
            sa.Column("variant", sa.String(32), nullable=False, default="default", index=True),
            sa.Column("task_id", sa.String(64), nullable=False, index=True),
            sa.Column("role", sa.String(32), nullable=False, index=True),
            sa.Column("success", sa.Integer(), nullable=False, default=1),
            sa.Column("latency_ms", sa.Integer(), nullable=True),
            sa.Column("prompt_tokens", sa.Integer(), nullable=True),
            sa.Column("completion_tokens", sa.Integer(), nullable=True),
            sa.Column("retry_count", sa.Integer(), nullable=False, default=0),
            sa.Column("validation_errors", sa.Text(), nullable=True),
            sa.Column("verification_score", sa.Float(), nullable=True),
            sa.Column("citation_count", sa.Integer(), nullable=True),
            sa.Column("evidence_strength", sa.Float(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.UniqueConstraint("template_id", "task_id", name="uq_prompt_eval_task"),
        )

    if not inspector.has_table("prompt_examples"):
        op.create_table(
            "prompt_examples",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("template_name", sa.String(128), nullable=False, index=True),
            sa.Column("role", sa.String(32), nullable=False, index=True),
            sa.Column("input_text", sa.Text(), nullable=False),
            sa.Column("expected_output_json", sa.Text(), nullable=True),
            sa.Column("embedding_id", sa.String(64), nullable=True, index=True),
            sa.Column("tags", sa.Text(), nullable=True),
            sa.Column("quality_score", sa.Float(), nullable=False, default=0.5),
            sa.Column("use_count", sa.Integer(), nullable=False, default=0),
            sa.Column("success_count", sa.Integer(), nullable=False, default=0),
            sa.Column("is_active", sa.Integer(), nullable=False, default=1, index=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
            sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now(), onupdate=sa.func.now()),
        )


def downgrade() -> None:
    op.drop_table("prompt_examples")
    op.drop_table("prompt_evaluations")
    op.drop_table("prompt_templates")
