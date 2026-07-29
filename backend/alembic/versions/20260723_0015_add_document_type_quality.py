"""Add document type profile and quality reports.

Revision ID: 20260723_0015
Revises: 20260723_0014
Create Date: 2026-07-23
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "20260723_0015"
down_revision: Union[str, None] = "20260723_0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _has_column(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    return any(column["name"] == column_name for column in inspector.get_columns(table_name))


def _has_index(inspector: sa.Inspector, table_name: str, index_name: str) -> bool:
    return any(index["name"] == index_name for index in inspector.get_indexes(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if not inspector.has_table("documents"):
        return

    document_columns = [
        ("doc_type", sa.Column("doc_type", sa.String(length=64), nullable=True)),
        ("title", sa.Column("title", sa.String(length=255), nullable=True)),
        ("source", sa.Column("source", sa.String(length=64), nullable=True)),
        ("tags_json", sa.Column("tags_json", sa.Text(), nullable=True)),
        ("target_roles_json", sa.Column("target_roles_json", sa.Text(), nullable=True)),
        ("seniority", sa.Column("seniority", sa.String(length=64), nullable=True)),
        ("business_visibility", sa.Column("business_visibility", sa.String(length=32), nullable=True)),
        ("quality_score", sa.Column("quality_score", sa.Integer(), nullable=True)),
        ("quality_level", sa.Column("quality_level", sa.String(length=32), nullable=True)),
        ("structured_summary_json", sa.Column("structured_summary_json", sa.Text(), nullable=True)),
    ]
    for column_name, column in document_columns:
        if not _has_column(inspector, "documents", column_name):
            op.add_column("documents", column)

    inspector = sa.inspect(bind)
    for index_name, column_name in [
        ("ix_documents_doc_type", "doc_type"),
        ("ix_documents_quality_level", "quality_level"),
    ]:
        if not _has_index(inspector, "documents", index_name):
            op.create_index(index_name, "documents", [column_name])

    inspector = sa.inspect(bind)
    if not inspector.has_table("document_quality_reports"):
        op.create_table(
            "document_quality_reports",
            sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
            sa.Column("document_id", sa.Integer(), sa.ForeignKey("documents.id"), nullable=False),
            sa.Column("doc_type", sa.String(length=64), nullable=False),
            sa.Column("quality_score", sa.Integer(), nullable=False),
            sa.Column("level", sa.String(length=32), nullable=False),
            sa.Column("summary_json", sa.Text(), nullable=True),
            sa.Column("issues_json", sa.Text(), nullable=True),
            sa.Column("suggestions_json", sa.Text(), nullable=True),
            sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=True),
        )
    inspector = sa.inspect(bind)
    for index_name, columns in [
        ("ix_document_quality_reports_document_id", ["document_id"]),
        ("ix_document_quality_reports_doc_type", ["doc_type"]),
        ("ix_document_quality_reports_level", ["level"]),
        ("ix_document_quality_reports_created_at", ["created_at"]),
    ]:
        if not _has_index(inspector, "document_quality_reports", index_name):
            op.create_index(index_name, "document_quality_reports", columns)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if inspector.has_table("document_quality_reports"):
        for index_name in [
            "ix_document_quality_reports_created_at",
            "ix_document_quality_reports_level",
            "ix_document_quality_reports_doc_type",
            "ix_document_quality_reports_document_id",
        ]:
            if _has_index(inspector, "document_quality_reports", index_name):
                op.drop_index(index_name, table_name="document_quality_reports")
        op.drop_table("document_quality_reports")

    inspector = sa.inspect(bind)
    if not inspector.has_table("documents"):
        return
    for index_name in ["ix_documents_quality_level", "ix_documents_doc_type"]:
        if _has_index(inspector, "documents", index_name):
            op.drop_index(index_name, table_name="documents")

    inspector = sa.inspect(bind)
    for column_name in [
        "structured_summary_json",
        "quality_level",
        "quality_score",
        "business_visibility",
        "seniority",
        "target_roles_json",
        "tags_json",
        "source",
        "title",
        "doc_type",
    ]:
        if _has_column(inspector, "documents", column_name):
            op.drop_column("documents", column_name)
