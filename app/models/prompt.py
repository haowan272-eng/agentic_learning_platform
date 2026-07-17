"""Prompt template, version, A/B evaluation, and few-shot example models."""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint, func

from app.core.database import Base


class PromptTemplate(Base):
    """Versioned prompt templates with Jinja2 syntax, stored in DB for hot-reload."""

    __tablename__ = "prompt_templates"
    __table_args__ = (UniqueConstraint("name", "version", name="uq_prompt_name_version"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(128), nullable=False, index=True, comment="Logical name, e.g. supervisor_plan")
    role = Column(String(32), nullable=False, index=True, comment="planner / architect / judge")
    version = Column(Integer, nullable=False, default=1, index=True)
    template_text = Column(Text, nullable=False, comment="Jinja2 template")
    variables_schema_json = Column(Text, nullable=True, comment="JSON Schema for template variables")
    description = Column(Text, nullable=True)
    is_active = Column(Integer, nullable=False, default=0, index=True)
    is_default = Column(Integer, nullable=False, default=0, index=True, comment="Default variant for this name")
    deployment_status = Column(
        String(16), nullable=False, default="draft", index=True,
        comment="draft / staged / active / retired",
    )
    created_by = Column(String(64), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class PromptEvaluation(Base):
    """Per-invocation metrics for prompt variants — powers A/B comparisons."""

    __tablename__ = "prompt_evaluations"
    __table_args__ = (UniqueConstraint("template_id", "task_id", name="uq_prompt_eval_task"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_id = Column(Integer, ForeignKey("prompt_templates.id", ondelete="CASCADE"), nullable=False, index=True)
    template_name = Column(String(128), nullable=False, index=True)
    template_version = Column(Integer, nullable=False, index=True)
    variant = Column(String(32), nullable=False, default="default", index=True)

    task_id = Column(String(64), nullable=False, index=True)
    role = Column(String(32), nullable=False, index=True)

    # Outcome metrics
    success = Column(Integer, nullable=False, default=1, comment="Structured output parsed successfully")
    latency_ms = Column(Integer, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    retry_count = Column(Integer, nullable=False, default=0)
    validation_errors = Column(Text, nullable=True, comment="JSON array of Pydantic validation errors")

    # Quality signals from downstream (filled after Verifier runs)
    verification_score = Column(Float, nullable=True)
    citation_count = Column(Integer, nullable=True)
    evidence_strength = Column(Float, nullable=True, comment="0-1, from Verifier confidence")

    created_at = Column(DateTime, server_default=func.now())


class PromptExample(Base):
    """Few-shot examples with embeddings for dynamic selection."""

    __tablename__ = "prompt_examples"

    id = Column(Integer, primary_key=True, autoincrement=True)
    template_name = Column(String(128), nullable=False, index=True)
    role = Column(String(32), nullable=False, index=True)

    input_text = Column(Text, nullable=False, comment="User input or query")
    expected_output_json = Column(Text, nullable=True, comment="Expected structured output")
    embedding_id = Column(String(64), nullable=True, index=True, comment="Qdrant point id for similarity search")
    tags = Column(Text, nullable=True, comment="Comma-separated tags for filtering")

    quality_score = Column(Float, nullable=False, default=0.5, comment="0-1 human-labeled quality")
    use_count = Column(Integer, nullable=False, default=0)
    success_count = Column(Integer, nullable=False, default=0)

    is_active = Column(Integer, nullable=False, default=1, index=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
