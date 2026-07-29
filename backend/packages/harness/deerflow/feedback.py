"""Verification feedback loop — collect, analyze, and act on verification failures.

The loop has three stages:

1. **Collect** — structured failure records are persisted via the memory event
   system whenever the verification tool rejects or flags a proposal.
2. **Analyze** — a lightweight query aggregates recent failures into actionable
   patterns (per failure-type, per knowledge base, per time window).
3. **Act** — a rule engine maps patterns to recommended parameter adjustments
   (top_k bump, chunk-size change, retrieval-mode switch).  Recommendations
   are surfaced to the Planner prompt so the Supervisor can adapt proactively.

Design principle: the loop never mutates production config automatically; it
produces *recommendations* that human operators or the Supervisor can apply.
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from app.core.database import SessionLocal
from app.models.agent_runtime import AgentVerification, MemoryEvent

logger = logging.getLogger(__name__)

# ── types ─────────────────────────────────────────────────────────────

FailureType = Literal[
    "citation_missing",
    "retrieval_empty",
    "evidence_weak",
    "citation_count_low",
    "source_diversity_low",
    "hallucination_suspected",
    "context_irrelevant",
    "judge_unavailable",
    "other",
]

RepairStrategy = Literal[
    "rewrite_query",
    "expand_top_k",
    "dense_only",
    "sparse_only",
    "hybrid_boost",
    "decompose_query",
    "expand_chunk_context",
    "switch_embedding_model",
    "request_user_clarification",
]


@dataclass
class FailureSignal:
    """A single verification-detected failure, stored for later analysis."""

    task_id: str
    run_id: str | None
    failure_type: FailureType
    message: str
    repair_strategy_used: RepairStrategy | None = None
    repair_successful: bool | None = None
    kb_id: int | None = None
    query_snippet: str = ""
    top_k_used: int = 5
    source_count: int = 0
    citation_count: int = 0
    confidence: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class FailurePattern:
    """Aggregated pattern from a window of recent failures."""

    failure_type: FailureType
    count: int
    pct_of_total: float
    avg_source_count: float
    avg_citation_count: float
    top_repair_strategies: list[tuple[RepairStrategy, int]]
    sample_queries: list[str]


@dataclass
class RetrievalRecommendation:
    """Actionable recommendation derived from failure patterns."""

    parameter: str
    current_value: object
    recommended_value: object
    rationale: str
    confidence: float
    auto_apply: bool = False
    repair_strategy: RepairStrategy | None = None


# ── strategy registry ─────────────────────────────────────────────────

REPAIR_STRATEGIES: dict[RepairStrategy, dict[str, Any]] = {
    "rewrite_query": {
        "description": "重写查询以增加关键词覆盖",
        "modifies": {"rewrite_query": True},
        "cost": "low",
    },
    "expand_top_k": {
        "description": "扩大 top_k 以获取更多候选",
        "modifies": {"top_k_multiplier": 2.0},
        "cost": "low",
    },
    "dense_only": {
        "description": "仅使用稠密向量检索",
        "modifies": {"bm25_weight": 0.0},
        "cost": "low",
    },
    "sparse_only": {
        "description": "仅使用 BM25 关键词检索",
        "modifies": {"bm25_weight": 1.0},
        "cost": "low",
    },
    "hybrid_boost": {
        "description": "同时提升稠密和稀疏权重",
        "modifies": {"bm25_weight": 0.5, "top_k_multiplier": 1.5},
        "cost": "medium",
    },
    "decompose_query": {
        "description": "将复杂查询分解为多个子查询并行检索",
        "modifies": {"decompose": True},
        "cost": "high",
    },
    "expand_chunk_context": {
        "description": "启用父chunk上下文扩展",
        "modifies": {"parent_context": True},
        "cost": "medium",
    },
    "switch_embedding_model": {
        "description": "切换到备选 embedding 模型",
        "modifies": {"embedding_model": "fallback"},
        "cost": "high",
    },
    "request_user_clarification": {
        "description": "暂停并要求用户提供更具体的查询",
        "modifies": {"approval_required": True},
        "cost": "low",
    },
}


# ── collect ───────────────────────────────────────────────────────────


def record_verification_failure(signal: FailureSignal) -> int | None:
    """Persist a structured failure signal for later analysis."""
    with SessionLocal() as db:
        row = MemoryEvent(
            user_id=None,  # populated by the calling context if available
            session_id=None,
            task_id=signal.task_id,
            event_type="verification_failure",
            category="feedback",
            content=(
                f"[{signal.failure_type}] {signal.message} "
                f"(sources={signal.source_count}, citations={signal.citation_count}, "
                f"top_k={signal.top_k_used}, confidence={signal.confidence})"
            ),
            source="verifier_feedback",
            metadata_json=__import__("json").dumps(
                {
                    "failure_type": signal.failure_type,
                    "repair_strategy_used": signal.repair_strategy_used,
                    "repair_successful": signal.repair_successful,
                    "kb_id": signal.kb_id,
                    "query_snippet": signal.query_snippet[:500],
                    "top_k_used": signal.top_k_used,
                    "source_count": signal.source_count,
                    "citation_count": signal.citation_count,
                    "confidence": signal.confidence,
                },
                ensure_ascii=False,
            ),
            created_at=signal.created_at,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        return row.id


# ── analyze ───────────────────────────────────────────────────────────


def analyze_recent_failures(
    *,
    kb_id: int | None = None,
    window_days: int = 7,
    limit: int = 200,
) -> list[FailurePattern]:
    """Aggregate recent verification failures into actionable patterns.

    Returns a list ordered by count descending — the most frequent failure
    types come first.
    """
    try:
        with SessionLocal() as db:
            since = datetime.now(timezone.utc) - timedelta(days=max(1, int(window_days)))
            rows = (
                db.query(MemoryEvent)
                .filter(
                    MemoryEvent.event_type == "verification_failure",
                    MemoryEvent.category == "feedback",
                    MemoryEvent.created_at >= since,
                )
                .order_by(MemoryEvent.id.desc())
                .limit(max(1, min(int(limit), 1000)))
                .all()
            )
    except Exception:
        logger.debug("feedback analysis unavailable", exc_info=True)
        return []

    if not rows:
        return []

    import json

    type_counter: Counter[FailureType] = Counter()
    type_sources: dict[FailureType, list[float]] = {}
    type_citations: dict[FailureType, list[float]] = {}
    type_strategies: dict[FailureType, Counter[RepairStrategy]] = {}
    type_queries: dict[FailureType, list[str]] = {}

    for row in rows:
        try:
            meta = json.loads(row.metadata_json or "{}")
        except json.JSONDecodeError:
            continue

        ftype: FailureType = meta.get("failure_type") or "other"
        type_counter[ftype] += 1
        type_sources.setdefault(ftype, []).append(float(meta.get("source_count") or 0))
        type_citations.setdefault(ftype, []).append(float(meta.get("citation_count") or 0))
        strategy: RepairStrategy | None = meta.get("repair_strategy_used")
        if strategy:
            type_strategies.setdefault(ftype, Counter())[strategy] += 1
        query = str(meta.get("query_snippet") or "")[:120]
        if query:
            type_queries.setdefault(ftype, []).append(query)

    total = max(1, sum(type_counter.values()))
    patterns: list[FailurePattern] = []
    for ftype, count in type_counter.most_common(10):
        sources = type_sources.get(ftype, [0.0])
        citations = type_citations.get(ftype, [0.0])
        strategies = type_strategies.get(ftype, Counter())
        patterns.append(
            FailurePattern(
                failure_type=ftype,
                count=count,
                pct_of_total=round(count / total * 100, 1),
                avg_source_count=round(sum(sources) / max(1, len(sources)), 1),
                avg_citation_count=round(sum(citations) / max(1, len(citations)), 1),
                top_repair_strategies=strategies.most_common(3),
                sample_queries=type_queries.get(ftype, [])[:5],
            )
        )
    return patterns


# ── act ───────────────────────────────────────────────────────────────


def recommend_parameter_changes(
    patterns: list[FailurePattern],
    *,
    current_top_k: int = 5,
    current_chunk_size: int = 500,
    current_bm25_weight: float = 0.3,
) -> list[RetrievalRecommendation]:
    """Map failure patterns to actionable retrieval-parameter recommendations.

    Rules are intentionally conservative: a pattern must appear at least 3
    times and represent ≥15% of recent failures to trigger a recommendation.
    """
    recommendations: list[RetrievalRecommendation] = []
    total = sum(p.count for p in patterns)

    for pattern in patterns:
        if pattern.count < 3:
            continue
        pct = pattern.count / max(1, total)

        if pattern.failure_type == "retrieval_empty" and pct >= 0.15:
            recommendations.append(
                RetrievalRecommendation(
                    parameter="top_k",
                    current_value=current_top_k,
                    recommended_value=min(20, max(8, current_top_k * 2)),
                    rationale=f"检索为空占{pct:.0%}的失败，扩大top_k可增加召回覆盖率",
                    confidence=0.75,
                    auto_apply=True,
                    repair_strategy="expand_top_k",
                )
            )
            if current_bm25_weight < 0.5:
                recommendations.append(
                    RetrievalRecommendation(
                        parameter="bm25_weight",
                        current_value=current_bm25_weight,
                        recommended_value=0.5,
                        rationale="检索为空时增加BM25权重可提升关键词匹配",
                        confidence=0.60,
                        auto_apply=False,
                        repair_strategy="hybrid_boost",
                    )
                )

        if pattern.failure_type == "citation_missing" and pct >= 0.15:
            recommendations.append(
                RetrievalRecommendation(
                    parameter="parent_context_enabled",
                    current_value=False,
                    recommended_value=True,
                    rationale=f"引用缺失占{pct:.0%}的失败，父chunk上下文可提供更多可引用片段",
                    confidence=0.70,
                    auto_apply=True,
                    repair_strategy="expand_chunk_context",
                )
            )

        if pattern.failure_type == "evidence_weak" and pct >= 0.10:
            recommendations.append(
                RetrievalRecommendation(
                    parameter="chunk_size",
                    current_value=current_chunk_size,
                    recommended_value=max(300, current_chunk_size - 100),
                    rationale=f"证据薄弱占{pct:.0%}，减小chunk可提高检索精度",
                    confidence=0.55,
                    auto_apply=False,
                    repair_strategy=None,
                )
            )

        if pattern.failure_type == "context_irrelevant" and pct >= 0.10:
            recommendations.append(
                RetrievalRecommendation(
                    parameter="rewrite_query",
                    current_value=True,
                    recommended_value=True,
                    rationale="上下文不相关，强制开启query改写并增加改写多样性",
                    confidence=0.65,
                    auto_apply=True,
                    repair_strategy="rewrite_query",
                )
            )

    # Deduplicate by parameter, keeping highest confidence.
    seen: set[str] = set()
    deduped: list[RetrievalRecommendation] = []
    for rec in sorted(recommendations, key=lambda r: r.confidence, reverse=True):
        if rec.parameter not in seen:
            seen.add(rec.parameter)
            deduped.append(rec)
    return deduped


def get_feedback_summary_for_prompt(
    *,
    kb_id: int | None = None,
    window_days: int = 14,
) -> str:
    """Return a concise feedback summary suitable for injection into Planner prompts.

    The Supervisor uses this to adapt future plans based on what has been
    failing recently.
    """
    patterns = analyze_recent_failures(kb_id=kb_id, window_days=window_days)
    if not patterns:
        return "近期无验证失败反馈。"

    recs = recommend_parameter_changes(patterns)
    lines = [f"近{window_days}天验证反馈（共{sum(p.count for p in patterns)}条失败）："]
    for p in patterns[:5]:
        lines.append(
            f"  - {p.failure_type}: {p.count}次 ({p.pct_of_total}%), "
            f"平均来源数={p.avg_source_count}, 平均引用数={p.avg_citation_count}"
        )
    if recs:
        lines.append("系统建议的参数调整：")
        for r in recs[:5]:
            auto = "🔧自动应用" if r.auto_apply else "👀建议审查"
            lines.append(
                f"  {auto} {r.parameter}: {r.current_value} → {r.recommended_value} "
                f"({r.rationale}, 置信度={r.confidence:.0%})"
            )
    return "\n".join(lines)

