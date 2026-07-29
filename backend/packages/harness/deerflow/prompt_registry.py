"""Prompt Registry — DB-backed templates, Jinja2 rendering, A/B testing, few-shot.

Architecture
------------
Templates are stored in ``prompt_templates`` (see ``app.models.prompt``).
At startup the registry loads active templates into an in-memory cache.
Render requests use the cache; a background refresh picks up new versions.

A/B testing
-----------
When multiple *active* versions exist for the same ``name``, the registry
selects a variant probabilistically (weighted by version number — newer
versions get higher weight).  Each invocation is recorded in
``prompt_evaluations`` so operators can compare metrics later.

Few-shot selection
------------------
``PromptExample`` rows can be tagged and optionally indexed in Qdrant for
embedding-based similarity search.  ``pick_examples()`` returns the top-k
most relevant examples for a given query.

Fallback
--------
If the DB is unavailable or no templates are active, the registry falls
back to the module-level ``FALLBACK_TEMPLATES`` (mirrors the old hard-coded
prompts in ``planner.py``).
"""
from __future__ import annotations

import json
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from jinja2 import BaseLoader, Environment, TemplateNotFound, meta
from sqlalchemy import func

from app.core.database import SessionLocal

logger = logging.getLogger(__name__)

# ── types ─────────────────────────────────────────────────────────────

PromptRole = Literal["planner", "architect", "judge"]


@dataclass
class PromptVariant:
    id: int
    name: str
    role: str
    version: int
    template_text: str
    variables_schema: dict[str, Any] | None
    deployment_status: str
    weight: float = 1.0  # for A/B selection


@dataclass
class RenderResult:
    text: str
    variant_id: int
    variant_name: str
    variant_version: int
    source: Literal["db", "fallback"] = "db"


@dataclass
class ABMetric:
    template_name: str
    variant_version: int
    task_id: str
    success: bool
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    retry_count: int
    validation_errors: list[str] | None
    verification_score: float | None


# ── fallback templates (mirror planner.py) ────────────────────────────

FALLBACK_TEMPLATES: dict[str, str] = {
    "supervisor_agent": """\
You are the Supervisor Agent for an interview-improvement learning platform.
Your only job is delegation. Do not answer the user directly and do not execute tools.

User input: {{ user_input }}
Task type: {{ task_type }}
Trusted memory context: {{ memory_context | tojson }}
Has scoped knowledge/document/conversation retrieval: {{ has_rag_scope }}
Existing artifact count: {{ existing_artifact_count }}
Has proposal: {{ has_proposal }}
Allowed child agents: {{ allowed_child_agents | tojson }}
Allowed graph routes: {{ allowed_routes | tojson }}

Delegation rules:
1. Choose child_agents=["answer_agent"], route="answer" for chat, concept questions, knowledge-base questions, document questions, and lightweight current-fact lookups.
2. Choose child_agents=["planner_agent","research_agent"], route="research" for plans, diagnosis, learning improvement, architecture, refactors, multi-step work, approvals, or requests requiring verified evidence.
3. When uncertain, choose route="research". The answer route must never upgrade into research later.
4. Set needs_tools only when answer_agent should use registered retrieval, web, GitHub, or verification tools; supervisor itself never calls tools.
5. Only choose values from the allowed lists and produce data that satisfies the required JSON schema.
6. Only choose values from the allowed lists and produce data that satisfies the required JSON schema.
""",

    "planner_agent": """\
你是面试提优学习系统中的 Supervisor Agent。
你的任务是把用户目标拆成 1 到 4 个彼此独立、可并行执行的知识检索任务。

用户目标：{{ user_input }}
任务类型：{{ task_type }}
可信记忆上下文：{{ memory_context | tojson }}

系统可用工具：{{ available_tools | tojson }}

近期验证失败反馈：{{ feedback_summary }}

要求：
1. 各 query 必须相互补充而非同义改写。
2. 如反馈显示"检索为空"在增长，应增大 top_k。
3. 输出必须能通过给定 JSON 结构校验。""",

    "plan_judge": """\
你是 Planning Judge。在候选 Supervisor 计划中选择最适合的一个。

用户目标：{{ user_input }}
候选计划：{{ candidates | tojson }}

要求：
1. selected_index 必须指向有效下标。
2. 优先选择覆盖完整、任务去重的计划。""",

    "architect_proposal": """\
你是 proposal generation tool。基于 Research Agent 提供的证据产物生成项目改造建议。

用户目标：{{ user_input }}
证据产物：{{ artifacts | tojson }}

要求：
1. 每个关键建议都应关联已有 research artifact。
2. 无证据的建议必须进入 open_questions。""",

    "verifier_decision": """\
你是 LLM-as-Judge verification tool。评估 proposal 方案的证据覆盖度。

proposal：{{ proposal | tojson }}
artifacts：{{ artifacts | tojson }}

要求：
1. 关键建议缺少证据时选择 repair。
2. 无法通过检索修复时选择 fallback。""",

    "answer_agent": """\
你是学习提升平台的 Tool Agent。你的任务是基于工具包和注册表选择必要且最少的工具调用，并把选择交给运行时通过注册表执行。

用户输入：{{ user_input }}
路由决策：{{ route_decision | tojson }}
来源策略：{{ source_policy }}
工具包：{{ tool_package }}
可用工具：{{ available_tools | tojson }}
已有 artifacts：{{ artifacts | tojson }}
已有 proposal：{{ proposal | tojson }}

工具规划原则：
1. 只选择可用工具列表中的工具，不能编造工具。
2. source_policy=auto 时，根据任务自主选择本地 RAG、网页搜索或 GitHub 工具；RAG 为空不能单独判定任务失败。
3. source_policy=local_only 时，只选择 RAG 工具，并将本地 citations 作为必要证据。
4. 检索为空或证据不足时可以选择 knowledge.repair_retrieval。
5. 需要公开网络资料、最新技术信息或外部参考链接时选择 web.search_duckduckgo，并使用返回 citations 标明来源。
6. 记忆上下文由 Runtime 在规划前自动注入，不是 Tool。
7. 已有 proposal 且需要证据校验时可以选择注册表中的 verification 工具，然后 next_action=research_agent。
8. 工具数量越少越好，输出必须通过 JSON 结构校验。""",

    "research_sources": """\
你是 Research Agent 的证据源选择器。根据当前研究子任务，从可用工具中选择最少但足够的证据源；你只做选择，不执行工具，不生成结论。

用户目标：{{ user_input }}
研究 Query：{{ query }}
研究目标：{{ objective }}
来源策略：{{ source_policy }}
可用工具：{{ available_tools | tojson }}

规则：
1. source_policy=auto 时，自主决定是否使用本地 RAG、网页搜索或 GitHub 项目检索；不要因为某个来源为空就把任务判定为失败。
2. source_policy=local_only 时，只能选择 knowledge.answer。
3. 只有需要用户私域材料时选择 knowledge.answer；需要公开事实或时效信息时选择 web.search_duckduckgo；需要开源项目入口时选择 github.search_repositories。
4. 不得编造工具名，输出必须满足 JSON 结构。""",
}


# ── Jinja2 environment ────────────────────────────────────────────────


def _build_jinja_env() -> Environment:
    env = Environment(
        loader=BaseLoader(),
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Register the ``|tojson`` filter (safe JSON dump).
    env.filters["tojson"] = lambda v: json.dumps(v, ensure_ascii=False, default=str)
    return env


_jinja_env = _build_jinja_env()


# ── registry singleton ────────────────────────────────────────────────


class PromptRegistry:
    """Central prompt management: load, render, A/B select, evaluate."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cache: dict[str, list[PromptVariant]] = {}  # name → active variants
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 300.0  # 5 minutes
        self._refresh()

    # ── load ──────────────────────────────────────────────────────

    def _refresh(self) -> None:
        """Reload active templates from DB into memory."""
        try:
            with SessionLocal() as db:
                rows = (
                    db.execute(
                        __import__("sqlalchemy").text(
                            "SELECT id, name, role, version, template_text, "
                            "variables_schema_json, deployment_status "
                            "FROM prompt_templates "
                            "WHERE is_active = 1 AND deployment_status IN ('active', 'staged') "
                            "ORDER BY name, version DESC"
                        )
                    )
                    .mappings()
                    .all()
                )
        except Exception as exc:
            logger.debug("PromptRegistry DB load skipped: %s", exc)
            return

        grouped: dict[str, list[PromptVariant]] = {}
        for row in rows:
            schema_raw = row.get("variables_schema_json")
            variables_schema = None
            if schema_raw:
                try:
                    variables_schema = json.loads(str(schema_raw))
                except json.JSONDecodeError:
                    pass

            variant = PromptVariant(
                id=int(row["id"]),
                name=str(row["name"]),
                role=str(row["role"]),
                version=int(row["version"]),
                template_text=str(row["template_text"]),
                variables_schema=variables_schema,
                deployment_status=str(row["deployment_status"]),
            )
            grouped.setdefault(variant.name, []).append(variant)

        # Assign weights: newer versions get higher weight for A/B.
        for name, variants in grouped.items():
            if len(variants) <= 1:
                if variants:
                    variants[0].weight = 1.0
                continue
            total_ver = sum(v.version for v in variants)
            for v in variants:
                v.weight = v.version / max(1, total_ver)

        with self._lock:
            self._cache = grouped
            self._cache_ts = time.monotonic()

    def _ensure_fresh(self) -> None:
        if time.monotonic() - self._cache_ts > self._cache_ttl:
            self._refresh()

    # ── A/B selection ──────────────────────────────────────────────

    def _select_variant(self, name: str) -> PromptVariant | None:
        """Weighted random selection among active variants for *name*."""
        self._ensure_fresh()
        with self._lock:
            variants = self._cache.get(name, [])
        if not variants:
            return None
        if len(variants) == 1:
            return variants[0]
        # Weighted random selection.
        total = sum(v.weight for v in variants)
        r = random.random() * total
        cumulative = 0.0
        for v in variants:
            cumulative += v.weight
            if r <= cumulative:
                return v
        return variants[-1]

    # ── render ────────────────────────────────────────────────────

    def render(self, name: str, variables: dict[str, Any], *, role: str | None = None) -> RenderResult:
        """Render a prompt template with Jinja2 variables.

        If DB templates are available, uses A/B-weighted selection.
        Otherwise falls back to ``FALLBACK_TEMPLATES``.
        """
        variant = self._select_variant(name)
        if variant is not None:
            try:
                tmpl = _jinja_env.from_string(variant.template_text)
                text = tmpl.render(**variables)
                return RenderResult(
                    text=text,
                    variant_id=variant.id,
                    variant_name=variant.name,
                    variant_version=variant.version,
                    source="db",
                )
            except Exception as exc:
                logger.warning("Prompt render failed for %s v%d: %s", name, variant.version, exc)

        # Fallback to code-based templates.
        fallback_text = FALLBACK_TEMPLATES.get(name)
        if fallback_text is None:
            raise ValueError(f"Prompt template '{name}' not found in DB or fallback registry.")
        try:
            tmpl = _jinja_env.from_string(fallback_text)
            text = tmpl.render(**variables)
        except Exception:
            # Last resort: use Python's str.format on the raw template.
            text = fallback_text.replace("{{ ", "{").replace(" }}", "}").replace("{{", "{").replace("}}", "}")
            text = text.format(**{k: str(v) for k, v in variables.items()})
        return RenderResult(
            text=text,
            variant_id=0,
            variant_name=name,
            variant_version=0,
            source="fallback",
        )

    def list_variables(self, name: str) -> set[str]:
        """Return the set of undeclared variables in a template."""
        self._ensure_fresh()
        template_text: str | None = None
        with self._lock:
            variants = self._cache.get(name, [])
        if variants:
            template_text = variants[0].template_text
        else:
            template_text = FALLBACK_TEMPLATES.get(name)
        if template_text is None:
            return set()
        try:
            ast = _jinja_env.parse(template_text)
            return meta.find_undeclared_variables(ast)
        except Exception:
            return set()

    def list_active_variants(self, name: str) -> list[dict[str, Any]]:
        """Return metadata about active variants (for dashboards)."""
        self._ensure_fresh()
        with self._lock:
            variants = self._cache.get(name, [])
        return [
            {
                "id": v.id,
                "version": v.version,
                "deployment_status": v.deployment_status,
                "weight": round(v.weight, 3),
            }
            for v in variants
        ]

    # ── evaluation ─────────────────────────────────────────────────

    @staticmethod
    def record_evaluation(metric: ABMetric) -> int | None:
        """Persist a prompt invocation metric for A/B analysis."""
        try:
            from app.models.prompt import PromptEvaluation

            with SessionLocal() as db:
                row = PromptEvaluation(
                    template_id=0,  # resolved later if variant tracking is needed
                    template_name=metric.template_name,
                    template_version=metric.variant_version,
                    variant=f"v{metric.variant_version}",
                    task_id=metric.task_id,
                    role="planner",  # override per call site
                    success=1 if metric.success else 0,
                    latency_ms=metric.latency_ms,
                    prompt_tokens=metric.prompt_tokens,
                    completion_tokens=metric.completion_tokens,
                    retry_count=metric.retry_count,
                    validation_errors=json.dumps(metric.validation_errors or [], ensure_ascii=False),
                    verification_score=metric.verification_score,
                )
                db.add(row)
                db.commit()
                return row.id
        except Exception as exc:
            logger.debug("Prompt evaluation record skipped: %s", exc)
            return None

    @staticmethod
    def compare_variants(name: str, *, window_days: int = 14) -> dict[str, Any]:
        """Return A/B comparison stats for variants of *name*."""
        try:
            with SessionLocal() as db:
                since = datetime.now(timezone.utc) - __import__("datetime").timedelta(days=max(1, window_days))
                from app.models.prompt import PromptEvaluation

                rows = (
                    db.query(PromptEvaluation)
                    .filter(
                        PromptEvaluation.template_name == name,
                        PromptEvaluation.created_at >= since,
                    )
                    .all()
                )
        except Exception:
            return {"error": "DB unavailable", "variants": []}

        if not rows:
            return {"variants": [], "total_invocations": 0}

        from collections import defaultdict

        groups: dict[int, list[PromptEvaluation]] = defaultdict(list)
        for r in rows:
            groups[r.template_version].append(r)

        variants = []
        for ver, evals in sorted(groups.items()):
            n = len(evals)
            success_rate = sum(1 for e in evals if e.success) / max(1, n)
            avg_latency = sum(e.latency_ms or 0 for e in evals) / max(1, n)
            avg_verification = (
                sum(e.verification_score or 0 for e in evals) / max(1, n)
            )
            variants.append({
                "version": ver,
                "count": n,
                "success_rate": round(success_rate, 3),
                "avg_latency_ms": round(avg_latency, 1),
                "avg_verification_score": round(avg_verification, 3),
            })

        return {"variants": variants, "total_invocations": len(rows)}

    # ── few-shot ───────────────────────────────────────────────────

    @staticmethod
    def pick_examples(
        template_name: str, query: str, *, top_k: int = 2,
    ) -> list[dict[str, Any]]:
        """Return the top-k most relevant few-shot examples for *query*.

        Tries embedding-based search first (via Qdrant); falls back to
        tag-based filtering + random sampling.
        """
        try:
            from app.models.prompt import PromptExample

            with SessionLocal() as db:
                # Try Qdrant-powered similarity search first.
                examples = _search_examples_via_qdrant(db, template_name, query, top_k)
                if examples:
                    return examples

                # Fallback: filter by tags, pick highest-quality.
                rows = (
                    db.query(PromptExample)
                    .filter(
                        PromptExample.template_name == template_name,
                        PromptExample.is_active == 1,
                    )
                    .order_by(PromptExample.quality_score.desc())
                    .limit(top_k * 3)
                    .all()
                )
                if not rows:
                    return []
                # Simple random sample from top-quality candidates.
                selected = random.sample(rows, min(top_k, len(rows)))
                return [
                    {"input": r.input_text, "output": r.expected_output_json, "quality": r.quality_score}
                    for r in selected
                ]
        except Exception as exc:
            logger.debug("Few-shot example pick skipped: %s", exc)
            return []


def _search_examples_via_qdrant(db, template_name: str, query: str, top_k: int) -> list[dict[str, Any]]:
    """Try to find similar examples via Qdrant embedding search."""
    from app.models.prompt import PromptExample

    try:
        from app.rag.embeddings import get_embedder
        embedder = get_embedder()
        query_vec = embedder.embed_query(query)
    except Exception:
        return []

    try:
        from app.core.config import QDRANT_COLLECTION_NAME, QDRANT_URL, QDRANT_API_KEY
        from qdrant_client import QdrantClient
        from qdrant_client.models import PointStruct

        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=5, trust_env=False)
        collection = f"{QDRANT_COLLECTION_NAME}_prompt_examples"
        # Best-effort: search if collection exists.
        try:
            results = client.search(
                collection_name=collection,
                query_vector=query_vec,
                limit=top_k,
            )
        except Exception:
            return []

        point_ids = [hit.id for hit in results]
        if not point_ids:
            return []

        rows = (
            db.query(PromptExample)
            .filter(
                PromptExample.embedding_id.in_(point_ids),
                PromptExample.is_active == 1,
            )
            .all()
        )
        rows_by_embedding = {str(r.embedding_id): r for r in rows}
        ordered = []
        for pid in point_ids:
            r = rows_by_embedding.get(str(pid))
            if r:
                ordered.append({"input": r.input_text, "output": r.expected_output_json, "quality": r.quality_score})
        return ordered[:top_k]
    except Exception:
        return []


# ── singleton ─────────────────────────────────────────────────────────

prompt_registry = PromptRegistry()
