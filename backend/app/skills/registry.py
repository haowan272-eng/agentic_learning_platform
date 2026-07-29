from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .evolution import (
    builtin_skills_root,
    create_skill_file,
    evolve_skill_file,
    format_skill_stats,
    generated_skills_root,
    record_online_skill_provenance,
    record_skill_feedback,
    record_skill_invocation,
    record_skill_usage_judgments,
    user_skills_root,
)
from .frontmatter import parse_frontmatter


@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    when_to_use: str | None = None
    allowed_tools: list[str] | None = None
    user_invocable: bool = True
    context: str = "inline"
    prompt_template: str = ""
    source: str = "generated"
    skill_dir: str = ""
    version: str = "0.1.0"


_cached_skills: list[SkillDefinition] | None = None


def reset_skill_cache() -> None:
    global _cached_skills
    _cached_skills = None


def discover_skills() -> list[SkillDefinition]:
    global _cached_skills
    if _cached_skills is not None:
        return _cached_skills

    skills: dict[str, SkillDefinition] = {}
    # Precedence: user overrides generated, generated overrides built-in.
    _load_skills_from_dir(builtin_skills_root(), "builtin", skills, overwrite=True)
    _load_skills_from_dir(generated_skills_root(), "generated", skills, overwrite=True)
    _load_skills_from_dir(user_skills_root(), "user", skills, overwrite=True)
    _cached_skills = list(skills.values())
    return _cached_skills


def _load_skills_from_dir(
    base_dir: Path,
    source: str,
    skills: dict[str, SkillDefinition],
    *,
    overwrite: bool,
) -> None:
    if not base_dir.is_dir():
        return
    for entry in base_dir.iterdir():
        if not entry.is_dir():
            continue
        skill_file = entry / "SKILL.md"
        if not skill_file.is_file():
            continue
        skill = _parse_skill_file(skill_file, source, str(entry))
        if skill and (overwrite or skill.name not in skills):
            skills[skill.name] = skill


def _parse_allowed_tools(raw_tools: str) -> list[str] | None:
    if not raw_tools:
        return None
    if raw_tools.startswith("["):
        try:
            value = json.loads(raw_tools)
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
        except Exception:
            pass
        return [part.strip().strip('"\'') for part in raw_tools.strip("[]").split(",") if part.strip()]
    return [part.strip() for part in raw_tools.split(",") if part.strip()]


def _parse_skill_file(file_path: Path, source: str, skill_dir: str) -> SkillDefinition | None:
    try:
        raw = file_path.read_text(encoding="utf-8")
        parsed = parse_frontmatter(raw)
        meta = parsed.meta
        name = meta.get("name") or file_path.parent.name
        return SkillDefinition(
            name=name,
            description=meta.get("description", ""),
            when_to_use=meta.get("when_to_use") or meta.get("when-to-use"),
            allowed_tools=_parse_allowed_tools(meta.get("allowed-tools", "")),
            user_invocable=meta.get("user-invocable", "true").lower() != "false",
            context="fork" if meta.get("context") == "fork" else "inline",
            prompt_template=parsed.body,
            source=source,
            skill_dir=skill_dir,
            version=meta.get("version", "0.1.0"),
        )
    except Exception:
        return None


def get_skill_by_name(skill_name: str) -> SkillDefinition | None:
    wanted = str(skill_name or "").strip()
    for skill in discover_skills():
        if skill.name == wanted:
            return skill
    return None


def resolve_skill_prompt(skill: SkillDefinition, args: object) -> str:
    prompt = skill.prompt_template
    prompt = re.sub(r"\$ARGUMENTS|\$\{ARGUMENTS\}", str(args or ""), prompt)
    prompt = prompt.replace("${SKILL_DIR}", skill.skill_dir)
    prompt = prompt.replace("${CLAUDE_SKILL_DIR}", skill.skill_dir)
    return prompt


def execute_skill(skill_name: str, args: object = "") -> dict[str, Any] | None:
    skill = get_skill_by_name(skill_name)
    if not skill:
        return None
    record_skill_invocation(
        skill_name=skill.name,
        source=skill.source,
        context=skill.context,
        args=args,
    )
    return {
        "prompt": resolve_skill_prompt(skill, args),
        "allowed_tools": skill.allowed_tools,
        "context": skill.context,
        "source": skill.source,
        "skill_dir": skill.skill_dir,
        "version": skill.version,
    }


_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]{1,2}")
_STOP_TOKENS = {
    "help",
    "please",
    "user",
    "assistant",
    "answer",
    "question",
    "task",
    "the",
    "and",
    "or",
    "to",
    "of",
    "in",
    "请帮",
    "帮我",
    "这个",
    "那个",
    "用户",
    "问题",
    "回答",
    "生成",
    "使用",
}


def _token_list(text: str) -> list[str]:
    raw = str(text or "").lower().replace("_", " ").replace("-", " ")
    tokens = [match.group(0) for match in _TOKEN_RE.finditer(raw)]
    for chunk in re.findall(r"[\u4e00-\u9fff]+", raw):
        if len(chunk) >= 2:
            tokens.extend(chunk[index : index + 2] for index in range(len(chunk) - 1))
    expanded: list[str] = []
    for token in tokens:
        if not token.strip() or token in _STOP_TOKENS:
            continue
        expanded.append(token)
        if len(token) > 3 and token.endswith("s"):
            expanded.append(token[:-1])
    return expanded


def retrieve_relevant_skills(
    query: str,
    *,
    limit: int = 3,
    min_score: float = 0.08,
) -> list[dict[str, Any]]:
    query_terms = _token_list(query)
    query_tokens = set(query_terms)
    if not query_tokens:
        return []

    docs: list[tuple[SkillDefinition, list[str]]] = []
    document_frequency: Counter[str] = Counter()
    for skill in discover_skills():
        meta_terms = _token_list(
            "\n".join([skill.name, skill.description, skill.when_to_use or ""])
        )
        body_terms = _token_list(skill.prompt_template[:2500])
        terms = (meta_terms * 3) + body_terms
        if not terms:
            continue
        docs.append((skill, terms))
        document_frequency.update(set(terms))
    if not docs:
        return []

    avg_doc_len = sum(len(terms) for _, terms in docs) / max(1, len(docs))
    doc_count = len(docs)
    k1 = 1.4
    b = 0.75
    hits: list[dict[str, Any]] = []
    for skill, terms in docs:
        term_counts = Counter(terms)
        overlap = query_tokens & set(term_counts)
        if not overlap:
            continue
        raw_score = 0.0
        doc_len = max(1, len(terms))
        for token in overlap:
            tf = term_counts[token]
            idf = math.log(
                1
                + (doc_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            denom = tf + k1 * (1 - b + b * doc_len / max(1.0, avg_doc_len))
            raw_score += idf * (tf * (k1 + 1)) / max(denom, 0.0001)
        name_bonus = 0.15 if skill.name.lower() in str(query or "").lower() else 0.0
        score = min(1.0, (raw_score / max(3.0, len(query_tokens))) + name_bonus)
        if score < float(min_score):
            continue
        hits.append(
            {
                "score": float(score),
                "name": skill.name,
                "description": skill.description,
                "when_to_use": skill.when_to_use or "",
                "source": skill.source,
                "context": skill.context,
                "user_invocable": bool(skill.user_invocable),
                "skill_dir": skill.skill_dir,
                "version": skill.version,
            }
        )
    hits.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return hits[: max(1, int(limit or 1))]


def format_retrieved_skill_context(
    query: str,
    *,
    limit: int = 3,
) -> tuple[str, dict[str, Any] | None]:
    hits = retrieve_relevant_skills(query, limit=limit)
    if not hits:
        return "", None
    lines = [
        "<retrieved_skills>",
        "Use a skill only if it directly matches the user request; otherwise ignore this block.",
    ]
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"{index}. {hit['name']} (score={float(hit['score']):.3f}, "
            f"source={hit['source']}, version={hit.get('version', '')}): "
            f"{hit['description']}"
        )
        if hit.get("when_to_use"):
            lines.append(f"   When to use: {hit['when_to_use']}")
    lines.append("</retrieved_skills>")
    top = dict(hits[0])
    top["all_hits"] = hits
    return "\n".join(lines), top


def build_skill_injection_context(
    query: str,
    *,
    role: str = "planner",
    limit: int = 3,
    expand_limit: int = 2,
    min_score: float = 0.08,
    expand_min_score: float = 0.12,
) -> tuple[str, dict[str, Any] | None]:
    """Build Codex-style progressive Skill context for prompt injection.

    Discovery stays cheap: retrieve a few candidates by metadata/body tokens.
    Only directly relevant Skills are expanded into full instructions, which
    keeps prompts small while still letting the runtime apply durable guidance.
    """
    hits = retrieve_relevant_skills(query, limit=limit, min_score=min_score)
    if not hits:
        return "", None

    expanded: list[dict[str, Any]] = []
    lines = [
        "<skill_context>",
        "The runtime retrieved reusable Skills for this task. Apply a Skill only when its trigger clearly matches the user goal.",
        "Skills are prompt guidance, not tools. Do not claim tool execution because a Skill was injected.",
        "",
        "## Retrieved Skills",
    ]
    for index, hit in enumerate(hits, start=1):
        lines.append(
            f"{index}. {hit['name']} (score={float(hit['score']):.3f}, "
            f"source={hit['source']}, version={hit.get('version', '')}): {hit['description']}"
        )
        if hit.get("when_to_use"):
            lines.append(f"   When to use: {hit['when_to_use']}")

    for hit in hits[: max(0, expand_limit)]:
        if float(hit.get("score", 0.0)) < expand_min_score:
            continue
        payload = execute_skill(str(hit.get("name") or ""), {"query": query, "role": role})
        if not payload:
            continue
        expanded.append({**hit, "expanded": True})
        lines.extend(
            [
                "",
                f"## Expanded Skill: {hit['name']}",
                str(payload.get("prompt") or "").strip()[:6000],
            ]
        )

    if not expanded:
        lines.extend(
            [
                "",
                "No retrieved Skill was expanded because candidate scores were below the expansion threshold.",
            ]
        )
    lines.append("</skill_context>")

    reference = dict(hits[0])
    reference["all_hits"] = hits
    reference["expanded"] = expanded
    return "\n".join(lines), reference


def create_skill(
    name: str,
    description: str,
    instructions: str,
    when_to_use: str = "",
    target: str = "generated",
    context: str = "inline",
    user_invocable: bool = False,
    allowed_tools: object = None,
    evidence: str = "",
    actor: str = "agent",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    result = create_skill_file(
        name=name,
        description=description,
        instructions=instructions,
        when_to_use=when_to_use,
        target=target,
        context=context,
        user_invocable=user_invocable,
        allowed_tools=allowed_tools,
        evidence=evidence,
        actor=actor,
        tags=tags,
    )
    if result.get("ok"):
        reset_skill_cache()
    return result


def evolve_skill(
    skill_name: str,
    lesson: str,
    rationale: str = "",
    target: str = "active",
    instructions: str = "",
    description: str = "",
    when_to_use: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    skill = get_skill_by_name(skill_name)
    result = evolve_skill_file(
        skill_name=skill_name,
        lesson=lesson,
        rationale=rationale,
        target=target,
        active_dir=skill.skill_dir if skill else "",
        instructions=instructions,
        description=description,
        when_to_use=when_to_use,
        tags=tags,
    )
    if result.get("ok"):
        reset_skill_cache()
    return result


def record_online_provenance(
    *,
    action: str,
    skill_name: str = "",
    result: dict[str, Any] | None = None,
    messages: list[dict[str, Any]] | None = None,
    retrieved_reference: dict[str, Any] | None = None,
    decision: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    record_online_skill_provenance(
        action=action,
        skill_name=skill_name,
        result=result,
        messages=messages,
        retrieved_reference=retrieved_reference,
        decision=decision,
        error=error,
    )


def record_feedback(skill_name: str, rating: str, note: str = "") -> None:
    record_skill_feedback(skill_name=skill_name, rating=rating, note=note)


def skill_stats() -> str:
    return format_skill_stats()


def record_usage_judgments(judgments: list[dict[str, Any]]) -> dict[str, Any]:
    result = record_skill_usage_judgments(judgments)
    if result.get("pruned"):
        reset_skill_cache()
    return result
