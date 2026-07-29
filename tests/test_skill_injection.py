from __future__ import annotations

from pathlib import Path


def test_skill_progressive_injection_expands_matching_skill(tmp_path, monkeypatch):
    builtin = tmp_path / "skills" / "builtin"
    generated = tmp_path / "skills" / "generated"
    user = tmp_path / "user-skills"
    evolution = tmp_path / "evolution"
    skill_dir = builtin / "resume-project-polish"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: resume-project-polish
description: Improve resume project bullets for Agentic RAG interviews.
version: 0.1.0
when-to-use: Use when the user asks to polish resume project experience.
user-invocable: false
context: inline
---

# Skill Instructions

Rewrite project experience with problem, design, implementation, and outcome.
Mention runtime orchestration and evidence-backed RAG only when supported.
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("SKILL_BUILTIN_DIR", str(builtin))
    monkeypatch.setenv("SKILL_GENERATED_DIR", str(generated))
    monkeypatch.setenv("SKILL_USER_DIR", str(user))
    monkeypatch.setenv("SKILL_EVOLUTION_DIR", str(evolution))

    from app.skills import build_skill_injection_context, reset_skill_cache

    reset_skill_cache()
    context, reference = build_skill_injection_context(
        "帮我润色 Agentic RAG 面试项目经历 resume project",
        role="planner",
    )

    assert "resume-project-polish" in context
    assert "Skill Instructions" in context
    assert reference is not None
    assert reference["expanded"][0]["name"] == "resume-project-polish"
    assert (evolution / "usage.jsonl").exists()


def test_skill_evolution_creates_snapshot_and_bumps_version(tmp_path, monkeypatch):
    generated = tmp_path / "skills" / "generated"
    evolution = tmp_path / "evolution"
    monkeypatch.setenv("SKILL_BUILTIN_DIR", str(tmp_path / "builtin"))
    monkeypatch.setenv("SKILL_GENERATED_DIR", str(generated))
    monkeypatch.setenv("SKILL_USER_DIR", str(tmp_path / "user"))
    monkeypatch.setenv("SKILL_EVOLUTION_DIR", str(evolution))

    from app.skills import create_skill, evolve_skill, reset_skill_cache

    reset_skill_cache()
    created = create_skill(
        name="agent-runtime-review",
        description="Review Agent runtime designs.",
        instructions="# Skill Instructions\n\nFocus on state, routing, fallback, and observability.",
        when_to_use="Use when reviewing Agent runtime architecture.",
    )
    assert created["ok"] is True

    evolved = evolve_skill(
        "agent-runtime-review",
        lesson="Always check whether skills are injected as prompt context instead of modeled as tools.",
        rationale="Skill and Tool have different runtime semantics.",
    )

    assert evolved["ok"] is True
    skill_file = Path(evolved["file"])
    text = skill_file.read_text(encoding="utf-8")
    assert "version: 0.1.1" in text
    assert "Evolution Notes" in text
    assert "instead of modeled as tools" in text
    assert (evolution / "history" / "agent-runtime-review.jsonl").exists()
