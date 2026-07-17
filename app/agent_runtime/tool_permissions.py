from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPermission:
    agent: str
    tool: str


ALLOWED_TOOLS_BY_AGENT: dict[str, frozenset[str]] = {
    "context_agent": frozenset({
        "memory.read_profile",
        "memory.read_context",
        "memory.write_event",
        "memory.consolidate",
        "memory.summarize_session",
    }),
    "research_agent": frozenset({"knowledge.answer", "knowledge.repair_retrieval"}),
    # Architect Agent is an LLM subgraph, not a static pseudo-tool.
    "architect_agent": frozenset(),
    "verifier_agent": frozenset({"knowledge.verify_claim"}),
    "executor": frozenset({
        "memory.read_profile",
        "memory.read_context",
        "memory.write_event",
        "memory.consolidate",
        "memory.summarize_session",
        "knowledge.answer",
        "knowledge.repair_retrieval",
        "knowledge.verify_claim",
    }),
}


class ToolPermissionError(PermissionError):
    pass


def assert_tool_allowed(agent: str, tool: str) -> ToolPermission:
    allowed = ALLOWED_TOOLS_BY_AGENT.get(agent, frozenset())
    if tool not in allowed:
        raise ToolPermissionError(f"Agent '{agent}' is not allowed to use tool '{tool}'")
    return ToolPermission(agent=agent, tool=tool)
