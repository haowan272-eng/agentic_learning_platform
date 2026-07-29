from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPermission:
    agent: str
    tool: str


ALLOWED_TOOLS_BY_AGENT: dict[str, frozenset[str]] = {
    "research_agent": frozenset({
        "web.search_duckduckgo",
        "github.search_repositories",
        "github.read_readme",
        "knowledge.answer",
        "knowledge.repair_retrieval",
        "architecture.generate_proposal",
        "verification.verify_proposal",
        "planning.repair_research_tasks",
    }),
    "answer_agent": frozenset({
        "web.search_duckduckgo",
        "github.search_repositories",
        "github.read_readme",
        "knowledge.answer",
        "knowledge.repair_retrieval",
        "knowledge.verify_claim",
        "learning.scenario_blueprint",
    }),
}


class ToolPermissionError(PermissionError):
    pass


def assert_tool_allowed(agent: str, tool: str) -> ToolPermission:
    allowed = ALLOWED_TOOLS_BY_AGENT.get(agent, frozenset())
    if tool not in allowed:
        raise ToolPermissionError(f"Agent '{agent}' is not allowed to use tool '{tool}'")
    return ToolPermission(agent=agent, tool=tool)
