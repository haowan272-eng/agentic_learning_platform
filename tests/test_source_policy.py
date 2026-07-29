from __future__ import annotations

from deerflow.planner import ResearchSourceDecision, default_tool_agent_decision, generate_research_source_decision
from deerflow.source_policy import resolve_source_policy


AVAILABLE_TOOLS = [
    {"name": "knowledge.answer", "category": "rag"},
    {"name": "web.search_duckduckgo", "category": "web"},
]


def test_source_policy_defaults_to_auto_and_supports_local_only_intent():
    assert resolve_source_policy({"user_input": "Explain LangGraph"}) == "auto"
    assert resolve_source_policy({"user_input": "仅参考本地知识库解释 LangGraph"}) == "local_only"
    assert resolve_source_policy({"user_input": "Explain LangGraph", "source_policy": "local_only"}) == "local_only"


def test_auto_source_policy_selects_local_and_web_evidence():
    decision = default_tool_agent_decision(
        {"user_input": "Compare LangGraph and AutoGen", "source_policy": "auto"},
        available_tools=AVAILABLE_TOOLS,
    )

    assert [call.tool_name for call in decision.calls] == [
        "knowledge.answer",
        "web.search_duckduckgo",
    ]


def test_local_only_source_policy_excludes_public_web_search():
    decision = default_tool_agent_decision(
        {"user_input": "仅参考本地知识库解释 RAG", "source_policy": "auto"},
        available_tools=AVAILABLE_TOOLS,
    )

    assert [call.tool_name for call in decision.calls] == ["knowledge.answer"]


def test_research_agent_keeps_the_llm_selected_source_in_auto_mode(monkeypatch):
    def fake_invoke(*_args, **_kwargs):
        return ResearchSourceDecision(
            tool_names=["github.search_repositories"],
            reason="Open-source project examples are the best evidence for this query.",
        ), "mock", 0, 0

    monkeypatch.setattr("deerflow.planner._invoke_structured", fake_invoke)
    decision, source, error = generate_research_source_decision(
        {"user_input": "Find multi-agent implementation examples", "source_policy": "auto"},
        query="multi-agent orchestration examples",
        objective="Find implementation references.",
        available_tools=[
            {"name": "knowledge.answer", "category": "rag"},
            {"name": "web.search_duckduckgo", "category": "web"},
            {"name": "github.search_repositories", "category": "github"},
        ],
    )

    assert decision.tool_names == ["github.search_repositories"]
    assert source == "mock"
    assert error is None
