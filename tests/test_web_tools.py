from __future__ import annotations

import deerflow.tools as tools
from deerflow.tools import call_tool, list_tools


class FakeDDGS:
    def text(self, query: str, *, max_results: int):
        assert query == "LangGraph multi-agent runtime"
        assert max_results == 2
        return [
            {
                "title": "LangGraph overview",
                "href": "https://example.com/langgraph",
                "body": "A graph runtime for stateful multi-agent workflows.",
            },
            {
                "title": "Duplicate result",
                "href": "https://example.com/langgraph",
                "body": "This duplicate should be removed.",
            },
            {
                "title": "Agent orchestration guide",
                "href": "https://example.com/agents",
                "body": "A guide to orchestrating research agents.",
            },
        ]


def test_duckduckgo_search_is_registered_and_returns_citations(monkeypatch):
    monkeypatch.setattr(tools, "DDGS", FakeDDGS)

    assert "web.search_duckduckgo" in {item["name"] for item in list_tools()}
    result = call_tool(
        "web.search_duckduckgo",
        {"query": "LangGraph multi-agent runtime", "limit": 2},
        agent="research_agent",
    )

    assert result["ok"] is True
    assert [item["url"] for item in result["data"]["results"]] == [
        "https://example.com/langgraph",
        "https://example.com/agents",
    ]
    assert result["citations"][0]["source"] == "duckduckgo"
    assert result["grounding"]["external_source"] == "duckduckgo"


def test_duckduckgo_search_requires_agent_permission():
    result = call_tool(
        "web.search_duckduckgo",
        {"query": "LangGraph multi-agent runtime"},
        agent="untrusted_agent",
    )

    assert result["ok"] is False
    assert result["error"]["type"] == "tool_permission_denied"
