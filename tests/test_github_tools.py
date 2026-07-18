from __future__ import annotations

from typing import Any

import app.agent_runtime.tools as tools
from app.agent_runtime.tools import call_tool, list_tools


class FakeGitHubResponse:
    def __init__(self, payload: dict[str, Any] | None = None, *, text: str = "", status_code: int = 200) -> None:
        self._payload = payload or {}
        self.text = text
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"GitHub API failed with status {self.status_code}")


def _disable_github_cache(monkeypatch) -> None:
    monkeypatch.setattr(tools, "get_redis", lambda: None)
    monkeypatch.setattr(tools, "GITHUB_TOKEN", "")


def test_github_search_tool_is_registered() -> None:
    tool_names = {item["name"] for item in list_tools()}

    assert "github.search_repositories" in tool_names
    assert "github.read_readme" in tool_names


def test_tool_agent_can_search_github_repositories(monkeypatch) -> None:
    _disable_github_cache(monkeypatch)

    def fake_get(url: str, *, headers: dict[str, str], params: dict[str, Any] | None = None, timeout: float = 15.0):
        assert url == "https://api.github.com/search/repositories"
        assert headers["Accept"] == "application/vnd.github+json"
        assert params == {"q": "langgraph learning agent", "sort": "stars", "order": "desc", "per_page": 1}
        assert timeout == 15.0
        return FakeGitHubResponse({
            "items": [{
                "full_name": "langchain-ai/langgraph",
                "name": "langgraph",
                "owner": {"login": "langchain-ai"},
                "description": "Build resilient language agents as graphs.",
                "html_url": "https://github.com/langchain-ai/langgraph",
                "stargazers_count": 12000,
                "forks_count": 1800,
                "language": "Python",
                "topics": ["agents", "langgraph"],
                "updated_at": "2026-07-01T00:00:00Z",
                "license": {"spdx_id": "MIT"},
            }]
        })

    monkeypatch.setattr(tools.httpx, "get", fake_get)

    result = call_tool(
        "github.search_repositories",
        {"query": "langgraph learning agent", "limit": 1},
        agent="tool_agent",
    )

    assert result["ok"] is True
    assert result["data"]["repositories"][0]["full_name"] == "langchain-ai/langgraph"
    assert result["data"]["repositories"][0]["html_url"] == "https://github.com/langchain-ai/langgraph"
    assert result["data"]["repositories"][0]["stargazers_count"] == 12000
    assert result["grounding"]["external_source"] == "github"
    assert result["citations"][0]["url"] == "https://github.com/langchain-ai/langgraph"


def test_tool_agent_can_read_github_readme(monkeypatch) -> None:
    _disable_github_cache(monkeypatch)

    def fake_get(url: str, *, headers: dict[str, str], params: dict[str, Any] | None = None, timeout: float = 15.0):
        assert url == "https://api.github.com/repos/langchain-ai/langgraph/readme"
        assert headers["Accept"] == "application/vnd.github.raw+json"
        assert params is None
        return FakeGitHubResponse(text="# LangGraph\n\nGraph runtime for agents.")

    monkeypatch.setattr(tools.httpx, "get", fake_get)

    result = call_tool(
        "github.read_readme",
        {"repo": "langchain-ai/langgraph", "max_chars": 1000},
        agent="tool_agent",
    )

    assert result["ok"] is True
    assert result["data"]["readme"].startswith("# LangGraph")
    assert result["grounding"]["repo"] == "langchain-ai/langgraph"


def test_verifier_agent_cannot_call_github_tools() -> None:
    result = call_tool("github.read_readme", {"repo": "langchain-ai/langgraph"}, agent="verifier_agent")

    assert result["ok"] is False
    assert result["error"]["type"] == "tool_permission_denied"


def test_github_search_failure_returns_structured_tool_result(monkeypatch) -> None:
    _disable_github_cache(monkeypatch)

    def fake_get(url: str, *, headers: dict[str, str], params: dict[str, Any] | None = None, timeout: float = 15.0):
        return FakeGitHubResponse(status_code=500)

    monkeypatch.setattr(tools.httpx, "get", fake_get)

    result = call_tool("github.search_repositories", {"query": "langgraph", "limit": 1}, agent="tool_agent")

    assert result["ok"] is False
    assert result["data"]["repositories"] == []
    assert result["error"]["type"] == "github_search_failed"
