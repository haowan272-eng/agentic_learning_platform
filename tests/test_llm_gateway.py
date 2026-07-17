"""Unit tests for LLM Gateway circuit breaker, error classification, and caching."""
from __future__ import annotations

import time

import pytest

from app.agent_runtime.llm_gateway import (
    LLMConfigurationError,
    LLMGateway,
    LLMGatewayError,
    LLMResponse,
    _configuration_error,
    _hash_prompt,
    _retryable,
    _text,
)


# ── helpers ────────────────────────────────────────────────────────────


def _make_gateway() -> LLMGateway:
    """Return a fresh gateway instance for each test."""
    return LLMGateway()


# ── _text ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value, expected",
    [
        ("hello", "hello"),
        (["a", "b", "c"], "abc"),
        ({"text": "from dict"}, "from dict"),
        ({"content": "from content"}, "from content"),
        ({"other": 42}, ""),  # dict w/o text/content returns empty
        (None, ""),
    ],
)
def test_text_extraction(value, expected):
    assert _text(value) == expected


# ── _retryable ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "message, expected",
    [
        ("request timed out", True),
        ("connection refused", True),
        ("rate limit exceeded", True),
        ("HTTP 429", True),
        ("service 503 unavailable", True),
        ("server overloaded", True),
        ("invalid api key", False),
        ("model not found", False),
        ("content filtered", False),
    ],
)
def test_retryable_error_classification(message, expected):
    exc = Exception(message)
    assert _retryable(exc) is expected


# ── _configuration_error ───────────────────────────────────────────────


@pytest.mark.parametrize(
    "message, expected",
    [
        ("unauthorized", True),
        ("authentication failed", True),
        ("invalid api key", True),
        ("incorrect api key", True),
        ("401", True),
        ("forbidden", True),
        ("invalid request", True),
        ("400", True),
        ("timeout", False),
        ("server error", False),
    ],
)
def test_configuration_error_classification(message, expected):
    exc = Exception(message)
    assert _configuration_error(exc) is expected


# ── circuit breaker ────────────────────────────────────────────────────


class TestCircuitBreaker:
    """Tests for the circuit breaker behaviour of LLMGateway."""

    def test_initial_state_allows_invocation(self):
        gw = _make_gateway()
        # A provider that has never been tried should be allowed.
        assert gw._can_invoke("deepseek") is True

    def test_success_resets_failures(self):
        gw = _make_gateway()
        gw._report_failure("deepseek")  # failures=1
        gw._report_success("deepseek")   # resets to fresh circuit
        assert gw._can_invoke("deepseek") is True
        # Verify the circuit was reset by checking failures are 0.
        with gw._lock:
            c = gw._circuits.get("deepseek")
        assert c is not None and c.failures == 0

    def test_failure_below_threshold_still_allows(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_FAILURE_THRESHOLD", 3,
        )
        gw = _make_gateway()
        gw._report_failure("deepseek")  # 1
        gw._report_failure("deepseek")  # 2
        assert gw._can_invoke("deepseek") is True

    def test_failure_at_threshold_opens_circuit(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_FAILURE_THRESHOLD", 2,
        )
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_COOLDOWN_SECONDS", 30,
        )
        gw = _make_gateway()
        gw._report_failure("deepseek")  # 1
        gw._report_failure("deepseek")  # 2 → opens circuit
        assert gw._can_invoke("deepseek") is False

    def test_circuit_recloses_after_cooldown(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_FAILURE_THRESHOLD", 2,
        )
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_COOLDOWN_SECONDS", 5,
        )
        gw = _make_gateway()
        # Use a controlled clock so the cooldown window is precise.
        fake_now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)

        gw._report_failure("deepseek")  # failures=1, open_until = fake_now + 5 = 1005
        gw._report_failure("deepseek")  # failures=2, circuit opens
        # Circuit is still open (fake_now < 1005)
        assert gw._can_invoke("deepseek") is False

        # Advance past cooldown.
        fake_now = 1010.0
        # Half-open probe: resets failures.
        assert gw._can_invoke("deepseek") is True
        with gw._lock:
            c = gw._circuits.get("deepseek")
        assert c is not None and c.failures == 0

    def test_half_open_single_failure_does_not_reopen_immediately(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_FAILURE_THRESHOLD", 2,
        )
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_COOLDOWN_SECONDS", 5,
        )
        gw = _make_gateway()
        fake_now = 1000.0
        monkeypatch.setattr(time, "monotonic", lambda: fake_now)

        gw._report_failure("deepseek")
        gw._report_failure("deepseek")  # circuit opens
        assert gw._can_invoke("deepseek") is False

        # Advance past cooldown → half-open probe resets failures.
        fake_now = 1010.0
        assert gw._can_invoke("deepseek") is True
        # One more failure should NOT re-open (failures=1, below threshold of 2)
        gw._report_failure("deepseek")
        assert gw._can_invoke("deepseek") is True

    def test_independent_providers(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_FAILURE_THRESHOLD", 2,
        )
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_LLM_PROVIDER_COOLDOWN_SECONDS", 30,
        )
        gw = _make_gateway()
        gw._report_failure("deepseek")
        gw._report_failure("deepseek")  # deepseek circuit open
        assert gw._can_invoke("deepseek") is False
        assert gw._can_invoke("openai") is True  # unaffected


# ── credentials ────────────────────────────────────────────────────────


class TestCredentials:
    def test_missing_key_returns_false(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.DEEPSEEK_API_KEY", "",
        )
        assert LLMGateway._credentials_present("deepseek") is False

    def test_present_key_returns_true(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.DEEPSEEK_API_KEY", "sk-123",
        )
        assert LLMGateway._credentials_present("deepseek") is True


# ── chain ordering ─────────────────────────────────────────────────────


class TestChainOrdering:
    def test_chain_returns_configured_providers(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_PLANNER_MODEL_CHAIN", "deepseek,openai",
        )
        assert LLMGateway._chain("planner") == ["deepseek", "openai"]

    def test_chain_ignores_whitespace(self, monkeypatch):
        monkeypatch.setattr(
            "app.agent_runtime.llm_gateway.AGENT_ARCHITECT_MODEL_CHAIN",
            "  deepseek , anthropic  ",
        )
        assert LLMGateway._chain("architect") == ["deepseek", "anthropic"]

    def test_judge_uses_zero_temperature(self):
        from app.agent_runtime.llm_gateway import ROLE_POLICIES
        assert ROLE_POLICIES["judge"].temperature == 0.0


# ── cache ──────────────────────────────────────────────────────────────


class TestCache:
    @staticmethod
    def _get(gw: LLMGateway, key: str, role: str = "architect", prompt_hash: str = "abc123") -> LLMResponse | None:
        return gw._cache_get(key, role, prompt_hash)

    @staticmethod
    def _set(gw: LLMGateway, key: str, response: LLMResponse, role: str = "architect", prompt_hash: str = "abc123") -> None:
        gw._cache_set(key, role, prompt_hash, response)

    def test_cache_hit_returns_stored_response(self):
        gw = _make_gateway()
        gw._cache_ttl_seconds = 60.0
        key = "architect:abc123"
        resp = LLMResponse(content="cached", provider="test", model="m")
        self._set(gw, key, resp)
        assert self._get(gw, key) is resp

    def test_cache_miss_returns_none(self):
        gw = _make_gateway()
        assert self._get(gw, "nonexistent") is None

    def test_cache_expires_after_ttl(self):
        gw = _make_gateway()
        gw._cache_ttl_seconds = 0.0  # immediate expiry
        resp = LLMResponse(content="ephemeral", provider="test", model="m")
        self._set(gw, "judge:test", resp, role="judge")
        assert self._get(gw, "judge:test", role="judge") is None

    def test_cache_key_differs_by_role(self):
        gw = _make_gateway()
        gw._cache_ttl_seconds = 60.0
        self._set(gw, "architect:prompt", LLMResponse(content="a", provider="t", model="m"), role="architect")
        self._set(gw, "judge:prompt", LLMResponse(content="b", provider="t", model="m"), role="judge")
        assert self._get(gw, "architect:prompt", role="architect").content == "a"
        assert self._get(gw, "judge:prompt", role="judge").content == "b"


# ── hash ───────────────────────────────────────────────────────────────


def test_hash_prompt_is_deterministic():
    assert _hash_prompt("hello") == _hash_prompt("hello")


def test_hash_prompt_differs_by_content():
    assert _hash_prompt("a") != _hash_prompt("b")


# ── gateway error propagation ──────────────────────────────────────────


class TestErrorPropagation:
    def test_configuration_error_bypasses_fallback(self):
        """A configuration error (401, invalid key) should raise immediately
        rather than attempting the next provider."""
        exc = Exception("unauthorized: invalid api key")
        assert _configuration_error(exc) is True
        assert _retryable(exc) is False

    def test_retryable_error_does_not_raise_configuration(self):
        exc = Exception("connection timeout")
        assert _retryable(exc) is True
        assert _configuration_error(exc) is False
