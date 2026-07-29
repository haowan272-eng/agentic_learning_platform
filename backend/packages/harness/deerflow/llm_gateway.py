"""Role-aware LLM routing with controlled fallback, structured output, and streaming callbacks.

Cache architecture
------------------
Two-tier: fast in-process dict (L1) + Redis shared cache (L2).
L1 avoids deserialization overhead for repeated prompts within the same worker.
L2 shares cached responses across workers, reducing duplicate LLM calls.
Both layers use the same TTL.  When Redis is unavailable the gateway degrades
gracefully to L1-only mode.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

from pydantic import BaseModel

from app.core.redis import get_redis
from app.observability import trace_span

from app.core.config import (
    AGENT_ARCHITECT_MODEL_CHAIN,
    AGENT_JUDGE_MODEL_CHAIN,
    AGENT_LLM_PROVIDER_COOLDOWN_SECONDS,
    AGENT_LLM_PROVIDER_FAILURE_THRESHOLD,
    AGENT_LLM_TIMEOUT_SECONDS,
    AGENT_PLANNER_MODEL_CHAIN,
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL,
)

logger = logging.getLogger(__name__)


LLMRole = Literal["planner", "architect", "judge"]
TokenCallback = Callable[[str, dict[str, str]], None]


class LLMGatewayError(RuntimeError):
    """No eligible provider produced a response for this request."""


class LLMConfigurationError(LLMGatewayError):
    """A configured provider rejected credentials or request parameters."""


@dataclass(frozen=True)
class GenerationPolicy:
    temperature: float
    max_retries: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _Circuit:
    failures: int = 0
    open_until: float = 0.0


@dataclass
class _CacheEntry:
    response: LLMResponse
    created_at: float = field(default_factory=time.monotonic)


ROLE_POLICIES: dict[LLMRole, GenerationPolicy] = {
    "planner": GenerationPolicy(temperature=0.55),
    "architect": GenerationPolicy(temperature=0.30),
    "judge": GenerationPolicy(temperature=0.0),
}


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_text(item) for item in value)
    if isinstance(value, dict):
        candidate = value.get("text") or value.get("content")
        return _text(candidate) if candidate is not None else ""
    return str(value or "")


def _extract_token_usage(response: object) -> tuple[int, int, int]:
    """Best-effort extraction of token counts from LangChain response metadata."""
    try:
        meta = getattr(response, "response_metadata", {}) or {}
        usage = meta.get("token_usage", {}) or meta.get("usage", {}) or {}
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        total = int(usage.get("total_tokens") or prompt + completion)
        return prompt, completion, total
    except Exception:
        return 0, 0, 0


def _retryable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "timeout", "timed out", "connection", "temporarily", "rate limit", "429",
        "500", "502", "503", "504", "service unavailable", "overloaded",
    ))


def _configuration_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in (
        "unauthorized", "authentication", "invalid api key", "incorrect api key", "401",
        "forbidden", "invalid request", "unprocessable", "400",
    ))


def _build_model(provider: str, policy: GenerationPolicy) -> object:
    """Build a LangChain chat model for the given provider and policy."""
    if provider in {"deepseek", "openai"}:
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=DEEPSEEK_API_KEY if provider == "deepseek" else OPENAI_API_KEY,
            base_url=DEEPSEEK_BASE_URL if provider == "deepseek" else OPENAI_BASE_URL,
            model=DEEPSEEK_MODEL if provider == "deepseek" else OPENAI_MODEL,
            temperature=policy.temperature,
            timeout=AGENT_LLM_TIMEOUT_SECONDS,
            max_retries=policy.max_retries,
        )
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            api_key=ANTHROPIC_API_KEY,
            model=ANTHROPIC_MODEL,
            temperature=policy.temperature,
            timeout=AGENT_LLM_TIMEOUT_SECONDS,
            max_retries=policy.max_retries,
        )
    raise LLMConfigurationError(f"Unsupported LLM provider: {provider}")


def _model_name(provider: str) -> str:
    return {"deepseek": DEEPSEEK_MODEL, "openai": OPENAI_MODEL, "anthropic": ANTHROPIC_MODEL}[provider]


def _hash_prompt(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


_REDIS_CACHE_KEY_PREFIX = "llm:cache"


def _redis_cache_key(role: str, prompt_hash: str) -> str:
    return f"{_REDIS_CACHE_KEY_PREFIX}:{role}:{prompt_hash}"


def _response_to_json(response: LLMResponse) -> str:
    return json.dumps({
        "content": response.content,
        "provider": response.provider,
        "model": response.model,
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "total_tokens": response.total_tokens,
    }, ensure_ascii=False)


def _response_from_json(raw: str) -> LLMResponse:
    data = json.loads(raw)
    return LLMResponse(
        content=data["content"],
        provider=data["provider"],
        model=data["model"],
        prompt_tokens=data.get("prompt_tokens", 0),
        completion_tokens=data.get("completion_tokens", 0),
        total_tokens=data.get("total_tokens", 0),
    )


class LLMGateway:
    """Centralizes provider selection so agents never bind to one vendor.

    Supports plain-text invocation, streaming callbacks, structured-output
    (Pydantic model) invocation, and a short-lived cache for deterministic
    (low-temperature) roles.
    """

    def __init__(self) -> None:
        self._circuits: dict[str, _Circuit] = {}
        self._lock = threading.Lock()
        self._cache: dict[str, _CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl_seconds: float = 300.0  # 5-minute TTL for low-temp roles

    # ── circuit breaker ──────────────────────────────────────────────

    @staticmethod
    def _chain(role: LLMRole) -> list[str]:
        raw = {
            "planner": AGENT_PLANNER_MODEL_CHAIN,
            "architect": AGENT_ARCHITECT_MODEL_CHAIN,
            "judge": AGENT_JUDGE_MODEL_CHAIN,
        }[role]
        return [item.strip().lower() for item in raw.split(",") if item.strip()]

    @staticmethod
    def _credentials_present(provider: str) -> bool:
        return {
            "deepseek": bool(DEEPSEEK_API_KEY),
            "openai": bool(OPENAI_API_KEY),
            "anthropic": bool(ANTHROPIC_API_KEY),
        }.get(provider, False)

    def _can_invoke(self, provider: str) -> bool:
        """Return True when the circuit allows requests to *provider*.

        Circuit-breaker terminology (closed = requests flow, open = blocked):
        we return True when the circuit is *closed* (healthy) and False when
        it is *open* (tripped).  A missing circuit entry means the provider
        has never been tried, so we default to closed / healthy.
        """
        with self._lock:
            circuit = self._circuits.get(provider)
            if circuit is None:
                return True
            if circuit.open_until > time.monotonic():
                return False
            # Half-open probe window: reset failure count so a single
            # transient failure does not re-trip the circuit immediately.
            if circuit.open_until:
                circuit.failures = 0
                circuit.open_until = 0.0
            return True

    def _report_success(self, provider: str) -> None:
        with self._lock:
            self._circuits[provider] = _Circuit()

    def _report_failure(self, provider: str) -> None:
        with self._lock:
            circuit = self._circuits.setdefault(provider, _Circuit())
            circuit.failures += 1
            if circuit.failures >= AGENT_LLM_PROVIDER_FAILURE_THRESHOLD:
                circuit.open_until = time.monotonic() + AGENT_LLM_PROVIDER_COOLDOWN_SECONDS

    # ── cache (two-tier: L1 in-process dict + L2 Redis) ────────────────

    def _cache_get(self, cache_key: str, role: str, prompt_hash: str) -> LLMResponse | None:
        """Check L1 (process memory) then L2 (Redis), populating L1 on L2 hit."""
        if self._cache_ttl_seconds <= 0:
            with self._cache_lock:
                self._cache.pop(cache_key, None)
            return None

        # L1: fast in-process lookup (no deserialization overhead).
        with self._cache_lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                if time.monotonic() - entry.created_at < self._cache_ttl_seconds:
                    return entry.response
                del self._cache[cache_key]

        # L2: Redis shared cache for cross-worker reuse.
        redis_key = _redis_cache_key(role, prompt_hash)
        redis_client = get_redis()
        if redis_client is not None:
            try:
                raw = redis_client.get(redis_key)
                if raw is not None:
                    response = _response_from_json(raw)
                    # Populate L1 so subsequent calls in this worker skip Redis.
                    with self._cache_lock:
                        self._cache[cache_key] = _CacheEntry(response=response)
                    return response
            except Exception:
                logger.debug("LLM cache: Redis read failed for key %s", redis_key, exc_info=True)

        return None

    def _cache_set(
        self, cache_key: str, role: str, prompt_hash: str, response: LLMResponse,
    ) -> None:
        """Write to L1 (process memory) and L2 (Redis) concurrently."""
        if self._cache_ttl_seconds <= 0:
            return

        # L1: always fast and available.
        with self._cache_lock:
            self._cache[cache_key] = _CacheEntry(response=response)

        # L2: best-effort Redis write; failures must not break the caller.
        redis_key = _redis_cache_key(role, prompt_hash)
        redis_client = get_redis()
        if redis_client is not None:
            try:
                redis_client.setex(
                    redis_key,
                    int(self._cache_ttl_seconds),
                    _response_to_json(response),
                )
            except Exception:
                logger.debug(
                    "LLM cache: Redis write failed for key %s", redis_key, exc_info=True,
                )

    # ── invocation ────────────────────────────────────────────────────

    def invoke(
        self, *, role: LLMRole, prompt: str,
        on_token: TokenCallback | None = None,
    ) -> LLMResponse:
        """Plain-text (or streaming) LLM invocation with provider fallback."""
        attempted: list[str] = []
        policy = ROLE_POLICIES[role]

        # Low-temperature roles may benefit from short-lived caching.
        if role in {"architect", "judge"} and on_token is None:
            cache_key = f"{role}:{_hash_prompt(prompt)}"
            cached = self._cache_get(cache_key, role, _hash_prompt(prompt))
            if cached is not None:
                return cached

        with trace_span(f"llm.{role}", kind="CLIENT", attrs={"role": role, "streaming": str(on_token is not None)}) as span:
            response = self._invoke_inner(role, prompt, on_token, policy, attempted)
            if span:
                span.set_attribute("provider", response.provider)
                span.set_attribute("model", response.model)
                span.set_attribute("prompt_tokens", response.prompt_tokens)
                span.set_attribute("completion_tokens", response.completion_tokens)
            return response

    def _invoke_inner(
        self, role: LLMRole, prompt: str, on_token: TokenCallback | None,
        policy: GenerationPolicy, attempted: list[str],
    ) -> LLMResponse:
        for provider in self._chain(role):
            if provider not in {"deepseek", "openai", "anthropic"}:
                attempted.append(f"{provider}: unsupported")
                continue
            if not self._credentials_present(provider):
                attempted.append(f"{provider}: not configured")
                continue
            if not self._can_invoke(provider):
                attempted.append(f"{provider}: circuit open")
                continue

            model_name = _model_name(provider)
            try:
                model = _build_model(provider, policy)
                if on_token is None:
                    llm_result = model.invoke(prompt)
                    content = _text(llm_result.content)
                    prompt_tk, completion_tk, total_tk = _extract_token_usage(llm_result)
                else:
                    parts: list[str] = []
                    for chunk in model.stream(prompt):
                        token = _text(getattr(chunk, "content", chunk))
                        if token:
                            parts.append(token)
                            on_token(token, {"provider": provider, "model": model_name, "role": role})
                    content = "".join(parts)
                    prompt_tk, completion_tk, total_tk = 0, 0, 0

                if not content.strip():
                    raise RuntimeError("Provider returned an empty response.")

                self._report_success(provider)
                response = LLMResponse(
                    content=content, provider=provider, model=model_name,
                    prompt_tokens=prompt_tk, completion_tokens=completion_tk,
                    total_tokens=total_tk,
                )
                if role in {"architect", "judge"} and on_token is None:
                    self._cache_set(
                        f"{role}:{_hash_prompt(prompt)}", role, _hash_prompt(prompt), response,
                    )
                return response

            except Exception as exc:  # noqa: BLE001
                attempted.append(f"{provider}: {exc}")
                if _configuration_error(exc):
                    raise LLMConfigurationError(
                        "LLM provider rejected the request: " + str(exc)
                    ) from exc
                if not _retryable(exc):
                    raise LLMGatewayError(
                        "LLM provider failed without a safe fallback condition: " + str(exc)
                    ) from exc
                self._report_failure(provider)

        raise LLMGatewayError(
            "No LLM provider was available. " + " | ".join(attempted)
        )

    def invoke_structured(
        self, *, role: LLMRole, prompt: str, schema: type[BaseModel],
        on_token: TokenCallback | None = None,
    ) -> tuple[BaseModel, LLMResponse]:
        """Invoke LLM with structured output (Pydantic model) via native
        provider tool-calling / JSON-mode, then validate the result.

        Returns (validated_model, raw_response_with_token_counts).
        """
        attempted: list[str] = []
        policy = ROLE_POLICIES[role]

        for provider in self._chain(role):
            if provider not in {"deepseek", "openai", "anthropic"}:
                attempted.append(f"{provider}: unsupported")
                continue
            if not self._credentials_present(provider):
                attempted.append(f"{provider}: not configured")
                continue
            if not self._can_invoke(provider):
                attempted.append(f"{provider}: circuit open")
                continue

            model_name = _model_name(provider)
            try:
                base_model = _build_model(provider, policy)
                structured_model = base_model.with_structured_output(schema)
                llm_result = structured_model.invoke(prompt)

                if isinstance(llm_result, schema):
                    validated: BaseModel = llm_result
                else:
                    # Defensive: if the provider returns a dict instead of
                    # the expected Pydantic instance, validate manually.
                    validated = schema.model_validate(
                        llm_result if isinstance(llm_result, dict) else {"raw": str(llm_result)}
                    )

                prompt_tk, completion_tk, total_tk = _extract_token_usage(llm_result)
                self._report_success(provider)
                response = LLMResponse(
                    content=validated.model_dump_json(),
                    provider=provider, model=model_name,
                    prompt_tokens=prompt_tk, completion_tokens=completion_tk,
                    total_tokens=total_tk,
                )
                return validated, response

            except Exception as exc:  # noqa: BLE001
                attempted.append(f"{provider}: {exc}")
                if _configuration_error(exc):
                    raise LLMConfigurationError(
                        "LLM provider rejected the structured request: " + str(exc)
                    ) from exc
                if not _retryable(exc):
                    raise LLMGatewayError(
                        "Structured LLM call failed without a safe fallback: " + str(exc)
                    ) from exc
                self._report_failure(provider)

        raise LLMGatewayError(
            "No LLM provider was available for structured output. " + " | ".join(attempted)
        )


llm_gateway = LLMGateway()
