"""Kubernetes/Docker liveness and readiness probes + dependency health dashboard."""
from __future__ import annotations

import time

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import QDRANT_API_KEY, QDRANT_PREFER_GRPC, QDRANT_URL
from app.core.database import engine
from app.core.redis import get_redis

router = APIRouter(tags=["Health"])


def _check_database() -> None:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))


def _check_redis() -> None:
    client = get_redis()
    if client is None or not client.ping():
        raise RuntimeError("Redis client unavailable")


def _check_qdrant() -> None:
    from qdrant_client import QdrantClient

    QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        prefer_grpc=QDRANT_PREFER_GRPC,
        timeout=2,
        trust_env=False,
    ).get_collections()


def _llm_provider_status() -> dict[str, dict[str, object]]:
    """Return circuit-breaker state and credential availability per provider.

    This reads live state from the singleton LLMGateway — no extra DB calls.
    """
    try:
        from app.agent_runtime.llm_gateway import llm_gateway
    except Exception:
        return {}

    providers = ("deepseek", "openai", "anthropic")
    status_map: dict[str, dict[str, object]] = {}
    for provider in providers:
        has_key = llm_gateway._credentials_present(provider)
        can_invoke = llm_gateway._can_invoke(provider)
        with llm_gateway._lock:
            circuit = llm_gateway._circuits.get(provider)
        status_map[provider] = {
            "credentials_configured": has_key,
            "circuit_allows": can_invoke,
            "failures": circuit.failures if circuit else 0,
            "open_until": circuit.open_until if circuit and circuit.open_until else None,
            "available": has_key and can_invoke,
        }
    return status_map


def _feedback_stats() -> dict[str, object]:
    """Return a lightweight summary of recent Verifier feedback."""
    try:
        from app.agent_runtime.feedback import analyze_recent_failures
        patterns = analyze_recent_failures(window_days=7, limit=100)
        total = sum(p.count for p in patterns)
        return {
            "recent_failures_7d": total,
            "top_failure_types": [
                {"type": p.failure_type, "count": p.count, "pct": p.pct_of_total}
                for p in patterns[:5]
            ],
        }
    except Exception:
        return {"recent_failures_7d": "unavailable", "top_failure_types": []}


@router.get("/health/live")
def liveness():
    return {"status": "ok"}


@router.get("/health")
@router.get("/health/ready")
def readiness(response: Response):
    checks = {}
    for name, check in (
        ("postgresql", _check_database),
        ("redis", _check_redis),
        ("qdrant", _check_qdrant),
    ):
        started = time.perf_counter()
        try:
            check()
            checks[name] = {
                "status": "up",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            checks[name] = {
                "status": "down",
                "error": type(exc).__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
    ready = all(item["status"] == "up" for item in checks.values())
    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ready" if ready else "not_ready", "checks": checks}


@router.get("/health/dependencies")
def dependencies_health(response: Response):
    """Aggregated dependency health dashboard.

    Returns infrastructure status (postgresql / redis / qdrant) plus LLM
    provider circuit-breaker state and recent Verifier feedback stats.
    """
    # Infrastructure checks (same as /health/ready).
    infra: dict[str, dict[str, object]] = {}
    for name, check in (
        ("postgresql", _check_database),
        ("redis", _check_redis),
        ("qdrant", _check_qdrant),
    ):
        started = time.perf_counter()
        try:
            check()
            infra[name] = {
                "status": "up",
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }
        except Exception as exc:
            infra[name] = {
                "status": "down",
                "error": type(exc).__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            }

    # LLM provider states.
    llm = _llm_provider_status()

    # Feedback loop stats.
    feedback = _feedback_stats()

    infra_ok = all(v["status"] == "up" for v in infra.values())
    llm_ok = any(v.get("available") for v in llm.values()) if llm else True
    overall = "healthy" if (infra_ok and llm_ok) else "degraded"

    if not infra_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return {
        "status": overall,
        "infrastructure": infra,
        "llm_providers": llm,
        "feedback": feedback,
    }


__all__ = ["router"]
