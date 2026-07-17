"""Lightweight Prometheus exposition + optional OpenTelemetry-compatible tracing.

The tracing helpers work with or without the ``opentelemetry-api`` package.
When the package is absent the context-managers are no-ops so production
deployments that don't need distributed tracing incur zero overhead.
"""
from __future__ import annotations

import logging
from collections import Counter, defaultdict
from contextlib import contextmanager
from threading import Lock
from typing import Any, Iterator

logger = logging.getLogger(__name__)

# ── in-process metrics (no external deps) ─────────────────────────────

_lock = Lock()
_counters: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
_latency_total: defaultdict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
_latency_count: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()


def _labels(labels: dict[str, object] | None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))


def increment(name: str, labels: dict[str, object] | None = None, value: int = 1) -> None:
    with _lock:
        _counters[(name, _labels(labels))] += value


def observe_ms(name: str, value: float, labels: dict[str, object] | None = None) -> None:
    key = (name, _labels(labels))
    with _lock:
        _latency_total[key] += max(0.0, float(value))
        _latency_count[key] += 1


def prometheus_text() -> str:
    lines: list[str] = []
    with _lock:
        for (name, labels), value in sorted(_counters.items()):
            suffix = "" if not labels else "{" + ",".join(f'{key}="{val}"' for key, val in labels) + "}"
            lines.append(f"{name}{suffix} {value}")
        for (name, labels), total in sorted(_latency_total.items()):
            suffix = "" if not labels else "{" + ",".join(f'{key}="{val}"' for key, val in labels) + "}"
            lines.append(f"{name}_sum{suffix} {total}")
            lines.append(f"{name}_count{suffix} {_latency_count[(name, labels)]}")
    return "\n".join(lines) + "\n"


# ── optional distributed tracing (works w/ or w/o opentelemetry) ──────

_otel_available: bool | None = None


def _check_otel() -> bool:
    global _otel_available
    if _otel_available is None:
        try:
            import opentelemetry.trace  # noqa: F401
            _otel_available = True
        except ImportError:
            _otel_available = False
            logger.debug("opentelemetry-api not installed; tracing is disabled")
    return _otel_available


@contextmanager
def trace_span(name: str, *, kind: str = "INTERNAL", attrs: dict[str, Any] | None = None) -> Iterator[Any | None]:
    """Create a trace span when OpenTelemetry is available; no-op otherwise.

    Usage::

        with trace_span("tool.knowledge.answer", kind="CLIENT",
                        attrs={"agent": "research_agent"}) as span:
            result = handler(args)
            if span:
                span.set_attribute("ok", result["ok"])
    """
    if not _check_otel():
        yield None
        return

    from opentelemetry import trace
    from opentelemetry.trace import SpanKind, Status, StatusCode

    kind_map = {
        "INTERNAL": SpanKind.INTERNAL,
        "CLIENT": SpanKind.CLIENT,
        "SERVER": SpanKind.SERVER,
        "PRODUCER": SpanKind.PRODUCER,
        "CONSUMER": SpanKind.CONSUMER,
    }
    tracer = trace.get_tracer("interview_improvement_rag")
    span = tracer.start_span(name, kind=kind_map.get(kind, SpanKind.INTERNAL))
    if attrs:
        for key, val in attrs.items():
            span.set_attribute(key, str(val)[:256])

    exc: BaseException | None = None
    try:
        yield span
    except Exception as _e:
        exc = _e
        raise
    finally:
        if exc is not None:
            span.set_status(Status(StatusCode.ERROR, str(exc)[:512]))
        else:
            span.set_status(Status(StatusCode.OK))
        span.end()
