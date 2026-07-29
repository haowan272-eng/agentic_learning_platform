"""Client boundary for untrusted execution; never run sandboxed code in workers."""
from __future__ import annotations

from typing import Any

import httpx

from app.core.config import AGENT_SANDBOX_TOKEN, AGENT_SANDBOX_URL, AGENT_SANDBOX_TIMEOUT_SECONDS


class SandboxUnavailable(RuntimeError):
    pass


def execute_in_sandbox(payload: dict[str, Any]) -> dict[str, Any]:
    """Delegate execution to a separately deployed, resource-limited service.

    The worker never falls back to local subprocess execution.  The sandbox
    service is responsible for a disposable container, no host mounts/secrets,
    CPU/memory limits, egress policy and an execution deadline.
    """
    if not AGENT_SANDBOX_URL:
        raise SandboxUnavailable("AGENT_SANDBOX_URL is not configured; sandboxed execution is disabled.")
    headers = {"Authorization": f"Bearer {AGENT_SANDBOX_TOKEN}"} if AGENT_SANDBOX_TOKEN else {}
    with httpx.Client(timeout=AGENT_SANDBOX_TIMEOUT_SECONDS) as client:
        response = client.post(f"{AGENT_SANDBOX_URL.rstrip('/')}/execute", json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Sandbox returned an invalid response.")
    return data
