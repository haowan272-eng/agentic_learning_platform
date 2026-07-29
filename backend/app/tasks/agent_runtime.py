"""Agent Runtime Celery 任务。

两个 Celery 任务：
- run_agent_task_task:     执行单个 Agent 任务（幂等租约 + 重试）
- dispatch_pending_agent_tasks: 周期性扫描 outbox 并投递待处理任务（beat schedule）
"""
from __future__ import annotations

from uuid import uuid4

from deerflow.runtime import AgentBudgetExceeded, AgentTaskCancelled, run_agent_task
from deerflow.store import agent_store
from app.core.celery import celery_app
from app.core.config import AGENT_TASK_LEASE_SECONDS, AGENT_TASK_MAX_RETRIES


def _build_initial_state(task: dict) -> dict:
    saved = dict(task.get("state") or {})
    trusted = {
        "session_id": task["session_id"],
        "task_id": task["task_id"],
        "run_id": task["run_id"],
        "username": task["username"],
        "user_id": task.get("user_id"),
        "user_input": task["user_input"],
        "task_type": task["task_type"],
        "kb_id": task.get("kb_id"),
        "document_id": task.get("document_id"),
        "conversation_id": task.get("conversation_id"),
        "status": "pending",
        "budget": task.get("budget") or {},
    }
    # Only runtime progress is resumed. Identity and data scope are rebuilt
    # from the authoritative database task to prevent checkpoint tampering.
    return {**saved, **trusted}


def _append_runtime_failure_event(task: dict, exc: Exception, *, event_type: str) -> None:
    agent_store.append_event(
        {
            "session_id": task["session_id"],
            "task_id": task["task_id"],
            "run_id": task["run_id"],
            "event_type": event_type,
            "event_index": len(agent_store.list_events(task["task_id"])) + 1,
            "agent_name": "runtime",
            "message": str(exc),
            "payload": {"error_type": type(exc).__name__},
        }
    )


@celery_app.task(
    bind=True,
    name="app.tasks.agent_runtime.run_agent_task",
    acks_late=True,
    reject_on_worker_lost=True,
    max_retries=AGENT_TASK_MAX_RETRIES,
)
def run_agent_task_task(self, task_id: str) -> dict:
    owner = f"celery:{uuid4()}"
    task = agent_store.acquire_task_lease(task_id, owner, AGENT_TASK_LEASE_SECONDS)
    if not task:
        return {"status": "skipped", "task_id": task_id, "reason": "missing, terminal, cancelled, or already leased"}

    state = _build_initial_state(task)
    attempt = int(self.request.retries or 0)
    try:
        resume_payload = task.get("resume_payload")
        final_state = run_agent_task(state, agent_store, resume_payload=resume_payload)
        if resume_payload is not None:
            agent_store.clear_resume_payload(task_id)
        agent_store.release_task_lease(task_id, owner)
        return {
            "status": final_state.get("status", "completed"),
            "task_id": task_id,
            "run_id": task.get("run_id"),
        }
    except AgentTaskCancelled as exc:
        state["status"] = "cancelled"
        state["final_answer"] = "Task cancelled by user."
        agent_store.save_task_state(task_id, state)
        agent_store.release_task_lease(task_id, owner, error=str(exc))
        _append_runtime_failure_event(task, exc, event_type="task.cancelled")
        return {"status": "cancelled", "task_id": task_id}
    except AgentBudgetExceeded as exc:
        state["status"] = "failed"
        state["final_answer"] = f"Task stopped by execution budget: {exc}"
        agent_store.save_task_state(task_id, state)
        agent_store.release_task_lease(task_id, owner, error=str(exc))
        _append_runtime_failure_event(task, exc, event_type="task.budget_exceeded")
        return {"status": "failed", "task_id": task_id, "error": str(exc)}
    except Exception as exc:
        agent_store.release_task_lease(task_id, owner, error=str(exc))
        if attempt < AGENT_TASK_MAX_RETRIES:
            _append_runtime_failure_event(task, exc, event_type="task.retrying")
            raise self.retry(exc=exc, countdown=min(60, max(5, 2 ** attempt)))

        state["status"] = "failed"
        state["final_answer"] = f"Agent task failed: {exc}"
        agent_store.save_task_state(task_id, state)
        _append_runtime_failure_event(task, exc, event_type="task.failed")
        raise


@celery_app.task(name="app.tasks.agent_runtime.dispatch_pending_agent_tasks")
def dispatch_pending_agent_tasks(limit: int = 100) -> dict:
    """Publish durable outbox rows; safe to run periodically or after recovery."""
    dispatched = 0
    failed = 0
    for row in agent_store.pending_outbox(limit=limit):
        try:
            run_agent_task_task.delay(row["task_id"])
            agent_store.mark_outbox_dispatched(row["id"])
            dispatched += 1
        except Exception as exc:  # noqa: BLE001
            agent_store.mark_outbox_failed(row["id"], str(exc))
            failed += 1
    return {"dispatched": dispatched, "failed": failed}


__all__ = ["run_agent_task_task", "dispatch_pending_agent_tasks"]
