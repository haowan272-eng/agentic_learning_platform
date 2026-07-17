from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models.agent_runtime import AgentEvent, AgentStep, AgentToolCall, AgentVerification, SessionSummary


SECTION_TITLES = {
    "goal": "目标",
    "decisions": "已确认决策",
    "progress": "当前进度",
    "tooling": "工具/RAG调用",
    "verification": "验证与修复",
    "errors": "失败与降级",
    "todos": "待办事项",
}


def _json_load(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _add(bucket: dict[str, list[str]], key: str, value: str, *, max_items: int = 8) -> None:
    value = " ".join((value or "").split()).strip()
    if not value or value in bucket[key]:
        return
    bucket[key].append(value)
    if len(bucket[key]) > max_items:
        del bucket[key][0 : len(bucket[key]) - max_items]


def _render(bucket: dict[str, list[str]]) -> str:
    parts = []
    for key in ("goal", "decisions", "progress", "tooling", "verification", "errors", "todos"):
        lines = bucket.get(key) or []
        if lines:
            parts.append(f"【{SECTION_TITLES[key]}】\n" + "\n".join(f"- {line}" for line in lines))
    return "\n\n".join(parts) or "【当前进度】\n- 暂无可压缩 Agent 事件。"


def summarize_session(
    db: Session,
    *,
    user_id: int | None,
    session_id: str | None,
    task_id: str | None = None,
) -> dict[str, Any] | None:
    if user_id is None or not session_id:
        return None

    previous = (
        db.query(SessionSummary)
        .filter(SessionSummary.user_id == int(user_id), SessionSummary.session_id == session_id)
        .order_by(SessionSummary.id.desc())
        .first()
    )
    after_event_id = previous.summary_until_event_id or 0 if previous else 0
    event_query = (
        db.query(AgentEvent)
        .filter(AgentEvent.session_id == session_id, AgentEvent.id > after_event_id)
        .order_by(AgentEvent.id.asc())
    )
    if task_id:
        event_query = event_query.filter(AgentEvent.task_id == task_id)
    events = event_query.limit(120).all()
    if not events:
        return {
            "created": False,
            "summary": previous.summary if previous else "",
            "summary_until_event_id": after_event_id,
        }

    bucket: dict[str, list[str]] = defaultdict(list)
    if previous and previous.summary:
        _add(bucket, "progress", "历史摘要已存在，本次摘要在其基础上追加最新 Agent 运行轨迹。", max_items=4)

    for event in events:
        payload = _json_load(event.payload_json, {})
        if event.event_type in {"task.started", "plan.created"}:
            _add(bucket, "goal", payload.get("goal") or event.message)
        elif event.event_type in {"step.started", "task.completed"}:
            _add(bucket, "progress", event.message)
        elif event.event_type.startswith("tool."):
            _add(bucket, "tooling", f"{event.tool_name or 'tool'}：{event.message}")
        elif event.event_type.startswith("verification."):
            _add(bucket, "verification", event.message)
        elif event.event_type in {"repair.started", "fallback.used", "tool.failed", "task.failed"}:
            _add(bucket, "errors", event.message)
        elif event.event_type == "route.decided":
            _add(bucket, "progress", event.message, max_items=6)

    if task_id:
        failed_steps = (
            db.query(AgentStep)
            .filter(AgentStep.task_id == task_id, AgentStep.status.in_(["failed", "repairing", "fallback_used"]))
            .limit(10)
            .all()
        )
        for step in failed_steps:
            _add(bucket, "errors", f"{step.step_id}: {step.error_type or step.status} {step.error_message or ''}")

        tool_calls = (
            db.query(AgentToolCall)
            .filter(AgentToolCall.task_id == task_id)
            .order_by(AgentToolCall.id.desc())
            .limit(10)
            .all()
        )
        for call in reversed(tool_calls):
            _add(
                bucket,
                "tooling",
                f"{call.tool_name}: ok={bool(call.ok)} latency_ms={call.latency_ms} retry={call.retry_count}",
            )

        verifications = (
            db.query(AgentVerification)
            .filter(AgentVerification.task_id == task_id)
            .order_by(AgentVerification.id.desc())
            .limit(10)
            .all()
        )
        for item in reversed(verifications):
            _add(bucket, "verification", f"{item.step_id or item.target_id}: {item.status} score={item.score}")

    summary = _render(bucket)
    if previous and previous.summary:
        summary = previous.summary.strip() + "\n\n" + summary
        # Keep deterministic bounded summary; no LLM needed here.
        summary = summary[-6000:]

    until_event_id = max(event.id for event in events)
    row = SessionSummary(
        session_id=session_id,
        user_id=int(user_id),
        summary=summary,
        summary_until_event_id=until_event_id,
        metadata_json=json.dumps(
            {
                "task_id": task_id,
                "event_count": len(events),
                "generated_by": "agent_session_summarizer",
            },
            ensure_ascii=False,
        ),
    )
    db.add(row)
    db.flush()
    return {
        "created": True,
        "summary": summary,
        "summary_until_event_id": until_event_id,
        "session_summary_id": row.id,
        "event_count": len(events),
        "generated_at": func.now(),
    }
