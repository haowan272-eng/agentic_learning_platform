"""Celery应用：Redis Broker/Backend与文档索引队列配置"""
from celery import Celery
from app.logging_config import setup_logging

setup_logging()

from app.core.config import (
    CELERY_BROKER_URL,
    CELERY_AGENT_QUEUE,
    CELERY_DOCUMENT_QUEUE,
    CELERY_RESULT_BACKEND,
    CELERY_RESULT_EXPIRES_SECONDS,
    CELERY_VISIBILITY_TIMEOUT_SECONDS,
)


celery_app = Celery(
    "interview_improvement_rag",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks.document_index", "app.tasks.agent_runtime"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=CELERY_RESULT_EXPIRES_SECONDS,
    timezone="Asia/Shanghai",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_hijack_root_logger=False,
    broker_connection_retry_on_startup=True,
    broker_transport_options={
        "visibility_timeout": CELERY_VISIBILITY_TIMEOUT_SECONDS,
    },
    result_backend_transport_options={
        "visibility_timeout": CELERY_VISIBILITY_TIMEOUT_SECONDS,
    },
    task_default_queue=CELERY_DOCUMENT_QUEUE,
    task_routes={
        "app.tasks.document_index.index_document": {
            "queue": CELERY_DOCUMENT_QUEUE,
        },
        "app.tasks.agent_runtime.run_agent_task": {
            "queue": CELERY_AGENT_QUEUE,
        }
    },
    beat_schedule={
        "dispatch-pending-agent-outbox": {
            "task": "app.tasks.agent_runtime.dispatch_pending_agent_tasks",
            "schedule": 15.0,
        },
    },
)


__all__ = ["celery_app"]
