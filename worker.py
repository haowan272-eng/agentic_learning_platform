"""Celery Worker启动入口；也可直接使用celery CLI启动"""
from app.core.celery import celery_app


def main() -> None:
    celery_app.worker_main([
        "worker",
        "--loglevel=INFO",
        "--queues=document_index,agent_runtime",
        "--pool=solo",
        "--concurrency=1",
    ])


if __name__ == "__main__":
    main()
