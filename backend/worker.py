"""Celery worker entry point."""
from pathlib import Path
import sys

HARNESS_PATH = Path(__file__).resolve().parent / "packages" / "harness"
if str(HARNESS_PATH) not in sys.path:
    sys.path.insert(0, str(HARNESS_PATH))

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
