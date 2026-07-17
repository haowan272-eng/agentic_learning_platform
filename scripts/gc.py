from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.database import SessionLocal
from app.services.gc_service import run_global_gc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run global RAG garbage collection.")
    parser.add_argument("--execute", action="store_true", help="Actually delete garbage. Defaults to dry-run.")
    parser.add_argument("--failed-days", type=int, default=7, help="Delete failed documents older than this many days.")
    parser.add_argument("--stuck-hours", type=int, default=24, help="Delete uploaded/indexing documents older than this many hours.")
    parser.add_argument("--skip-files", action="store_true", help="Skip orphan upload files and assets.")
    parser.add_argument("--skip-qdrant", action="store_true", help="Skip orphan Qdrant vector points.")
    parser.add_argument("--skip-stale-docs", action="store_true", help="Skip failed/stuck document cleanup.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    db = SessionLocal()
    try:
        report = run_global_gc(
            db,
            execute=args.execute,
            failed_days=args.failed_days,
            stuck_hours=args.stuck_hours,
            skip_files=args.skip_files,
            skip_qdrant=args.skip_qdrant,
            skip_stale_docs=args.skip_stale_docs,
        )
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if not report.errors else 1
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
