"""Analyze per-stage RAG timings saved by run_rag_on_testset.py."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "testsets" / "tongchuan" / "ragas_eval_input.jsonl"


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * pct)))
    return ordered[index]


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze RAG timing metadata")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--top", type=int, default=8)
    args = parser.parse_args()

    path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    rows = load_rows(path)
    if not rows:
        raise SystemExit(f"No rows found in {path}")

    stage_values: dict[str, list[float]] = {}
    row_summaries = []
    rewritten_count = 0
    for row in rows:
        meta = row.get("rag_metadata") or {}
        timings = meta.get("timings_ms") or {}
        if meta.get("rewritten_query"):
            rewritten_count += 1
        if meta.get("client_latency_ms") is not None:
            timings = {**timings, "client_latency_ms": meta.get("client_latency_ms")}
        for key, value in timings.items():
            if isinstance(value, (int, float)):
                stage_values.setdefault(key, []).append(float(value))
        row_summaries.append({
            "id": row.get("id"),
            "question": row.get("user_input"),
            "total_ms": float(timings.get("total_ms") or timings.get("client_latency_ms") or 0),
            "client_latency_ms": float(timings.get("client_latency_ms") or 0),
            "timings": timings,
            "rewritten_query": meta.get("rewritten_query"),
        })

    print(f"rows: {len(rows)}")
    print(f"rewritten_query_nonempty: {rewritten_count}")
    print("\nstage timing ms:")
    for key in sorted(stage_values, key=lambda name: mean(stage_values[name]), reverse=True):
        values = stage_values[key]
        print(
            f"  {key:20s} avg={mean(values):8.2f} "
            f"p50={percentile(values, 0.50):8.2f} "
            f"p95={percentile(values, 0.95):8.2f} "
            f"max={max(values):8.2f}"
        )

    print(f"\nslowest {args.top} rows:")
    for item in sorted(row_summaries, key=lambda x: x["total_ms"], reverse=True)[: args.top]:
        print(f"- {item['id']} total={item['total_ms']:.2f} client={item['client_latency_ms']:.2f} {item['question']}")
        timings = item["timings"]
        detail = ", ".join(f"{k}={v}" for k, v in sorted(timings.items()))
        print(f"  {detail}")
        if item.get("rewritten_query"):
            print(f"  rewritten: {item['rewritten_query']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())