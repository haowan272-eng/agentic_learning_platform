"""Evaluate pure dense retrieval Hit@K against the Tongchuan testset.

This script is intentionally retrieval-only:
- embeds the user question with the configured BGE model
- searches Qdrant directly
- does not call the answer LLM
- does not use BM25, RRF, query rewrite, or Cross-Encoder rerank

The hit check uses `metadata.source_chunk_ids` from the testset and resolves
those ids through `chunks_manifest.jsonl`, then compares the expected source
text with the retrieved Qdrant payload text.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "evaluation" / "testsets" / "tongchuan" / "ragas_testset_small.jsonl"
DEFAULT_MANIFEST = PROJECT_ROOT / "evaluation" / "testsets" / "tongchuan" / "chunks_manifest.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "evaluation" / "testsets" / "tongchuan" / "dense_hit_at5_eval.jsonl"


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                rows.append(json.loads(raw))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def expected_texts(row: dict[str, Any], manifest: dict[str, str]) -> list[str]:
    source_ids = (row.get("metadata") or {}).get("source_chunk_ids") or []
    return [normalize_text(manifest[source_id]) for source_id in source_ids if manifest.get(source_id)]


def hit_rank(expected: list[str], retrieved: list[dict[str, Any]]) -> tuple[bool, int | None, float]:
    """Return whether any expected text is present in retrieved top-k payloads."""
    best_score = 0.0
    best_rank: int | None = None
    retrieved_texts = [
        normalize_text((item.get("content") or "") + " " + (item.get("parent_content") or ""))
        for item in retrieved
    ]
    for expected_text in expected:
        if not expected_text:
            continue
        probes = [
            expected_text[:80],
            expected_text[:120],
            expected_text[80:200] if len(expected_text) > 120 else expected_text,
        ]
        shingles = [
            expected_text[index:index + 20]
            for index in range(0, max(1, len(expected_text) - 19), 20)
        ]
        for rank, retrieved_text in enumerate(retrieved_texts, start=1):
            contained = any(probe and probe in retrieved_text for probe in probes)
            coverage = sum(1 for shingle in shingles if shingle and shingle in retrieved_text) / max(1, len(shingles))
            score = 1.0 if contained else coverage
            if score > best_score:
                best_score = score
            if contained or coverage >= 0.18:
                best_rank = rank if best_rank is None else min(best_rank, rank)
    return best_rank is not None, best_rank, best_score


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate pure dense Qdrant retrieval Hit@K")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--kb-id", type=int, default=int(os.getenv("RAG_EVAL_KB_ID")) if os.getenv("RAG_EVAL_KB_ID") else None)
    parser.add_argument("--document-id", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-output", action="store_true")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    sys.path.insert(0, str(PROJECT_ROOT))

    from app.rag.embeddings import get_embedder
    from app.rag.vectorstore import get_qdrant_store

    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    manifest_path = args.manifest if args.manifest.is_absolute() else PROJECT_ROOT / args.manifest
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    manifest = {
        item["chunk_id"]: item.get("text", "")
        for item in load_jsonl(manifest_path)
        if item.get("chunk_id")
    }

    embedder = get_embedder()
    qdrant = get_qdrant_store()
    output_rows: list[dict[str, Any]] = []
    hits = 0
    reciprocal_ranks = 0.0
    latencies: list[float] = []

    for index, row in enumerate(rows, start=1):
        question = str(row.get("user_input") or row.get("query") or "")
        started = time.perf_counter()
        query_vector = embedder.encode([question])[0].tolist()
        retrieved = qdrant.search(
            query_vector=query_vector,
            top_k=args.top_k,
            document_id=args.document_id,
            kb_id=args.kb_id,
        )
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        latencies.append(latency_ms)

        expected = expected_texts(row, manifest)
        is_hit, rank, best_score = hit_rank(expected, retrieved)
        if is_hit:
            hits += 1
            reciprocal_ranks += 1.0 / float(rank or args.top_k)

        print(
            f"[{index}/{len(rows)}] {row.get('id', index)} "
            f"hit={is_hit} rank={rank or '-'} latency_ms={latency_ms}"
        )
        output_rows.append({
            "id": row.get("id"),
            "user_input": question,
            "expected_source_chunk_ids": (row.get("metadata") or {}).get("source_chunk_ids") or [],
            "hit": is_hit,
            "hit_rank": rank,
            "best_source_text_overlap": round(best_score, 4),
            "retrieved_sources": retrieved,
            "dense_metadata": {
                "top_k": args.top_k,
                "kb_id": args.kb_id,
                "document_id": args.document_id,
                "latency_ms": latency_ms,
                "mode": "dense_qdrant_only",
            },
        })

    total = len(rows)
    hit_rate = hits / total if total else 0.0
    mrr = reciprocal_ranks / total if total else 0.0
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    print("\nsummary:")
    print(f"rows: {total}")
    print(f"hit@{args.top_k}: {hits}/{total} = {hit_rate:.4f}")
    print(f"mrr@{args.top_k}: {mrr:.4f}")
    print(f"avg_latency_ms: {avg_latency:.2f}")
    print("mode: dense_qdrant_only")

    if not args.no_output:
        write_jsonl(output_path, output_rows)
        print(f"wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
