"""Run a RAGAS testset through the interview-improvement RAG HTTP API.

This script reads a JSONL testset whose rows contain `user_input` and `reference`,
calls `/embedding/rag/answer` for each question, and writes a JSONL file suitable
for RAGAS evaluation. It preserves the original testset fields and fills:

- response: the RAG generated answer
- retrieved_contexts: full parent contexts returned by the backend
- retrieved_sources: metadata for retrieved contexts
- citations: citations actually used by the generated answer

Examples:
    python evaluation/run_rag_on_testset.py --username alice --password secret --kb-id 1
    $env:RAG_ACCESS_TOKEN="..."; python evaluation/run_rag_on_testset.py --api-base http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    from dotenv import load_dotenv

    load_dotenv(PROJECT_ROOT / ".env")
except Exception:
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

DEFAULT_TESTSET = Path("evaluation/testsets/tongchuan/ragas_testset_small.jsonl")
DEFAULT_OUTPUT = Path("evaluation/testsets/tongchuan/ragas_eval_input.jsonl")


class ApiError(RuntimeError):
    def __init__(self, status: int, message: str, body: Any | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.body = body


def normalize_api_base(value: str) -> str:
    return value.rstrip("/")


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    token: str | None = None,
    timeout: int = 120,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            body = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            body = raw
        detail = body.get("detail") if isinstance(body, dict) else body
        raise ApiError(exc.code, str(detail or exc.reason), body) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach API at {url}: {exc.reason}") from exc


def login(api_base: str, username: str, password: str, timeout: int) -> str:
    payload = {"username": username, "password": password}
    data = request_json("POST", f"{api_base}/login", payload=payload, timeout=timeout)
    token = data.get("access_token")
    if not token:
        raise RuntimeError("Login response did not include access_token")
    return str(token)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("user_input"):
                raise ValueError(f"Row {line_no} missing user_input")
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def call_rag(
    api_base: str,
    token: str,
    question: str,
    top_k: int,
    bm25_weight: float,
    kb_id: int | None,
    document_id: int | None,
    timeout: int,
    rewrite_query: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": question,
        "top_k": top_k,
        "bm25_weight": bm25_weight,
        "rewrite_query": rewrite_query,
    }
    if kb_id is not None:
        payload["kb_id"] = kb_id
    if document_id is not None:
        payload["document_id"] = document_id
    started = time.perf_counter()
    response = request_json(
        "POST",
        f"{api_base}/embedding/rag/answer",
        payload=payload,
        token=token,
        timeout=timeout,
    )
    response["client_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
    return response

def select_ragas_contexts(rag_response: dict[str, Any], mode: str) -> list[str]:
    sources = rag_response.get("retrieved_sources") or []
    citations = rag_response.get("citations") or []
    parent_contexts = rag_response.get("retrieved_contexts") or []
    if mode == "parent":
        contexts = parent_contexts
    elif mode == "citation":
        contexts = [item.get("quote", "") for item in citations]
    else:
        contexts = [item.get("quote", "") for item in sources]
    contexts = [str(text).strip() for text in contexts if str(text or "").strip()]
    if not contexts:
        contexts = [str(text).strip() for text in parent_contexts if str(text or "").strip()]
    if not contexts:
        contexts = [str(item.get("quote", "")).strip() for item in citations if str(item.get("quote", "")).strip()]
    return contexts


def to_eval_row(test_row: dict[str, Any], rag_response: dict[str, Any], context_mode: str) -> dict[str, Any]:
    retrieved_contexts = select_ragas_contexts(rag_response, context_mode)

    row = dict(test_row)
    row["response"] = rag_response.get("answer", "")
    row["retrieved_contexts"] = retrieved_contexts
    row["retrieved_sources"] = rag_response.get("retrieved_sources", [])
    row["citations"] = rag_response.get("citations", [])
    row["rag_metadata"] = {
        "query": rag_response.get("query"),
        "rewritten_query": rag_response.get("rewritten_query"),
        "conversation_id": rag_response.get("conversation_id"),
        "retrieved_count": rag_response.get("retrieved_count"),
        "degraded": rag_response.get("degraded"),
        "context_compacted": rag_response.get("context_compacted"),
        "timings_ms": rag_response.get("timings_ms", {}),
        "client_latency_ms": rag_response.get("client_latency_ms"),
        "ragas_context_mode": context_mode,
    }
    return row

def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a RAGAS testset through the RAG API")
    parser.add_argument("--api-base", default=os.getenv("RAG_API_BASE") or os.getenv("RAG_EVAL_BASE_URL") or "http://127.0.0.1:8001")
    parser.add_argument("--input", type=Path, default=DEFAULT_TESTSET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--username", default=os.getenv("RAG_USERNAME") or os.getenv("RAG_EVAL_USERNAME"))
    parser.add_argument("--password", default=os.getenv("RAG_PASSWORD") or os.getenv("RAG_EVAL_PASSWORD"))
    parser.add_argument("--token", default=os.getenv("RAG_ACCESS_TOKEN") or os.getenv("RAG_EVAL_TOKEN"))
    parser.add_argument("--kb-id", type=int, default=int(os.getenv("RAG_EVAL_KB_ID")) if os.getenv("RAG_EVAL_KB_ID") else None)
    parser.add_argument("--document-id", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=int(os.getenv("RAG_EVAL_TOP_K", "3")) )
    parser.add_argument("--bm25-weight", type=float, default=float(os.getenv("RAG_EVAL_BM25_WEIGHT", "0.4")))
    parser.add_argument("--rewrite-query", action=argparse.BooleanOptionalAction, default=parse_bool(os.getenv("RAG_EVAL_REWRITE_QUERY"), True))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N rows")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between requests")
    parser.add_argument("--resume", action="store_true", help="Skip rows already present in output by id")
    parser.add_argument("--overwrite", action="store_true", help="Start from an empty output file and keep a timestamped backup if it exists")
    parser.add_argument("--ragas-context-mode", choices=["chunk", "parent", "citation"], default=os.getenv("RAG_EVAL_CONTEXT_MODE", "chunk"), help="Which returned text to store as retrieved_contexts for RAGAS")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_base = normalize_api_base(args.api_base)
    input_path = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    output_path = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output

    token = args.token
    if not token:
        if not args.username or not args.password:
            print(
                "Missing auth. Provide --token, RAG_ACCESS_TOKEN, or --username/--password "
                "(also available as RAG_USERNAME/RAG_PASSWORD or RAG_EVAL_USERNAME/RAG_EVAL_PASSWORD).",
                file=sys.stderr,
            )
            return 2
        print(f"Logging in as {args.username} ...")
        token = login(api_base, args.username, args.password, args.timeout)

    rows = load_jsonl(input_path)
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.overwrite and output_path.exists():
        backup_path = output_path.with_name(f"{output_path.stem}.bak_{datetime.now().strftime('%Y%m%d_%H%M%S')}{output_path.suffix}")
        shutil.copy2(output_path, backup_path)
        print(f"Backed up existing output to {backup_path}")

    done_by_id: dict[str, dict[str, Any]] = {}
    if args.resume and not args.overwrite and output_path.exists():
        for row in load_jsonl(output_path):
            row_id = str(row.get("id") or "")
            if row_id:
                done_by_id[row_id] = row

    output_rows: list[dict[str, Any]] = []
    if done_by_id:
        output_rows.extend(done_by_id.values())

    total = len(rows)
    for index, test_row in enumerate(rows, start=1):
        row_id = str(test_row.get("id") or index)
        if row_id in done_by_id:
            print(f"[{index}/{total}] skip {row_id} (already done)")
            continue
        question = str(test_row["user_input"])
        print(f"[{index}/{total}] {row_id}: {question}")
        try:
            rag_response = call_rag(
                api_base=api_base,
                token=token,
                question=question,
                top_k=args.top_k,
                bm25_weight=args.bm25_weight,
                kb_id=args.kb_id,
                document_id=args.document_id,
                timeout=args.timeout,
                rewrite_query=args.rewrite_query,
            )
            eval_row = to_eval_row(test_row, rag_response, args.ragas_context_mode)
        except Exception as exc:
            eval_row = dict(test_row)
            eval_row["response"] = ""
            eval_row["retrieved_contexts"] = []
            eval_row["retrieved_sources"] = []
            eval_row["citations"] = []
            eval_row["rag_error"] = str(exc)
            print(f"  ERROR: {exc}", file=sys.stderr)
        output_rows.append(eval_row)
        write_jsonl(output_path, output_rows)
        if args.sleep:
            time.sleep(args.sleep)

    failures = sum(1 for row in output_rows if row.get("rag_error"))
    print(f"Wrote {len(output_rows)} rows to {output_path}")
    if failures:
        print(f"Completed with {failures} failed rows", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())





