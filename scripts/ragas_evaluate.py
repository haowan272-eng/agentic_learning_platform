"""Run RAGAS on the prepared Tongchuan RAG evaluation dataset.

This script is intentionally focused on the current evaluation artifact:
  evaluation/testsets/tongchuan/ragas_eval_input.jsonl

Expected JSONL fields per row:
  id, user_input, reference, response, retrieved_contexts
Optional but preserved in output:
  reference_contexts, retrieved_sources, citations, rag_metadata

It does not call the RAG API. Use evaluation/run_rag_on_testset.py first when you
need to generate actual RAG responses.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.embeddings import Embeddings
from ragas.embeddings import HuggingFaceEmbeddings

os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from app.logging_config import setup_logging  # noqa: E402
except ImportError:
    def setup_logging() -> None:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )

setup_logging()
logger = logging.getLogger("ragas_evaluate")

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "testsets" / "tongchuan" / "ragas_eval_input.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "testsets" / "tongchuan" / "ragas_results"
DEFAULT_METRICS = "faithfulness,answer_relevancy,context_precision,context_recall"
EMBEDDING_METRICS = {"answer_relevancy", "response_relevancy"}



class _RagasHuggingFaceEmbeddingsWithQuery(HuggingFaceEmbeddings):
    """Compatibility shim for RAGAS 0.4.x AnswerRelevancy."""

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_texts(texts)

class _LocalSentenceTransformerEmbeddings(Embeddings):
    """LangChain embedding adapter for local sentence-transformers models."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            batch_size=int(os.getenv("RAGAS_LOCAL_EMBEDDING_BATCH_SIZE", "16")),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


@dataclass(frozen=True)
class EvalSample:
    case_id: str
    user_input: str
    reference: str
    response: str
    retrieved_contexts: list[str]
    reference_contexts: list[str] = field(default_factory=list)
    retrieved_sources: list[dict[str, Any]] = field(default_factory=list)
    citations: list[dict[str, Any]] = field(default_factory=list)
    rag_metadata: dict[str, Any] = field(default_factory=dict)
    conversation_id: int | None = None


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    question: str
    reference: str


def _load_project_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
    except Exception:
        pass


def _as_text(value: Any) -> str:
    return str(value or "").strip()


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("expected a list of strings")
    return [str(item).strip() for item in value if str(item or "").strip()]


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def load_cases(path: Path, limit: int | None = None) -> list[EvalCase]:
    cases: list[EvalCase] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc

            question = _as_text(item.get("question") or item.get("user_input"))
            reference = _as_text(item.get("reference") or item.get("ground_truth"))
            missing = [
                name
                for name, value in (
                    ("question", question),
                    ("reference", reference),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"{path}:{line_number} missing question/reference field(s): {', '.join(missing)}")

            cases.append(
                EvalCase(
                    case_id=_as_text(item.get("id") or item.get("case_id") or line_number),
                    question=question,
                    reference=reference,
                )
            )
            if limit and len(cases) >= limit:
                break

    if not cases:
        raise ValueError(f"{path} contains no evaluation cases")
    return cases


def _contexts_from_answer(data: dict[str, Any]) -> list[str]:
    retrieved_contexts = _as_text_list(data.get("retrieved_contexts"))
    if retrieved_contexts:
        return retrieved_contexts

    contexts = []
    for source in _as_dict_list(data.get("retrieved_sources")):
        context = _as_text(source.get("context"))
        if context:
            contexts.append(context)
    if contexts:
        return contexts

    quotes = []
    for citation in _as_dict_list(data.get("citations")):
        quote = _as_text(citation.get("quote"))
        if quote:
            quotes.append(quote)
    return quotes


def collect_samples(
    client: Any,
    cases: list[EvalCase],
    token: str,
    retries: int = 2,
    endpoint: str = "/embedding/rag/answer",
) -> list[EvalSample]:
    samples: list[EvalSample] = []
    headers = {"Authorization": f"Bearer {token}"}
    for case in cases:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = client.post(endpoint, json={"query": case.question}, headers=headers)
                response.raise_for_status()
                data = response.json()
                samples.append(
                    EvalSample(
                        case_id=case.case_id,
                        user_input=case.question,
                        reference=case.reference,
                        response=_as_text(data.get("answer") or data.get("response")),
                        retrieved_contexts=_contexts_from_answer(data),
                        retrieved_sources=_as_dict_list(data.get("retrieved_sources")),
                        citations=_as_dict_list(data.get("citations")),
                        rag_metadata={
                            "degraded": data.get("degraded"),
                            "retrieved_count": data.get("retrieved_count"),
                            "rewritten_query": data.get("rewritten_query"),
                        },
                        conversation_id=data.get("conversation_id") if isinstance(data.get("conversation_id"), int) else None,
                    )
                )
                break
            except Exception as exc:
                last_error = exc
                if attempt >= retries:
                    raise RuntimeError(f"failed to collect RAG sample for case {case.case_id}") from last_error
    return samples


def load_samples(path: Path, limit: int | None = None) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number} is not valid JSON") from exc

            user_input = _as_text(item.get("user_input") or item.get("question"))
            reference = _as_text(item.get("reference") or item.get("ground_truth"))
            response = _as_text(item.get("response") or item.get("answer"))
            try:
                retrieved_contexts = _as_text_list(item.get("retrieved_contexts"))
                reference_contexts = _as_text_list(item.get("reference_contexts"))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} {exc}") from exc

            missing = [
                name
                for name, value in (
                    ("user_input", user_input),
                    ("reference", reference),
                    ("response", response),
                    ("retrieved_contexts", retrieved_contexts),
                )
                if not value
            ]
            if missing:
                raise ValueError(f"{path}:{line_number} missing required field(s): {', '.join(missing)}")

            samples.append(
                EvalSample(
                    case_id=_as_text(item.get("id") or item.get("case_id") or line_number),
                    user_input=user_input,
                    reference=reference,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                    reference_contexts=reference_contexts,
                    retrieved_sources=_as_dict_list(item.get("retrieved_sources")),
                    citations=_as_dict_list(item.get("citations")),
                    rag_metadata=_as_dict(item.get("rag_metadata")),
                )
            )
            if limit and len(samples) >= limit:
                break

    if not samples:
        raise ValueError(f"{path} contains no evaluation samples")
    return samples


def validate_samples(samples: list[EvalSample]) -> dict[str, Any]:
    context_counts = [len(item.retrieved_contexts) for item in samples]
    reference_context_counts = [len(item.reference_contexts) for item in samples]
    return {
        "sample_count": len(samples),
        "empty_retrieved_context_rate": round(sum(count == 0 for count in context_counts) / len(samples), 4),
        "avg_retrieved_contexts": round(sum(context_counts) / len(context_counts), 2),
        "avg_reference_contexts": round(sum(reference_context_counts) / len(reference_context_counts), 2),
        "first_case_id": samples[0].case_id,
        "last_case_id": samples[-1].case_id,
    }


def _build_ragas_components(metric_names: set[str]):
    _load_project_env()
    try:
        from langchain_core.embeddings import Embeddings
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas.embeddings import HuggingFaceEmbeddings, LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import (
            AnswerRelevancy,
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise RuntimeError("Missing evaluation dependencies. Run: uv sync --group evaluation") from exc

    factories = {
        "faithfulness": Faithfulness,
        "answer_relevancy": AnswerRelevancy,
        "context_precision": LLMContextPrecisionWithReference,
        "llm_context_precision_with_reference": LLMContextPrecisionWithReference,
        "context_recall": LLMContextRecall,
        "response_relevancy": ResponseRelevancy,
    }
    unknown = metric_names - factories.keys()
    if unknown:
        raise ValueError(f"Unknown metric(s): {', '.join(sorted(unknown))}")

    llm_key = os.getenv("RAGAS_LLM_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
    llm_base = os.getenv("RAGAS_LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
    llm_model = os.getenv("RAGAS_LLM_MODEL") or os.getenv("DEEPSEEK_MODEL") or "deepseek-chat"
    if not llm_key:
        raise RuntimeError("Set RAGAS_LLM_API_KEY or DEEPSEEK_API_KEY before running RAGAS")

    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            api_key=llm_key,
            base_url=llm_base,
            model=llm_model,
            temperature=0,
            timeout=float(os.getenv("RAGAS_TIMEOUT_SECONDS", "120")),
            max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "2")),
        )
    )

    evaluator_embeddings = None
    if metric_names & EMBEDDING_METRICS:
        provider = os.getenv("RAGAS_EMBEDDING_PROVIDER", "local").strip().lower()
        if provider == "local":
            local_model = os.getenv("RAGAS_LOCAL_EMBEDDING_MODEL") or os.getenv("BGE_MODEL_PATH")
            if not local_model:
                local_model = str(PROJECT_ROOT.parent / "models" / "bge-large-zh-v1.5")
            evaluator_embeddings = _RagasHuggingFaceEmbeddingsWithQuery(
                model=local_model,
                normalize_embeddings=True,
                batch_size=int(os.getenv("RAGAS_LOCAL_EMBEDDING_BATCH_SIZE", "16")),
            )
        elif provider == "openai":
            embedding_key = os.getenv("RAGAS_EMBEDDING_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not embedding_key:
                raise RuntimeError(
                    "answer_relevancy/response_relevancy needs RAGAS_EMBEDDING_API_KEY or OPENAI_API_KEY "
                    "when RAGAS_EMBEDDING_PROVIDER=openai."
                )
            evaluator_embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(
                    api_key=embedding_key,
                    base_url=os.getenv("RAGAS_EMBEDDING_BASE_URL", "https://api.openai.com/v1"),
                    model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
                )
            )
        else:
            raise RuntimeError("RAGAS_EMBEDDING_PROVIDER must be local or openai")

    metrics = []
    for name in sorted(metric_names):
        factory = factories[name]
        if name == "answer_relevancy":
            metrics.append(factory(llm=evaluator_llm, embeddings=evaluator_embeddings, strictness=1))
        elif name in EMBEDDING_METRICS:
            metrics.append(factory(llm=evaluator_llm, embeddings=evaluator_embeddings))
        else:
            metrics.append(factory(llm=evaluator_llm))
    return metrics, evaluator_llm, evaluator_embeddings


def run_ragas(samples: list[EvalSample], metric_names: set[str]):
    from ragas import EvaluationDataset, SingleTurnSample, evaluate

    ragas_samples = []
    for item in samples:
        sample_kwargs = {
            "user_input": item.user_input,
            "retrieved_contexts": item.retrieved_contexts,
            "response": item.response,
            "reference": item.reference,
        }
        if item.reference_contexts:
            sample_kwargs["reference_contexts"] = item.reference_contexts
        ragas_samples.append(SingleTurnSample(**sample_kwargs))

    dataset = EvaluationDataset(samples=ragas_samples)
    metrics, evaluator_llm, evaluator_embeddings = _build_ragas_components(metric_names)
    return evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )


def summarize_scores(frame, samples: list[EvalSample], metrics: set[str]) -> dict[str, Any]:
    metric_summary: dict[str, float] = {}
    for column in frame.columns:
        values = []
        for value in frame[column].tolist():
            if isinstance(value, (int, float)) and not math.isnan(float(value)):
                values.append(float(value))
        if values:
            metric_summary[column] = round(sum(values) / len(values), 4)

    context_counts = [len(item.retrieved_contexts) for item in samples]
    return {
        "sample_count": len(samples),
        "metrics_requested": sorted(metrics),
        "metric_average": metric_summary,
        "empty_context_rate": round(sum(count == 0 for count in context_counts) / len(context_counts), 4),
        "average_retrieved_contexts": round(sum(context_counts) / len(context_counts), 2),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def add_metadata_columns(frame, samples: list[EvalSample]):
    for index, item in enumerate(samples):
        row = asdict(item)
        row["id"] = row.pop("case_id")
        for key, value in row.items():
            encoded = json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
            frame.loc[index, key] = encoded
    return frame


def parse_metrics(raw: str) -> set[str]:
    values = {name.strip() for name in raw.split(",") if name.strip()}
    if not values:
        raise ValueError("--metrics cannot be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RAGAS on prepared Tongchuan RAG responses")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--metrics", default=DEFAULT_METRICS)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true", help="Only validate and summarize the JSONL input")
    return parser.parse_args()


def main() -> int:
    _load_project_env()
    args = parse_args()
    dataset = args.dataset if args.dataset.is_absolute() else PROJECT_ROOT / args.dataset
    output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir

    samples = load_samples(dataset, args.limit)
    validation = validate_samples(samples)
    logger.info("loaded evaluation dataset: %s", validation)

    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    validation_path = output_dir / f"validation_{run_id}.json"
    write_json(validation_path, validation)

    if args.validate_only:
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        print(f"validation saved to: {validation_path}")
        return 0

    metrics = parse_metrics(args.metrics)
    result = run_ragas(samples, metrics)
    frame = result.to_pandas()
    summary = summarize_scores(frame, samples, metrics)
    frame = add_metadata_columns(frame, samples)

    csv_path = output_dir / f"ragas_scores_{run_id}.csv"
    summary_path = output_dir / f"summary_{run_id}.json"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    write_json(summary_path, summary)

    logger.info("RAGAS scores saved to %s", csv_path)
    logger.info("RAGAS summary saved to %s", summary_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"scores saved to: {csv_path}")
    print(f"summary saved to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
