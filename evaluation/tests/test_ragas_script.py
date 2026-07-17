import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ragas_evaluate.py"
SPEC = importlib.util.spec_from_file_location("ragas_evaluate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_load_cases_accepts_ground_truth_alias(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps({"question": "refund deadline?", "ground_truth": "seven days"}),
        encoding="utf-8",
    )
    cases = MODULE.load_cases(path)
    assert cases[0].reference == "seven days"


def test_load_cases_rejects_missing_reference(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"question": "refund deadline?"}), encoding="utf-8")
    with pytest.raises(ValueError, match=r"question.*eference"):
        MODULE.load_cases(path)


def test_collect_samples_uses_citation_quotes_as_contexts():
    def handler(request: httpx.Request):
        assert request.headers["Authorization"] == "Bearer token"
        return httpx.Response(
            200,
            json={
                "answer": "Submit refund requests within seven days [1].",
                "conversation_id": 9,
                "degraded": False,
                "citations": [{"quote": "Refund requests must be submitted within seven days."}],
            },
        )

    client = httpx.Client(base_url="http://test", transport=httpx.MockTransport(handler))
    samples = MODULE.collect_samples(
        client,
        [MODULE.EvalCase("1", "How long do I have to request a refund?", "seven days")],
        "token",
        retries=0,
    )
    assert samples[0].retrieved_contexts == ["Refund requests must be submitted within seven days."]
    assert samples[0].conversation_id == 9
