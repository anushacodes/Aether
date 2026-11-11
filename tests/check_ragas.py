"""Exercise real Ragas metrics with a closed-world fixture and SDK stubs, never a provider.

The fixture's declared blue value is synthetic contract data. Its stubbed scores are not
quality measurements or independently reviewed labels for the project's development corpus.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import boto3
from botocore.stub import ANY, Stubber

from rag.artifacts import save_answer_run
from rag.chunking import chunk_section
from rag.config import PipelineConfig
from rag.eval.answers import AnswerBinding, prepare_answers
from rag.eval.dataset import EvaluationDataset
from rag.eval.ragas_runner import run_ragas_evaluation
from rag.models import AnswerResult, SearchRequest, SearchScope, SectionRecord


def _reply(text: str) -> dict[str, object]:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
        "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        "metrics": {"latencyMs": 1},
    }


def check(root: Path) -> dict[str, object]:
    """Verify all four metric schemas, cross-run cache reuse, and shared call exhaustion."""
    text = "Fixture rule: the retained value is blue."
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    section = SectionRecord.model_validate(
        {
            "section_id": "fixture-rule",
            "source_id": "fixture",
            "corpus_id": "sdk-contract-fixture",
            "heading_path": ["Fixture rule"],
            "raw_start": 0,
            "raw_end": len(text),
            "raw_sha256": digest,
            "raw_text": text,
            "text": text,
            "source_scope": "general-purpose",
            "cleaning_version": "identity-fixture-v1",
            "provenance": {
                "source_id": "fixture",
                "guide": "s3-userguide",
                "revision": "a" * 40,
                "original_filename": "s3-userguide.txt",
                "repository_url": "https://example.com/fixture",
                "publisher_url": "https://example.com/fixture",
                "sha256": digest,
            },
        }
    )
    dataset = EvaluationDataset.model_validate(
        {
            "dataset_id": "synthetic-ragas-contract",
            "corpus_id": section.corpus_id,
            "review_status": "reviewed",
            "note": "Closed-world test fixture, not corpus benchmark labels.",
            "questions": [
                {
                    "question_id": "fixture-001",
                    "query": "Under the fixture rule, what value is retained?",
                    "category": "direct_fact",
                    "permitted_guides": ["s3-userguide"],
                    "source_scope": "general-purpose",
                    "reference_answer": "The retained value is blue.",
                    "reference_passage_ids": ["fixture-support"],
                    "expected_answerability": "answerable",
                    "split": "development",
                    "review_status": "reviewed",
                }
            ],
            "passages": [
                {
                    "passage_id": "fixture-support",
                    "source_id": section.source_id,
                    "section_id": section.section_id,
                    "raw_section_sha256": digest,
                    "cleaning_version": section.cleaning_version,
                    "text_start": 0,
                    "text_end": len(text),
                    "text_sha256": digest,
                }
            ],
        }
    )
    config = PipelineConfig()
    config.cloud.region = "us-east-1"
    config.cloud.development_model_budget_usd = 1  # Required configuration; test never calls AWS.
    config.evaluation.judge_model_id = "synthetic-judge"
    config.evaluation.cache_path = root / "judge.sqlite3"
    chunk = chunk_section(section, config.chunking)[0]
    question = dataset.questions[0]
    request = SearchRequest(
        query=question.query,
        scope=SearchScope(
            permitted_guides=question.permitted_guides,
            corpus_id=dataset.corpus_id,
        ),
    )
    result = AnswerResult(
        answer=f"{question.reference_answer} [{chunk.chunk_id}]",
        citation_ids=[chunk.chunk_id],
        status="answered",
        evidence=[chunk],
        timings_ms={},
    )
    save_answer_run(root / "answer", request, result, config)
    prepared = prepare_answers(
        dataset,
        [
            AnswerBinding(
                question_id=question.question_id,
                answer_path=root / "answer" / "answer.json",
            )
        ],
        [section],
        max_samples=5,
    )
    (root / "dataset.json").write_text(dataset.model_dump_json(), encoding="utf-8")
    (root / "sections.jsonl").write_text(section.model_dump_json() + "\n", encoding="utf-8")
    (root / "bindings.json").write_text(
        json.dumps(
            [
                {
                    "question_id": question.question_id,
                    "answer_path": str(root / "answer" / "answer.json"),
                }
            ]
        ),
        encoding="utf-8",
    )
    arguments = [
        "--dataset",
        str(root / "dataset.json"),
        "--answers",
        str(root / "bindings.json"),
        "--sections",
        str(root / "sections.jsonl"),
    ]
    environment = {**os.environ, "RAG_OPENSEARCH_URL": "http://127.0.0.1:9"}
    environment.pop("RAG_EVALUATOR_MODEL_ID", None)
    command = subprocess.run(
        ["s3-rag", "eval", "prepare-answers", *arguments, "--output", str(root / "cli-prepared")],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert command.returncode == 0, command.stderr
    exported = json.loads((root / "cli-prepared" / "prepared.jsonl").read_text(encoding="utf-8"))
    assert exported["sample"]["retrieved_contexts"] == [text]
    command = subprocess.run(
        ["s3-rag", "eval", "judge", *arguments, "--output", str(root / "missing-judge")],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert command.returncode != 0 and "RAG_EVALUATOR_MODEL_ID" in command.stderr
    draft = dataset.model_copy(update={"review_status": "draft"})
    (root / "dataset.json").write_text(draft.model_dump_json(), encoding="utf-8")
    command = subprocess.run(
        ["s3-rag", "eval", "judge", *arguments, "--output", str(root / "draft-judge")],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
    )
    assert command.returncode != 0 and "human review" in command.stderr
    statement = "The retained value is blue."
    verdict = {"statements": [{"statement": statement, "reason": "Fixture support", "verdict": 1}]}
    responses = [
        {"statements": [statement]},
        verdict,
        {"reason": "Fixture support", "verdict": 1},
        {
            "classifications": [
                {"statement": statement, "reason": "Fixture support", "attributed": 1}
            ]
        },
        {"claims": [statement]},
        verdict,
        {"claims": [statement]},
        verdict,
    ]
    with closing(
        boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
    ) as runtime:
        with Stubber(runtime) as stub:
            for value in responses:
                stub.add_response(
                    "converse",
                    _reply(json.dumps(value)),
                    {
                        "modelId": "synthetic-judge",
                        "system": ANY,
                        "messages": ANY,
                        "inferenceConfig": {"maxTokens": 2048, "temperature": 0},
                    },
                )
            first = run_ragas_evaluation(config, dataset, prepared, root / "first", runtime=runtime)
            if first["metric_outcomes"] != {"scored": 4}:
                print((root / "first" / "results.jsonl").read_text(encoding="utf-8"))
            assert first["metric_outcomes"] == {"scored": 4}, first
            stub.assert_no_pending_responses()
        with Stubber(runtime) as stub:
            repeated = run_ragas_evaluation(
                config, dataset, prepared, root / "repeat", runtime=runtime
            )
            assert repeated["model_calls"] == 0 and repeated["cache_hits"] == 8, repeated
            assert repeated["metric_outcomes"] == {"scored": 4}
            stub.assert_no_pending_responses()
        config.evaluation.cache_path = root / "budget.sqlite3"
        config.evaluation.max_model_calls = 1
        with Stubber(runtime) as stub:
            stub.add_response(
                "converse",
                _reply(json.dumps(responses[0])),
            )
            limited = run_ragas_evaluation(
                config, dataset, prepared, root / "limited", runtime=runtime
            )
            assert limited["model_calls"] == 1 and limited["metric_outcomes"] == {"error": 4}, (
                limited
            )
            stub.assert_no_pending_responses()
        for label, text, expected in (
            ("undefined", json.dumps({"statements": []}), {"undefined": 1, "error": 3}),
            ("invalid-json", "{", {"error": 4}),
        ):
            config.evaluation.cache_path = root / f"{label}.sqlite3"
            with Stubber(runtime) as stub:
                stub.add_response("converse", _reply(text))
                failure = run_ragas_evaluation(
                    config,
                    dataset,
                    prepared,
                    root / label,
                    runtime=runtime,
                )
                assert failure["metric_outcomes"] == expected and failure["model_calls"] == 1
                assert failure["known_input_tokens"] == 10 and failure["usage_complete"]
                stub.assert_no_pending_responses()
        config.evaluation.cache_path = root / "deadline.sqlite3"
        config.evaluation.timeout_seconds = 1
        with (
            patch("rag.eval.ragas_judge.monotonic", return_value=0) as clock,
            Stubber(runtime) as stub,
        ):
            stub.add_response("converse", _reply(json.dumps(responses[0])))
            injected = Mock(wraps=runtime)

            def late_response(**arguments: object) -> dict[str, object]:
                response = runtime.converse(**arguments)
                clock.return_value = 2
                return response

            injected.converse.side_effect = late_response
            late = run_ragas_evaluation(
                config, dataset, prepared, root / "deadline", runtime=injected
            )
            assert late["model_calls"] == 1 and late["metric_outcomes"] == {"error": 4}
            assert late["known_input_tokens"] == 10 and late["usage_complete"]
            rows = [
                json.loads(line)
                for line in (root / "deadline" / "results.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            assert all(row["error_type"] == "TimeoutError" for row in rows)
            stub.assert_no_pending_responses()
    return {
        "synthetic_metric_contract_check": "passed",
        "metrics": 4,
        "first_stub_calls": first["model_calls"],
        "cached_repeat_calls": 0,
        "budget_limited_calls": 1,
        "provider_calls": 0,
        "cli_preparation_and_preflight": "passed",
        "undefined_invalid_output_and_late_usage": "passed",
        "output": str(root),
    }


if __name__ == "__main__":
    with TemporaryDirectory(prefix="aether-ragas-check-") as directory:
        print(json.dumps(check(Path(directory)), indent=2))
