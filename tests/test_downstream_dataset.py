from pathlib import Path

import pytest
from pydantic import ValidationError

from src.evaluation.downstream.dataset import (
    DownstreamExample,
    GoldTarget,
    RetrievedChunk,
    load_examples,
    save_examples,
)


def test_save_and_load_examples_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    examples = [
        DownstreamExample(
            id="ex-1",
            benchmark="hotpotqa",
            split="validation",
            domain="nl",
            task_type="qa",
            query="Who won?",
            retrieved_context=[
                RetrievedChunk(doc_id="doc-1", rank=1, title="Doc 1", text="answer is A")
            ],
            gold=GoldTarget(answer="A"),
            scorer="qa_exact_f1",
            metadata={"source_subset": "distractor"},
        )
    ]

    save_examples(path, examples)
    loaded = load_examples(path)

    assert loaded == examples


def test_downstream_example_rejects_unexpected_fields() -> None:
    with pytest.raises(ValidationError, match="unexpected"):
        DownstreamExample(
            id="ex-1",
            benchmark="hotpotqa",
            split="validation",
            domain="nl",
            task_type="qa",
            query="Who won?",
            retrieved_context=[],
            gold=GoldTarget(answer="A"),
            scorer="qa_exact_f1",
            unexpected="value",
        )


def test_save_examples_serializes_json_safe_metadata(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    examples = [
        DownstreamExample(
            id="ex-1",
            benchmark="hotpotqa",
            split="validation",
            domain="nl",
            task_type="qa",
            query="Who won?",
            retrieved_context=[],
            gold=GoldTarget(answer="A"),
            scorer="qa_exact_f1",
            metadata={"artifact_path": Path("artifacts/doc.txt")},
        )
    ]

    save_examples(path, examples)

    loaded = load_examples(path)

    assert loaded[0].metadata["artifact_path"] == "artifacts/doc.txt"


def test_load_examples_respects_limit(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    examples = [
        DownstreamExample(
            id=f"ex-{index}",
            benchmark="hotpotqa",
            split="validation",
            domain="nl",
            task_type="qa",
            query="Who won?",
            retrieved_context=[],
            gold=GoldTarget(answer="A"),
            scorer="qa_exact_f1",
        )
        for index in range(3)
    ]

    save_examples(path, examples)

    loaded = load_examples(path, limit=2)

    assert [example.id for example in loaded] == ["ex-0", "ex-1"]


def test_load_examples_rejects_negative_limit(tmp_path: Path) -> None:
    path = tmp_path / "sample.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="limit must be non-negative"):
        load_examples(path, limit=-1)


def test_downstream_example_rejects_unsupported_scorer() -> None:
    with pytest.raises(ValidationError, match="qa_exact_f1|code_reference_match"):
        DownstreamExample(
            id="ex-1",
            benchmark="hotpotqa",
            split="validation",
            domain="nl",
            task_type="qa",
            query="Who won?",
            retrieved_context=[],
            gold=GoldTarget(answer="A"),
            scorer="not_supported",
        )
