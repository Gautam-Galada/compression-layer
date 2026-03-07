import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts import evaluate_downstream
from scripts.evaluate_downstream import TaskModelClientAdapter, build_parser, main
from src.evaluation.downstream.dataset import DownstreamExample, GoldTarget, RetrievedChunk


def _example(example_id: str, answer: str) -> DownstreamExample:
    return DownstreamExample(
        id=example_id,
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query=f"Who won for {example_id}?",
        retrieved_context=[
            RetrievedChunk(doc_id=f"doc-{example_id}", rank=1, title="Doc 1", text=answer)
        ],
        gold=GoldTarget(answer=answer),
        scorer="qa_exact_f1",
    )


class FakeAsyncClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def complete(self, prompt: str) -> dict[str, float | int | str]:
        self.prompts.append(prompt)
        if "Alice" in prompt:
            return {
                "text": "Alice",
                "input_tokens": 12,
                "output_tokens": 2,
                "latency_ms": 4.0,
            }
        return {
            "text": "Bob",
            "input_tokens": 12,
            "output_tokens": 2,
            "latency_ms": 4.0,
        }


def _append_result_rows(path: Path, results: list[dict[str, object]]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def _result_row(
    example_id: str,
    *,
    answer: Any = "Alice",
    cost: Any = 0.1,
    compressor: Any = "identity",
    task_model: Any = "gpt-4o-mini",
    dataset: Any = "data/eval/downstream/hotpotqa_validation_10.jsonl",
    **extra: object,
) -> dict[str, object]:
    row = {
        "dataset": dataset,
        "example_id": example_id,
        "benchmark": "hotpotqa",
        "compressor": compressor,
        "task_model": task_model,
        "full_output": answer,
        "compressed_output": answer,
        "full_exact_match": 1.0,
        "compressed_exact_match": 1.0,
        "delta_exact_match": 0.0,
        "full_f1": 1.0,
        "compressed_f1": 1.0,
        "delta_f1": 0.0,
        "context_tokens_full": 1,
        "context_tokens_compressed": 1,
        "compression_ratio": 1.0,
        "latency_ms_full": 1.0,
        "latency_ms_compressed": 1.0,
        "cost_usd_full": cost,
        "cost_usd_compressed": 0.0,
    }
    row.update(extra)
    return row


@pytest.mark.asyncio
async def test_task_model_client_adapter_uses_public_model_client_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModelClient:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        async def complete_response(
            self,
            prompt: str,
            max_tokens: int = 1024,
            temperature: float = 0.0,
            use_cache: bool = True,
        ) -> dict[str, object]:
            return {
                "text": "Alice",
                "input_tokens": 3,
                "output_tokens": 2,
                "model": "gpt-4o-mini",
            }

        async def _call_with_retry(
            self, prompt: str, max_tokens: int, temperature: float
        ) -> object:
            raise AssertionError("private retry path should not be used")

    monkeypatch.setattr(evaluate_downstream, "ModelClient", FakeModelClient)
    monkeypatch.setattr(
        evaluate_downstream, "model_type_from_string", lambda model_name: model_name
    )

    client = TaskModelClientAdapter("gpt-4o-mini")
    response = await client.complete("Hello")

    assert response["text"] == "Alice"
    assert response["input_tokens"] == 3


def test_estimate_example_cost_upper_bound_uses_model_pricing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_downstream, "count_tokens", lambda text: 100)

    gpt_estimate = evaluate_downstream.estimate_example_cost_upper_bound(
        _example("ex-1", "Alice"),
        compressor="identity",
        task_model="gpt-4o-mini",
        truncate_tokens_limit=256,
        extractive_chars=1000,
    )
    claude_estimate = evaluate_downstream.estimate_example_cost_upper_bound(
        _example("ex-1", "Alice"),
        compressor="identity",
        task_model="claude-sonnet-4-20250514",
        truncate_tokens_limit=256,
        extractive_chars=1000,
    )

    assert gpt_estimate == pytest.approx(0.0012588)
    assert claude_estimate == pytest.approx(0.03132)
    assert claude_estimate > gpt_estimate


def test_estimate_example_cost_upper_bound_uses_runner_pricing_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(evaluate_downstream, "count_tokens", lambda text: 100)
    monkeypatch.setattr(
        evaluate_downstream,
        "_estimate_cost_from_tokens",
        lambda tokens, model, *, is_input: 7.0 if is_input else 11.0,
    )

    estimate = evaluate_downstream.estimate_example_cost_upper_bound(
        _example("ex-1", "Alice"),
        compressor="identity",
        task_model="gpt-4o-mini",
        truncate_tokens_limit=256,
        extractive_chars=1000,
    )

    assert estimate == 18.0


def test_estimate_example_cost_upper_bound_for_adapter_backend_skips_adapter_compression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingAdapter:
        def compress(self, context: str) -> str:
            raise AssertionError("adapter compression should not run during cost estimation")

    monkeypatch.setattr(evaluate_downstream, "count_tokens", lambda text: 100)

    estimate = evaluate_downstream.estimate_example_cost_upper_bound(
        _example("ex-1", "Alice"),
        compressor="adapter_local",
        task_model="gpt-4o-mini",
        truncate_tokens_limit=256,
        extractive_chars=1000,
        adapter=ExplodingAdapter(),
    )

    assert estimate == pytest.approx(0.0012588)


def test_evaluate_downstream_cli_parses_optional_limit_and_max_cost() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_tinker",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            "models/eval/downstream_hotpotqa_identity_gpt4omini.jsonl",
            "--limit",
            "3",
            "--max-cost",
            "1.5",
            "--summary-output",
            "models/eval/downstream_hotpotqa_identity_gpt4omini.summary.json",
            "--truncate-tokens",
            "128",
            "--extractive-chars",
            "512",
            "--adapter-path",
            "models/adapters/mlx/latest",
            "--adapter-model",
            "mlx-community/Qwen3-4B-Instruct-4bit",
            "--checkpoint-path",
            "models/adapters/tinker/checkpoint",
        ]
    )

    assert args.limit == 3
    assert args.max_cost == 1.5
    assert args.compressor == "adapter_tinker"
    assert args.summary_output == Path(
        "models/eval/downstream_hotpotqa_identity_gpt4omini.summary.json"
    )
    assert args.truncate_tokens == 128
    assert args.extractive_chars == 512
    assert args.adapter_path == Path("models/adapters/mlx/latest")
    assert args.adapter_model == "mlx-community/Qwen3-4B-Instruct-4bit"
    assert args.checkpoint_path == "models/adapters/tinker/checkpoint"


def test_evaluate_downstream_cli_rejects_unknown_compressor() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--dataset",
                "data/eval/downstream/hotpotqa_validation_10.jsonl",
                "--compressor",
                "not_real",
                "--task-model",
                "gpt-4o-mini",
                "--output",
                "models/eval/downstream_hotpotqa_identity_gpt4omini.jsonl",
            ]
        )

    assert exc_info.value.code == 2


def test_evaluate_downstream_cli_rejects_negative_limit_in_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--dataset",
                "data/eval/downstream/hotpotqa_validation_10.jsonl",
                "--compressor",
                "identity",
                "--task-model",
                "gpt-4o-mini",
                "--output",
                "models/eval/downstream_hotpotqa_identity_gpt4omini.jsonl",
                "--limit",
                "-1",
            ]
        )

    assert exc_info.value.code == 2
    assert "--limit must be non-negative" in capsys.readouterr().err


def test_evaluate_downstream_cli_rejects_negative_max_cost_in_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(
            [
                "--dataset",
                "data/eval/downstream/hotpotqa_validation_10.jsonl",
                "--compressor",
                "identity",
                "--task-model",
                "gpt-4o-mini",
                "--output",
                "models/eval/downstream_hotpotqa_identity_gpt4omini.jsonl",
                "--max-cost",
                "-0.1",
            ]
        )

    assert exc_info.value.code == 2
    assert "--max-cost must be non-negative" in capsys.readouterr().err


def test_evaluate_downstream_main_writes_jsonl_and_default_summary_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    summary_path = tmp_path / "downstream_results.summary.json"
    fake_client = FakeAsyncClient()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: fake_client)

    main()

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(written) == 1
    assert written[0]["example_id"] == "ex-1"
    assert written[0]["task_model"] == "gpt-4o-mini"

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["dataset"] == "data/eval/downstream/hotpotqa_validation_10.jsonl"
    assert summary["compressor"] == "identity"
    assert summary["task_model"] == "gpt-4o-mini"
    assert summary["examples_loaded"] == 1
    assert summary["examples_completed_this_run"] == 1
    assert summary["examples_completed_total"] == 1
    assert summary["examples_evaluated"] == 1
    assert summary["examples"] == 1
    assert len(fake_client.prompts) == 2


def test_evaluate_downstream_main_resume_skips_existing_examples(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    output_path.write_text(
        json.dumps(_result_row("ex-1", cost=0.1), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fake_client = FakeAsyncClient()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice"), _example("ex-2", "Bob")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: fake_client)

    main()

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["example_id"] for row in written] == ["ex-1", "ex-2"]

    summary = json.loads((tmp_path / "downstream_results.summary.json").read_text(encoding="utf-8"))
    assert summary["examples_loaded"] == 2
    assert summary["examples_completed_this_run"] == 1
    assert summary["examples_completed_total"] == 2
    assert summary["examples_evaluated"] == 1
    assert summary["examples"] == 2
    assert len(fake_client.prompts) == 2


def test_evaluate_downstream_main_resume_accepts_real_output_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"

    first_client = FakeAsyncClient()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: first_client)

    main()

    written_first = [
        json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()
    ]
    assert written_first == [
        {
            **written_first[0],
            "dataset": "data/eval/downstream/hotpotqa_validation_10.jsonl",
        }
    ]

    second_client = FakeAsyncClient()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice"), _example("ex-2", "Bob")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: second_client)

    main()

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["example_id"] for row in written] == ["ex-1", "ex-2"]
    assert all(
        row["dataset"] == "data/eval/downstream/hotpotqa_validation_10.jsonl" for row in written
    )
    assert len(second_client.prompts) == 2


def test_evaluate_downstream_main_resume_tracks_duplicate_example_id_occurrences(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    output_path.write_text(
        json.dumps(_result_row("ex-1", answer="Alice", cost=0.1), ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    fake_client = FakeAsyncClient()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [
            _example("ex-1", "Alice"),
            _example("ex-1", "Bob"),
            _example("ex-2", "Bob"),
        ],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: fake_client)

    main()

    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["example_id"] for row in written] == ["ex-1", "ex-1", "ex-2"]
    assert [row["full_output"] for row in written] == ["Alice", "Bob", "Bob"]

    summary = json.loads((tmp_path / "downstream_results.summary.json").read_text(encoding="utf-8"))
    assert summary["examples_loaded"] == 3
    assert summary["examples_completed_this_run"] == 2
    assert summary["examples_completed_total"] == 3
    assert summary["examples_evaluated"] == 2
    assert summary["examples"] == 3
    assert len(fake_client.prompts) == 4


def test_evaluate_downstream_main_completed_resume_does_not_initialize_adapter_or_task_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    output_path.write_text(
        json.dumps(
            _result_row(
                "ex-1",
                compressor="adapter_tinker",
                checkpoint_path="tinker://run-id/weights/step-001000",
            ),
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_tinker",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
            "--checkpoint-path",
            "tinker://run-id/weights/step-001000",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice")],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "create_async_tinker_adapter",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("adapter init should not run")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "create_task_client",
        lambda *_: (_ for _ in ()).throw(AssertionError("task-model init should not run")),
    )

    main()

    summary = json.loads((tmp_path / "downstream_results.summary.json").read_text(encoding="utf-8"))
    assert summary["examples_loaded"] == 1
    assert summary["examples_completed_this_run"] == 0
    assert summary["examples_completed_total"] == 1
    assert summary["examples_evaluated"] == 0
    assert summary["examples"] == 1


def test_evaluate_downstream_main_rejects_invalid_resume_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    bad_row = _result_row("ex-1")
    del bad_row["compressed_output"]
    output_path.write_text(json.dumps(bad_row, ensure_ascii=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
        ],
    )
    monkeypatch.setattr(evaluate_downstream, "load_examples", lambda path, limit=None: [])

    with pytest.raises(
        RuntimeError,
        match="Invalid resumed downstream result row in .*downstream_results.jsonl:1.*compressed_output",
    ):
        main()


def test_evaluate_downstream_main_rejects_incompatible_resume_row(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    bad_row = _result_row("ex-1")
    bad_row["task_model"] = "claude-sonnet-4-20250514"
    output_path.write_text(json.dumps(bad_row, ensure_ascii=True) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
        ],
    )
    monkeypatch.setattr(evaluate_downstream, "load_examples", lambda path, limit=None: [])

    with pytest.raises(
        RuntimeError,
        match="Incompatible resumed downstream result row in .*downstream_results.jsonl:1.*task_model",
    ):
        main()


@pytest.mark.parametrize(
    ("compressor", "argv_suffix", "row_extra", "expected_field"),
    [
        ("truncate", ["--truncate-tokens", "128"], {"truncate_tokens": 64}, "truncate_tokens"),
        (
            "extractive",
            ["--extractive-chars", "512"],
            {"extractive_chars": 256},
            "extractive_chars",
        ),
    ],
)
def test_evaluate_downstream_main_rejects_incompatible_baseline_resume_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compressor: str,
    argv_suffix: list[str],
    row_extra: dict[str, object],
    expected_field: str,
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    output_path.write_text(
        json.dumps(_result_row("ex-1", compressor=compressor, **row_extra), ensure_ascii=True)
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            compressor,
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
            *argv_suffix,
        ],
    )
    monkeypatch.setattr(evaluate_downstream, "load_examples", lambda path, limit=None: [])

    with pytest.raises(
        RuntimeError,
        match=rf"Incompatible resumed downstream result row in .*downstream_results.jsonl:1.*{expected_field}",
    ):
        main()


def test_evaluate_downstream_main_rejects_incompatible_resume_before_initialization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    output_path.write_text(
        json.dumps(
            _result_row(
                "ex-1",
                compressor="adapter_local",
                adapter_path="models/adapters/local/b",
                adapter_model="mlx-community/model-a",
            ),
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_local",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
            "--adapter-path",
            "models/adapters/local/a",
            "--adapter-model",
            "mlx-community/model-a",
        ],
    )
    monkeypatch.setattr(evaluate_downstream, "load_examples", lambda path, limit=None: [])
    monkeypatch.setattr(
        evaluate_downstream,
        "create_local_generator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("adapter init should not run")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "create_task_client",
        lambda *_: (_ for _ in ()).throw(AssertionError("task-model init should not run")),
    )

    with pytest.raises(
        RuntimeError,
        match="Incompatible resumed downstream result row in .*downstream_results.jsonl:1.*adapter_path",
    ):
        main()


@pytest.mark.parametrize(
    ("compressor", "argv_suffix", "row_extra", "expected_field"),
    [
        (
            "adapter_local",
            [
                "--adapter-path",
                "models/adapters/local/a",
                "--adapter-model",
                "mlx-community/model-a",
            ],
            {
                "adapter_path": "models/adapters/local/b",
                "adapter_model": "mlx-community/model-a",
            },
            "adapter_path",
        ),
        (
            "adapter_tinker",
            ["--checkpoint-path", "tinker://run-id/weights/step-001000"],
            {"checkpoint_path": "tinker://run-id/weights/step-009999"},
            "checkpoint_path",
        ),
    ],
)
def test_evaluate_downstream_main_rejects_incompatible_adapter_resume_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    compressor: str,
    argv_suffix: list[str],
    row_extra: dict[str, object],
    expected_field: str,
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    output_path.write_text(
        json.dumps(
            _result_row(
                "ex-1",
                compressor=compressor,
                **row_extra,
            ),
            ensure_ascii=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            compressor,
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--resume",
            *argv_suffix,
        ],
    )
    monkeypatch.setattr(evaluate_downstream, "load_examples", lambda path, limit=None: [])
    if compressor == "adapter_tinker":
        monkeypatch.setattr(
            evaluate_downstream,
            "get_settings",
            lambda: SimpleNamespace(tinker_api_key="test-key"),
            raising=False,
        )
        monkeypatch.setattr(
            evaluate_downstream,
            "create_async_tinker_adapter",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("adapter init should not run")
            ),
            raising=False,
        )
    else:
        monkeypatch.setattr(
            evaluate_downstream,
            "create_local_generator",
            lambda model, adapter_path, verbose=False: (
                lambda text, system_prompt, max_tokens=512: (text, 1, 1)
            ),
            raising=False,
        )

    with pytest.raises(
        RuntimeError,
        match=rf"Incompatible resumed downstream result row in .*downstream_results.jsonl:1.*{expected_field}",
    ):
        main()


def test_evaluate_downstream_main_stops_before_next_example_would_exceed_max_cost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    summary_path = tmp_path / "downstream_results.summary.json"
    calls: list[list[str]] = []

    async def fake_evaluate_examples(
        examples: list[DownstreamExample],
        compressor: str,
        task_model: str,
        client: object,
        **kwargs: object,
    ) -> list[dict[str, object]]:
        calls.append([example.id for example in examples])
        results = [
            _result_row(example.id, answer=example.gold.answer or "Alice", cost=0.06)
            for example in examples
        ]
        output_path_arg = kwargs.get("output_path")
        if isinstance(output_path_arg, Path):
            output_path_arg.parent.mkdir(parents=True, exist_ok=True)
            _append_result_rows(output_path_arg, results)
        return results

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--max-cost",
            "0.1",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice"), _example("ex-2", "Bob")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: object())
    monkeypatch.setattr(evaluate_downstream, "evaluate_examples", fake_evaluate_examples)
    monkeypatch.setattr(
        evaluate_downstream,
        "estimate_example_cost_upper_bound",
        lambda *args, **kwargs: 0.06,
        raising=False,
    )

    main()

    assert calls == [["ex-1"]]
    written = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["example_id"] for row in written] == ["ex-1"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["examples_loaded"] == 2
    assert summary["examples_completed_this_run"] == 1
    assert summary["examples_completed_total"] == 1
    assert summary["examples_evaluated"] == 1
    assert summary["examples"] == 1


def test_evaluate_downstream_main_honors_summary_output_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    summary_path = tmp_path / "custom-summary.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "truncate",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--truncate-tokens",
            "1",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: FakeAsyncClient())

    main()

    assert summary_path.exists()
    assert not (tmp_path / "downstream_results.summary.json").exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["compressor"] == "truncate"
    assert summary["examples_completed_total"] == 1
    assert summary["examples"] == 1


@pytest.mark.parametrize(
    ("argv_suffix", "expected_message"),
    [
        (["--adapter-model", "mlx-community/test-model"], "adapter_local requires --adapter-path"),
        (
            ["--adapter-path", "models/adapters/mlx/latest"],
            "adapter_local requires --adapter-model",
        ),
    ],
)
def test_evaluate_downstream_main_requires_explicit_adapter_local_args(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
    argv_suffix: list[str],
    expected_message: str,
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_local",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            *argv_suffix,
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert expected_message in capsys.readouterr().err


def test_evaluate_downstream_main_requires_checkpoint_for_adapter_tinker(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_tinker",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 2
    assert "adapter_tinker requires --checkpoint-path" in capsys.readouterr().err


def test_evaluate_downstream_main_requires_tinker_api_key_for_adapter_tinker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_tinker",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--checkpoint-path",
            "tinker://run-id/weights/step-001000",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "get_settings",
        lambda: SimpleNamespace(tinker_api_key=""),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="TINKER_API_KEY is required for adapter_tinker"):
        main()


def test_evaluate_downstream_main_checks_tinker_api_key_before_loading_dataset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    missing_dataset = tmp_path / "missing.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            str(missing_dataset),
            "--compressor",
            "adapter_tinker",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--checkpoint-path",
            "tinker://run-id/weights/step-001000",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "get_settings",
        lambda: SimpleNamespace(tinker_api_key=""),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="TINKER_API_KEY is required for adapter_tinker"):
        main()


def test_evaluate_downstream_main_wires_adapter_object_into_runner_call(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    summary_path = tmp_path / "downstream_results.summary.json"
    captured: dict[str, object] = {}

    def fake_local_generator(model: str, adapter_path: Path, verbose: bool = False):
        captured["factory_args"] = (model, adapter_path, verbose)

        def generate(
            input_text: str, system_prompt: str, max_tokens: int = 512
        ) -> tuple[str, int, int]:
            captured["generator_call"] = (input_text, system_prompt, max_tokens)
            return (f"compressed::{input_text}", 7, 3)

        return generate

    async def fake_evaluate_examples(*args: object, **kwargs: object) -> list[dict[str, object]]:
        adapter = cast(Any, kwargs.get("adapter"))
        captured["compressor"] = kwargs.get("compressor")
        captured["adapter"] = adapter
        assert hasattr(adapter, "compress")
        assert adapter.compress("alpha beta") == "compressed::alpha beta"
        return [_result_row("ex-1")]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_local",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--adapter-path",
            str(tmp_path / "adapter"),
            "--adapter-model",
            "mlx-community/test-model",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: FakeAsyncClient())
    monkeypatch.setattr(
        evaluate_downstream,
        "create_local_generator",
        fake_local_generator,
        raising=False,
    )
    monkeypatch.setattr(evaluate_downstream, "evaluate_examples", fake_evaluate_examples)

    main()

    assert captured["compressor"] == "adapter_local"
    assert hasattr(captured["adapter"], "compress")
    assert captured["factory_args"] == (
        "mlx-community/test-model",
        tmp_path / "adapter",
        False,
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["compressor"] == "adapter_local"


def test_evaluate_downstream_main_uses_async_tinker_adapter_path_instead_of_sync_factory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_path = tmp_path / "downstream_results.jsonl"
    summary_path = tmp_path / "downstream_results.summary.json"
    captured: dict[str, object] = {}

    class FakeAsyncTinkerAdapter:
        async def compress(self, context: str) -> str:
            captured["compressed_context"] = context
            return f"compressed::{context}"

    async def fake_create_async_tinker_adapter(checkpoint_path: str, api_key: str) -> object:
        captured["factory_args"] = (checkpoint_path, api_key)
        return FakeAsyncTinkerAdapter()

    async def fake_evaluate_examples(*args: object, **kwargs: object) -> list[dict[str, object]]:
        captured["compressor"] = kwargs.get("compressor")
        captured["adapter"] = kwargs.get("adapter")
        return [_result_row("ex-1", compressor="adapter_tinker")]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate_downstream.py",
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "adapter_tinker",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--checkpoint-path",
            "tinker://run-id/weights/step-001000",
        ],
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "load_examples",
        lambda path, limit=None: [_example("ex-1", "Alice")],
    )
    monkeypatch.setattr(evaluate_downstream, "create_task_client", lambda _: FakeAsyncClient())
    monkeypatch.setattr(
        evaluate_downstream,
        "get_settings",
        lambda: SimpleNamespace(tinker_api_key="test-key"),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "create_tinker_generator",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("sync Tinker factory should not be used")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        evaluate_downstream,
        "create_async_tinker_adapter",
        fake_create_async_tinker_adapter,
        raising=False,
    )
    monkeypatch.setattr(evaluate_downstream, "evaluate_examples", fake_evaluate_examples)

    main()

    assert captured["compressor"] == "adapter_tinker"
    assert hasattr(captured["adapter"], "compress")
    assert captured["factory_args"] == ("tinker://run-id/weights/step-001000", "test-key")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["compressor"] == "adapter_tinker"


@pytest.mark.asyncio
async def test_create_async_tinker_adapter_prefers_async_setup_and_sampling_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {"to_thread_calls": []}

    class FakeTokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], add_generation_prompt: bool, tokenize: bool
        ) -> str:
            assert add_generation_prompt is True
            assert tokenize is False
            captured["messages"] = messages
            return "prompt"

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [1, 2, 3]

        def decode(self, token_ids: list[int]) -> str:
            return "decoded"

    class FakeSamplingClient:
        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

        async def sample(
            self, model_input: object, num_samples: int, sampling_params: object
        ) -> object:
            captured["sample_args"] = (model_input, num_samples, sampling_params)
            return SimpleNamespace(text="compressed async")

    class FakeTrainingClient:
        def __init__(self) -> None:
            self.sync_factory_called = False

        async def save_weights_and_get_sampling_client_async(self) -> FakeSamplingClient:
            captured["used_async_sampling_factory"] = True
            return FakeSamplingClient()

        def save_weights_and_get_sampling_client(self) -> object:
            self.sync_factory_called = True
            raise AssertionError("sync sampling factory should not be used")

    class FakeServiceClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        async def create_training_client_from_state_async(
            self, checkpoint_path: str
        ) -> FakeTrainingClient:
            captured["checkpoint_path"] = checkpoint_path
            return FakeTrainingClient()

        def create_training_client_from_state(self, checkpoint_path: str) -> object:
            raise AssertionError("sync training factory should not be used")

    fake_tinker = SimpleNamespace(
        ServiceClient=FakeServiceClient,
        ModelInput=SimpleNamespace(from_ints=lambda token_ids: ("model-input", token_ids)),
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        cast(list[str], captured["to_thread_calls"]).append(getattr(func, "__name__", repr(func)))
        return cast(Any, func)(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)
    monkeypatch.setattr(evaluate_downstream.asyncio, "to_thread", fake_to_thread)

    adapter = await evaluate_downstream.create_async_tinker_adapter(
        "tinker://run-id/weights/step-001000",
        "test-key",
    )
    compressed = await adapter.compress("Alpha context")

    assert compressed == "compressed async"
    assert captured["api_key"] == "test-key"
    assert captured["checkpoint_path"] == "tinker://run-id/weights/step-001000"
    assert captured["used_async_sampling_factory"] is True
    assert cast(list[str], captured["to_thread_calls"]) == []


@pytest.mark.asyncio
async def test_create_async_tinker_adapter_uses_to_thread_for_sync_only_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {"to_thread_calls": []}

    class FakeTokenizer:
        def apply_chat_template(
            self, messages: list[dict[str, str]], add_generation_prompt: bool, tokenize: bool
        ) -> str:
            return "prompt"

        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            return [1, 2, 3]

        def decode(self, token_ids: list[int]) -> str:
            return "decoded"

    class FakeSamplingClient:
        def get_tokenizer(self) -> FakeTokenizer:
            return FakeTokenizer()

    class FakeTrainingClient:
        def save_weights_and_get_sampling_client(self) -> FakeSamplingClient:
            captured["sync_sampling_factory_used"] = True
            return FakeSamplingClient()

    class FakeServiceClient:
        def __init__(self, api_key: str) -> None:
            captured["api_key"] = api_key

        def create_training_client_from_state(self, checkpoint_path: str) -> FakeTrainingClient:
            captured["checkpoint_path"] = checkpoint_path
            return FakeTrainingClient()

    fake_tinker = SimpleNamespace(
        ServiceClient=FakeServiceClient,
        ModelInput=SimpleNamespace(from_ints=lambda token_ids: ("model-input", token_ids)),
        SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
    )

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        cast(list[str], captured["to_thread_calls"]).append(getattr(func, "__name__", repr(func)))
        return cast(Any, func)(*args, **kwargs)

    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)
    monkeypatch.setattr(evaluate_downstream.asyncio, "to_thread", fake_to_thread)

    adapter = await evaluate_downstream.create_async_tinker_adapter(
        "tinker://run-id/weights/step-001000",
        "test-key",
    )

    assert hasattr(adapter, "compress")
    assert captured["api_key"] == "test-key"
    assert captured["checkpoint_path"] == "tinker://run-id/weights/step-001000"
    assert captured["sync_sampling_factory_used"] is True
    assert cast(list[str], captured["to_thread_calls"]) == [
        "create_training_client_from_state",
        "save_weights_and_get_sampling_client",
    ]
