from types import SimpleNamespace
from typing import Any, cast

import pytest

from scripts.evaluate_downstream import AsyncTinkerAdapter, build_parser
from src.evaluation.downstream.dataset import DownstreamExample, GoldTarget, RetrievedChunk
from src.evaluation.downstream.runner import (
    ADAPTER_COMPRESSORS,
    ALL_COMPRESSORS,
    aggregate_results,
    append_results_jsonl,
    build_task_prompt,
    compress_context,
    evaluate_examples,
    render_context,
    score_pair,
)
from src.validation.models import APIResponse, ModelClient, ModelType


def test_render_context_joins_ranked_chunks() -> None:
    example = DownstreamExample(
        id="ex-1",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(doc_id="doc-2", rank=2, title="Doc 2", text="Bob lost."),
            RetrievedChunk(doc_id="doc-1", rank=1, title="Doc 1", text="Alice won."),
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    rendered = render_context(example.retrieved_context)

    assert rendered == "[Doc 1]\nAlice won.\n\n[Doc 2]\nBob lost."


def test_build_task_prompt_is_deterministic_and_grounded() -> None:
    prompt = build_task_prompt(
        query="Who won?",
        rendered_context="[Doc 1]\nAlice won.",
    )

    assert prompt == (
        "Answer the question using only the provided context. If the context "
        "is insufficient, say you do not know.\n\n"
        "Question:\nWho won?\n\n"
        "Context:\n[Doc 1]\nAlice won.\n\n"
        "Answer:"
    )


def test_compress_context_supports_baseline_compressors() -> None:
    context = "alpha beta gamma delta"

    assert compress_context(context, "identity") == context
    assert compress_context(context, "truncate", truncate_tokens_limit=2) == "alpha beta"
    assert compress_context(context, "extractive", extractive_chars=10) == "alpha beta"


def test_compress_context_rejects_unsupported_compressor() -> None:
    with pytest.raises(ValueError, match="Unsupported downstream compressor: unknown"):
        compress_context("alpha beta", "unknown")


def test_compress_context_only_dispatches_adapter_for_explicit_names() -> None:
    class StubAdapter:
        def compress(self, context: str) -> str:
            return f"adapted::{context}"

    adapter = StubAdapter()

    assert compress_context("alpha beta", "adapter_local", adapter=adapter) == "adapted::alpha beta"
    assert (
        compress_context("alpha beta", "adapter_tinker", adapter=adapter) == "adapted::alpha beta"
    )

    with pytest.raises(ValueError, match="Unsupported downstream compressor: unknown"):
        compress_context("alpha beta", "unknown", adapter=adapter)


def test_compress_context_requires_adapter_instance_for_known_adapter_compressor() -> None:
    compressor = sorted(ADAPTER_COMPRESSORS)[0]

    with pytest.raises(
        RuntimeError,
        match=rf"Adapter compressor requires an adapter instance: {compressor}",
    ):
        compress_context("alpha beta", compressor)


@pytest.mark.asyncio
async def test_evaluate_examples_supports_async_adapter_compression() -> None:
    class AsyncAdapter:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def compress(self, context: str) -> str:
            self.calls.append(context)
            return "[Doc 1]\nAlice w"

    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            self.calls.append(prompt)
            if "Context:\n[Doc 1]\nAlice won the match decisively." in prompt:
                return {
                    "text": "Alice",
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "latency_ms": 12.5,
                }
            if "Context:\n[Doc 1]\nAlice w\n\nAnswer:" in prompt:
                return {
                    "text": "Alice",
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "latency_ms": 7.0,
                }
            raise AssertionError(f"unexpected prompt: {prompt}")

    example = DownstreamExample(
        id="ex-async-adapter",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    adapter = AsyncAdapter()
    client = FakeAsyncClient()

    results = await evaluate_examples(
        [example],
        compressor="adapter_tinker",
        task_model="gpt-4o-mini",
        client=client,
        adapter=cast(Any, adapter),
    )

    assert len(results) == 1
    assert adapter.calls == ["[Doc 1]\nAlice won the match decisively."]
    assert len(client.calls) == 2
    assert "Context:\n[Doc 1]\nAlice w\n\nAnswer:" in client.calls[1]


@pytest.mark.asyncio
async def test_async_tinker_adapter_handles_sync_sample_result_via_to_thread(
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
            return "sync sample output"

    class FakeFuture:
        def result(self) -> object:
            return SimpleResult()

    class SimpleResult:
        sequences = [type("Sequence", (), {"tokens": [7, 8, 9]})()]

    class FakeSamplingClient:
        def sample(
            self, model_input: object, num_samples: int, sampling_params: object
        ) -> FakeFuture:
            return FakeFuture()

    class FakeModelInput:
        @staticmethod
        def from_ints(token_ids: list[int]) -> tuple[str, list[int]]:
            return ("model-input", token_ids)

    async def fake_to_thread(func: object, *args: object, **kwargs: object) -> object:
        cast(list[str], captured["to_thread_calls"]).append(getattr(func, "__name__", repr(func)))
        return cast(Any, func)(*args, **kwargs)

    monkeypatch.setattr("scripts.evaluate_downstream.asyncio.to_thread", fake_to_thread)

    adapter = AsyncTinkerAdapter(
        tinker_module=SimpleNamespace(
            ModelInput=FakeModelInput,
            SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
        tokenizer=FakeTokenizer(),
        sampling_client=FakeSamplingClient(),
    )

    compressed = await adapter.compress("Alpha context")

    assert compressed == "sync sample output"
    assert cast(list[str], captured["to_thread_calls"]) == ["sample", "result"]


def test_all_compressors_contains_baselines_and_adapters() -> None:
    assert ALL_COMPRESSORS == (
        "identity",
        "truncate",
        "extractive",
        "adapter_local",
        "adapter_tinker",
    )


def test_score_pair_computes_delta_without_judge() -> None:
    full_output = "Alice"
    compressed_output = "Alice"
    result = score_pair(full_output, compressed_output, gold_answer="Alice")
    assert result == {
        "full_exact_match": 1.0,
        "compressed_exact_match": 1.0,
        "delta_exact_match": 0.0,
        "full_f1": 1.0,
        "compressed_f1": 1.0,
        "delta_f1": 0.0,
    }
    assert set(result) == {
        "full_exact_match",
        "compressed_exact_match",
        "delta_exact_match",
        "full_f1",
        "compressed_f1",
        "delta_f1",
    }


def test_score_pair_uses_code_reference_match_when_requested() -> None:
    result = score_pair(
        "result = df.iloc[list]",
        "df sort_values",
        gold_answer="df.iloc[List]",
        scorer="code_reference_match",
    )

    assert result == {
        "full_exact_match": 1.0,
        "compressed_exact_match": 0.0,
        "delta_exact_match": -1.0,
        "full_f1": 1.0,
        "compressed_f1": 0.0,
        "delta_f1": -1.0,
    }


def test_score_pair_keeps_code_reference_exact_match_binary() -> None:
    result = score_pair(
        "df.iloc sort_values",
        "df.iloc sort_values",
        gold_answer="df.iloc list",
        scorer="code_reference_match",
    )

    assert result["full_exact_match"] == 0.0
    assert result["compressed_exact_match"] == 0.0
    assert result["full_f1"] > 0.0
    assert result["compressed_f1"] > 0.0


def test_score_pair_uses_best_match_across_answer_and_aliases() -> None:
    result = score_pair(
        full_output="wrong",
        compressed_output="Beta",
        gold_answer="Alpha",
        gold_aliases=["Beta"],
    )

    assert result == {
        "full_exact_match": 0.0,
        "compressed_exact_match": 1.0,
        "delta_exact_match": 1.0,
        "full_f1": 0.0,
        "compressed_f1": 1.0,
        "delta_f1": 1.0,
    }


def test_score_pair_handles_missing_gold_answer_with_aliases_only() -> None:
    result = score_pair(
        full_output="wrong",
        compressed_output="Alias answer",
        gold_answer=None,
        gold_aliases=["Alias answer"],
    )

    assert result == {
        "full_exact_match": 0.0,
        "compressed_exact_match": 1.0,
        "delta_exact_match": 1.0,
        "full_f1": 0.0,
        "compressed_f1": 1.0,
        "delta_f1": 1.0,
    }


@pytest.mark.asyncio
async def test_evaluate_examples_runs_full_and_compressed_arms_and_returns_metrics(
    tmp_path,
) -> None:
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            self.calls.append(prompt)
            if "Context:\n[Doc 1]\nAlice won the match decisively." in prompt:
                return {
                    "text": "Alice",
                    "input_tokens": 20,
                    "output_tokens": 4,
                    "latency_ms": 12.5,
                }
            if "Context:\n[Doc 1]\nAlice w" in prompt:
                return {
                    "text": "Bob",
                    "input_tokens": 10,
                    "output_tokens": 3,
                    "latency_ms": 7.0,
                }
            raise AssertionError(f"unexpected prompt: {prompt}")

    examples = [
        DownstreamExample(
            id="ex-1",
            benchmark="hotpotqa",
            split="validation",
            domain="nl",
            task_type="qa",
            query="Who won?",
            retrieved_context=[
                RetrievedChunk(
                    doc_id="doc-1",
                    rank=1,
                    title="Doc 1",
                    text="Alice won the match decisively.",
                )
            ],
            gold=GoldTarget(answer="Alice"),
            scorer="qa_exact_f1",
        )
    ]
    client = FakeAsyncClient()

    results = await evaluate_examples(
        examples,
        compressor="extractive",
        task_model="gpt-4o-mini",
        client=client,
        extractive_chars=15,
        output_path=tmp_path / "results.jsonl",
    )

    assert len(client.calls) == 2
    assert len(results) == 1
    result = results[0]

    assert result["example_id"] == "ex-1"
    assert result["benchmark"] == "hotpotqa"
    assert result["compressor"] == "extractive"
    assert result["task_model"] == "gpt-4o-mini"
    assert result["full_output"] == "Alice"
    assert result["compressed_output"] == "Bob"
    assert result["full_exact_match"] == 1.0
    assert result["compressed_exact_match"] == 0.0
    assert result["delta_exact_match"] == -1.0
    assert result["full_f1"] == 1.0
    assert result["compressed_f1"] == 0.0
    assert result["delta_f1"] == -1.0
    assert result["context_tokens_full"] > result["context_tokens_compressed"]
    assert 0.0 < result["compression_ratio"] < 1.0
    assert result["latency_ms_full"] == 12.5
    assert result["latency_ms_compressed"] == 7.0
    assert result["cost_usd_full"] > result["cost_usd_compressed"] >= 0.0

    written = (tmp_path / "results.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(written) == 1
    assert '"example_id": "ex-1"' in written[0]


@pytest.mark.asyncio
async def test_evaluate_examples_calls_client_twice_even_when_prompts_match() -> None:
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            self.calls.append(prompt)
            return {
                "text": "Alice",
                "input_tokens": 10,
                "output_tokens": 2,
                "latency_ms": 5.0,
            }

    example = DownstreamExample(
        id="ex-identity",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )
    client = FakeAsyncClient()

    await evaluate_examples(
        [example],
        compressor="identity",
        task_model="gpt-4o-mini",
        client=client,
    )

    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]


@pytest.mark.asyncio
async def test_evaluate_examples_rejects_unsupported_client_response_shape() -> None:
    class BadAsyncClient:
        async def complete(self, prompt: str) -> str:
            return "Alice"

    example = DownstreamExample(
        id="ex-bad-shape",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    with pytest.raises(TypeError, match="Unsupported task-model client response shape"):
        await evaluate_examples(
            [example],
            compressor="identity",
            task_model="gpt-4o-mini",
            client=BadAsyncClient(),
        )


@pytest.mark.asyncio
async def test_evaluate_examples_uses_response_token_usage_for_costs() -> None:
    class FakeAsyncClient:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            self.calls += 1
            return {
                "text": "Alice",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 5.0,
            }

    example = DownstreamExample(
        id="ex-cost",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    results = await evaluate_examples(
        [example],
        compressor="identity",
        task_model="gpt-4o-mini",
        client=FakeAsyncClient(),
    )

    result = results[0]
    assert result["cost_usd_full"] == 0.00000075
    assert result["cost_usd_compressed"] == 0.00000075


@pytest.mark.asyncio
async def test_evaluate_examples_uses_shared_model_pricing_source(monkeypatch) -> None:
    class FakeAsyncClient:
        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            return {
                "text": "Alice",
                "input_tokens": 1,
                "output_tokens": 1,
                "latency_ms": 5.0,
            }

    pricing = {
        "claude-sonnet-4-20250514": (3.0, 15.0),
        "gpt-4o-mini": (1.0, 2.0),
        "gemini-2.0-flash": (0.15, 0.60),
    }
    monkeypatch.setattr("src.utils.costs.MODEL_PRICING", pricing)

    example = DownstreamExample(
        id="ex-shared-cost",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    results = await evaluate_examples(
        [example],
        compressor="identity",
        task_model="gpt-4o-mini",
        client=FakeAsyncClient(),
    )

    result = results[0]
    assert result["cost_usd_full"] == 0.000003
    assert result["cost_usd_compressed"] == 0.000003


@pytest.mark.asyncio
async def test_evaluate_examples_accepts_attribute_based_client_responses() -> None:
    class Response:
        def __init__(self, text: str, input_tokens: int, output_tokens: int, latency_ms: float):
            self.text = text
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.latency_ms = latency_ms

    class FakeAsyncClient:
        async def complete(self, prompt: str) -> Response:
            return Response("Alice", 3, 2, 9.5)

    example = DownstreamExample(
        id="ex-attr",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    results = await evaluate_examples(
        [example],
        compressor="identity",
        task_model="gpt-4o-mini",
        client=FakeAsyncClient(),
    )

    assert results[0]["full_output"] == "Alice"
    assert results[0]["latency_ms_full"] == 9.5


@pytest.mark.asyncio
async def test_model_client_complete_response_uses_same_cache_and_cost_logging(monkeypatch) -> None:
    cache_calls: dict[str, object] = {}
    tracker_calls: list[dict[str, object]] = []

    class FakeCache:
        def get(self, key: str) -> None:
            cache_calls["get"] = key
            return None

        def set(self, key: str, value: dict[str, object]) -> None:
            cache_calls["set"] = (key, value)

    class FakeTracker:
        def log_call(
            self, model: str, input_tokens: int, output_tokens: int, operation: str = "unknown"
        ) -> float:
            tracker_calls.append(
                {
                    "model": model,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "operation": operation,
                }
            )
            return 0.0

    async def fake_call_with_retry(prompt: str, max_tokens: int, temperature: float) -> APIResponse:
        return APIResponse(
            text="Alice",
            input_tokens=3,
            output_tokens=2,
            model="gpt-4o-mini",
        )

    monkeypatch.setattr(ModelClient, "_init_client", lambda self: None)
    monkeypatch.setattr(ModelClient, "_call_with_retry", staticmethod(fake_call_with_retry))
    monkeypatch.setattr("src.validation.models.get_cost_tracker", lambda: FakeTracker())

    client = ModelClient(
        ModelType.GPT4O_MINI,
        cache=cast(Any, FakeCache()),
        operation="downstream_eval",
    )

    response = await client.complete_response("Hello")

    assert response.text == "Alice"
    assert "get" in cache_calls
    assert cache_calls["set"] == (
        cache_calls["get"],
        {"text": "Alice"},
    )
    assert tracker_calls == [
        {
            "model": "gpt-4o-mini",
            "input_tokens": 3,
            "output_tokens": 2,
            "operation": "downstream_eval",
        }
    ]


@pytest.mark.asyncio
async def test_evaluate_examples_rejects_missing_gold_targets() -> None:
    class FakeAsyncClient:
        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            return {
                "text": "Alice",
                "input_tokens": 3,
                "output_tokens": 2,
                "latency_ms": 5.0,
            }

    example = DownstreamExample(
        id="ex-no-gold",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer=None, aliases=[]),
        scorer="qa_exact_f1",
    )

    with pytest.raises(
        ValueError, match="Downstream example ex-no-gold is missing gold answer and aliases"
    ):
        await evaluate_examples(
            [example],
            compressor="identity",
            task_model="gpt-4o-mini",
            client=FakeAsyncClient(),
        )


@pytest.mark.asyncio
async def test_evaluate_examples_rejects_unknown_task_model_pricing() -> None:
    class FakeAsyncClient:
        async def complete(self, prompt: str) -> dict[str, float | int | str]:
            return {
                "text": "Alice",
                "input_tokens": 3,
                "output_tokens": 2,
                "latency_ms": 5.0,
            }

    example = DownstreamExample(
        id="ex-unknown-model",
        benchmark="hotpotqa",
        split="validation",
        domain="nl",
        task_type="qa",
        query="Who won?",
        retrieved_context=[
            RetrievedChunk(
                doc_id="doc-1",
                rank=1,
                title="Doc 1",
                text="Alice won the match decisively.",
            )
        ],
        gold=GoldTarget(answer="Alice"),
        scorer="qa_exact_f1",
    )

    with pytest.raises(ValueError, match="Unsupported task model pricing: unknown-model"):
        await evaluate_examples(
            [example],
            compressor="identity",
            task_model="unknown-model",
            client=FakeAsyncClient(),
        )


def test_aggregate_results_computes_means_from_multiple_rows() -> None:
    summary = aggregate_results(
        [
            {
                "full_exact_match": 1.0,
                "compressed_exact_match": 0.0,
                "delta_exact_match": -1.0,
                "full_f1": 1.0,
                "compressed_f1": 0.0,
                "delta_f1": -1.0,
                "context_tokens_full": 100,
                "context_tokens_compressed": 50,
                "compression_ratio": 0.5,
                "latency_ms_full": 12.0,
                "latency_ms_compressed": 6.0,
                "cost_usd_full": 0.002,
                "cost_usd_compressed": 0.001,
            },
            {
                "full_exact_match": 0.0,
                "compressed_exact_match": 1.0,
                "delta_exact_match": 1.0,
                "full_f1": 0.2,
                "compressed_f1": 0.6,
                "delta_f1": 0.4,
                "context_tokens_full": 80,
                "context_tokens_compressed": 40,
                "compression_ratio": 0.5,
                "latency_ms_full": 8.0,
                "latency_ms_compressed": 4.0,
                "cost_usd_full": 0.0016,
                "cost_usd_compressed": 0.0008,
            },
        ]
    )

    assert summary == {
        "examples": 2,
        "avg_full_exact_match": 0.5,
        "avg_compressed_exact_match": 0.5,
        "avg_delta_exact_match": 0.0,
        "avg_full_f1": 0.6,
        "avg_compressed_f1": 0.3,
        "avg_delta_f1": -0.3,
        "avg_context_tokens_full": 90.0,
        "avg_context_tokens_compressed": 45.0,
        "avg_compression_ratio": 0.5,
        "avg_latency_ms_full": 10.0,
        "avg_latency_ms_compressed": 5.0,
        "total_cost_usd_full": 0.0036,
        "total_cost_usd_compressed": 0.0018,
    }


def test_append_results_jsonl_appends_without_overwriting(tmp_path) -> None:
    output_path = tmp_path / "results.jsonl"

    append_results_jsonl(output_path, [{"example_id": "ex-1", "full_exact_match": 1.0}])
    append_results_jsonl(output_path, [{"example_id": "ex-2", "full_exact_match": 0.0}])

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert '"example_id": "ex-1"' in lines[0]
    assert '"example_id": "ex-2"' in lines[1]


def test_evaluate_downstream_cli_accepts_required_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--dataset",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
            "--compressor",
            "identity",
            "--task-model",
            "gpt-4o-mini",
            "--output",
            "models/eval/downstream_hotpotqa_identity_gpt4omini.jsonl",
        ]
    )

    assert args.compressor == "identity"
    assert args.task_model == "gpt-4o-mini"
