"""Paired downstream evaluation helpers."""

import inspect
import json
from collections.abc import Awaitable, Iterable, Mapping
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Protocol, TypedDict

from src.evaluation.downstream.baselines import extractive_head, identity_context, truncate_tokens
from src.evaluation.downstream.dataset import DownstreamExample, RetrievedChunk
from src.evaluation.downstream.scoring import (
    code_reference_exact_match,
    code_reference_match,
    exact_match,
    token_f1,
)
from src.utils import costs
from src.utils.tokenizers import compression_ratio, count_tokens


class PairMetrics(TypedDict):
    full_exact_match: float
    compressed_exact_match: float
    delta_exact_match: float
    full_f1: float
    compressed_f1: float
    delta_f1: float


class ClientResponse(TypedDict):
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ExampleResult(PairMetrics):
    example_id: str
    benchmark: str
    compressor: str
    task_model: str
    full_output: str
    compressed_output: str
    context_tokens_full: int
    context_tokens_compressed: int
    compression_ratio: float
    latency_ms_full: float
    latency_ms_compressed: float
    cost_usd_full: float
    cost_usd_compressed: float


class SummaryMetrics(TypedDict):
    examples: int
    avg_full_exact_match: float
    avg_compressed_exact_match: float
    avg_delta_exact_match: float
    avg_full_f1: float
    avg_compressed_f1: float
    avg_delta_f1: float
    avg_context_tokens_full: float
    avg_context_tokens_compressed: float
    avg_compression_ratio: float
    avg_latency_ms_full: float
    avg_latency_ms_compressed: float
    total_cost_usd_full: float
    total_cost_usd_compressed: float


class ContextAdapter(Protocol):
    def compress(self, context: str) -> str | Awaitable[str]: ...


class AsyncTaskModelClient(Protocol):
    async def complete(self, prompt: str) -> Any: ...


AggregateMetricField = Literal[
    "full_exact_match",
    "compressed_exact_match",
    "delta_exact_match",
    "full_f1",
    "compressed_f1",
    "delta_f1",
    "context_tokens_full",
    "context_tokens_compressed",
    "compression_ratio",
    "latency_ms_full",
    "latency_ms_compressed",
    "cost_usd_full",
    "cost_usd_compressed",
]


BASELINE_COMPRESSORS = ("identity", "truncate", "extractive")
ADAPTER_COMPRESSORS = ("adapter_local", "adapter_tinker")
ALL_COMPRESSORS = BASELINE_COMPRESSORS + ADAPTER_COMPRESSORS


def render_context(chunks: list[RetrievedChunk]) -> str:
    parts: list[str] = []
    for chunk in sorted(chunks, key=lambda item: item.rank):
        label = chunk.title or chunk.doc_id
        parts.append(f"[{label}]\n{chunk.text}")
    return "\n\n".join(parts)


def build_task_prompt(query: str, rendered_context: str) -> str:
    return (
        "Answer the question using only the provided context. If the context "
        "is insufficient, say you do not know.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{rendered_context}\n\n"
        "Answer:"
    )


def compress_context(
    context: str,
    compressor: str,
    *,
    truncate_tokens_limit: int = 256,
    extractive_chars: int = 1000,
    adapter: ContextAdapter | None = None,
) -> str | Awaitable[str]:
    if compressor == "identity":
        return identity_context(context)
    if compressor == "truncate":
        return truncate_tokens(context, truncate_tokens_limit)
    if compressor == "extractive":
        return extractive_head(context, extractive_chars)
    if compressor in ADAPTER_COMPRESSORS:
        if adapter is None:
            raise RuntimeError(f"Adapter compressor requires an adapter instance: {compressor}")
        return adapter.compress(context)
    raise ValueError(f"Unsupported downstream compressor: {compressor}")


async def compress_context_async(
    context: str,
    compressor: str,
    *,
    truncate_tokens_limit: int = 256,
    extractive_chars: int = 1000,
    adapter: ContextAdapter | None = None,
) -> str:
    compressed = compress_context(
        context,
        compressor,
        truncate_tokens_limit=truncate_tokens_limit,
        extractive_chars=extractive_chars,
        adapter=adapter,
    )
    if inspect.isawaitable(compressed):
        return str(await compressed)
    return str(compressed)


def score_pair(
    full_output: str,
    compressed_output: str,
    gold_answer: str | None,
    gold_aliases: list[str] | None = None,
    scorer: str = "qa_exact_f1",
) -> PairMetrics:
    if scorer == "qa_exact_f1":
        exact_metric = exact_match
        f1_metric = token_f1
    elif scorer == "code_reference_match":
        exact_metric = code_reference_exact_match
        f1_metric = code_reference_match
    else:
        raise ValueError(f"Unsupported downstream scorer: {scorer}")

    candidates = [
        candidate for candidate in [gold_answer, *(gold_aliases or [])] if candidate is not None
    ]
    if not candidates:
        raise ValueError("Downstream example is missing gold answer and aliases")

    full_exact_match = max(exact_metric(full_output, candidate) for candidate in candidates)
    compressed_exact_match = max(
        exact_metric(compressed_output, candidate) for candidate in candidates
    )
    full_f1 = max(f1_metric(full_output, candidate) for candidate in candidates)
    compressed_f1 = max(f1_metric(compressed_output, candidate) for candidate in candidates)
    return {
        "full_exact_match": full_exact_match,
        "compressed_exact_match": compressed_exact_match,
        "delta_exact_match": compressed_exact_match - full_exact_match,
        "full_f1": full_f1,
        "compressed_f1": compressed_f1,
        "delta_f1": compressed_f1 - full_f1,
    }


def _normalize_response(raw_response: Any, elapsed_ms: float) -> ClientResponse:
    if isinstance(raw_response, Mapping):
        required_fields = ("text", "input_tokens", "output_tokens")
        if not all(field in raw_response for field in required_fields):
            raise TypeError("Unsupported task-model client response shape")

        latency_ms = float(raw_response.get("latency_ms", elapsed_ms))
        return {
            "text": str(raw_response["text"]),
            "input_tokens": int(raw_response["input_tokens"]),
            "output_tokens": int(raw_response["output_tokens"]),
            "latency_ms": latency_ms,
        }

    required_attrs = ("text", "input_tokens", "output_tokens")
    if not all(hasattr(raw_response, attr) for attr in required_attrs):
        raise TypeError("Unsupported task-model client response shape")

    latency_ms = float(getattr(raw_response, "latency_ms", elapsed_ms))
    return {
        "text": str(raw_response.text),
        "input_tokens": int(raw_response.input_tokens),
        "output_tokens": int(raw_response.output_tokens),
        "latency_ms": latency_ms,
    }


async def _complete(client: AsyncTaskModelClient, prompt: str) -> ClientResponse:
    started = perf_counter()
    raw_response = await client.complete(prompt)
    elapsed_ms = (perf_counter() - started) * 1000
    return _normalize_response(raw_response, elapsed_ms)


def _estimate_cost_from_tokens(tokens: int, model: str, *, is_input: bool) -> float:
    if model not in costs.MODEL_PRICING:
        raise ValueError(f"Unsupported task model pricing: {model}")

    rate_input, rate_output = costs.MODEL_PRICING[model]
    rate = rate_input if is_input else rate_output
    return (tokens / 1_000_000) * rate


def append_results_jsonl(path: Path, results: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=True) + "\n")


def aggregate_results(results: list[ExampleResult | Mapping[str, Any]]) -> SummaryMetrics:
    if not results:
        return {
            "examples": 0,
            "avg_full_exact_match": 0.0,
            "avg_compressed_exact_match": 0.0,
            "avg_delta_exact_match": 0.0,
            "avg_full_f1": 0.0,
            "avg_compressed_f1": 0.0,
            "avg_delta_f1": 0.0,
            "avg_context_tokens_full": 0.0,
            "avg_context_tokens_compressed": 0.0,
            "avg_compression_ratio": 0.0,
            "avg_latency_ms_full": 0.0,
            "avg_latency_ms_compressed": 0.0,
            "total_cost_usd_full": 0.0,
            "total_cost_usd_compressed": 0.0,
        }

    count = len(results)

    def metric_value(
        result: ExampleResult | Mapping[str, Any], field: AggregateMetricField
    ) -> float:
        return float(result[field])

    def avg(field: AggregateMetricField) -> float:
        return round(sum(metric_value(result, field) for result in results) / count, 10)

    def total(field: AggregateMetricField) -> float:
        return round(sum(metric_value(result, field) for result in results), 10)

    return {
        "examples": count,
        "avg_full_exact_match": avg("full_exact_match"),
        "avg_compressed_exact_match": avg("compressed_exact_match"),
        "avg_delta_exact_match": avg("delta_exact_match"),
        "avg_full_f1": avg("full_f1"),
        "avg_compressed_f1": avg("compressed_f1"),
        "avg_delta_f1": avg("delta_f1"),
        "avg_context_tokens_full": avg("context_tokens_full"),
        "avg_context_tokens_compressed": avg("context_tokens_compressed"),
        "avg_compression_ratio": avg("compression_ratio"),
        "avg_latency_ms_full": avg("latency_ms_full"),
        "avg_latency_ms_compressed": avg("latency_ms_compressed"),
        "total_cost_usd_full": total("cost_usd_full"),
        "total_cost_usd_compressed": total("cost_usd_compressed"),
    }


async def evaluate_examples(
    examples: list[DownstreamExample],
    compressor: str,
    task_model: str,
    client: AsyncTaskModelClient,
    *,
    truncate_tokens_limit: int = 256,
    extractive_chars: int = 1000,
    adapter: ContextAdapter | None = None,
    output_path: Path | None = None,
) -> list[ExampleResult]:
    results: list[ExampleResult] = []

    for example in examples:
        if example.gold.answer is None and not example.gold.aliases:
            raise ValueError(f"Downstream example {example.id} is missing gold answer and aliases")

        full_context = render_context(example.retrieved_context)
        compressed_context = await compress_context_async(
            full_context,
            compressor,
            truncate_tokens_limit=truncate_tokens_limit,
            extractive_chars=extractive_chars,
            adapter=adapter,
        )
        full_prompt = build_task_prompt(example.query, full_context)
        compressed_prompt = build_task_prompt(example.query, compressed_context)

        full_response = await _complete(client, full_prompt)
        compressed_response = await _complete(client, compressed_prompt)
        metrics = score_pair(
            full_response["text"],
            compressed_response["text"],
            gold_answer=example.gold.answer,
            gold_aliases=example.gold.aliases,
            scorer=example.scorer,
        )

        full_cost = _estimate_cost_from_tokens(
            full_response["input_tokens"],
            task_model,
            is_input=True,
        ) + _estimate_cost_from_tokens(
            full_response["output_tokens"],
            task_model,
            is_input=False,
        )
        compressed_cost = _estimate_cost_from_tokens(
            compressed_response["input_tokens"],
            task_model,
            is_input=True,
        ) + _estimate_cost_from_tokens(
            compressed_response["output_tokens"],
            task_model,
            is_input=False,
        )

        result: ExampleResult = {
            "example_id": example.id,
            "benchmark": example.benchmark,
            "compressor": compressor,
            "task_model": task_model,
            "full_output": full_response["text"],
            "compressed_output": compressed_response["text"],
            **metrics,
            "context_tokens_full": count_tokens(full_context),
            "context_tokens_compressed": count_tokens(compressed_context),
            "compression_ratio": compression_ratio(full_context, compressed_context),
            "latency_ms_full": full_response["latency_ms"],
            "latency_ms_compressed": compressed_response["latency_ms"],
            "cost_usd_full": round(full_cost, 10),
            "cost_usd_compressed": round(compressed_cost, 10),
        }
        results.append(result)

        if output_path is not None:
            append_results_jsonl(output_path, [result])

    return results
