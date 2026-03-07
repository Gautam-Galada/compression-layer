#!/usr/bin/env python3
"""CLI for runnable small-slice downstream evaluation."""

import argparse
import asyncio
import inspect
import json
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.evaluate_tinker import create_local_generator
from src.evaluation.downstream.dataset import load_examples
from src.evaluation.downstream.runner import (
    ADAPTER_COMPRESSORS,
    ALL_COMPRESSORS,
    _estimate_cost_from_tokens,
    aggregate_results,
    append_results_jsonl,
    build_task_prompt,
    compress_context,
    evaluate_examples,
    render_context,
)
from src.utils.config import get_settings
from src.utils.tokenizers import count_tokens
from src.validation.models import ModelClient, model_type_from_string

RESULT_REQUIRED_FIELDS = (
    "dataset",
    "example_id",
    "benchmark",
    "compressor",
    "task_model",
    "full_output",
    "compressed_output",
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
)

DEFAULT_ADAPTER_SYSTEM_PROMPT = (
    "You are a semantic compression engine. Compress the input into minimal tokens "
    "while preserving all information for equivalent LLM reasoning. Use dense notation: "
    "labeled fields, standard abbreviations, and symbols (-> | + @). Never lose information."
)


class TaskModelClientAdapter:
    def __init__(self, model_name: str):
        self._client = ModelClient(
            model_type_from_string(model_name),
            operation="downstream_eval",
        )

    async def complete(self, prompt: str) -> Any:
        return await self._client.complete_response(prompt, 1024, 0.0)


class GeneratorBackedAdapter:
    def __init__(
        self,
        generator: Callable[[str, str, int], tuple[str, int, int]],
        *,
        system_prompt: str = DEFAULT_ADAPTER_SYSTEM_PROMPT,
        max_tokens: int = 512,
    ):
        self._generator = generator
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    def compress(self, context: str) -> str:
        output, _, _ = self._generator(context, self._system_prompt, self._max_tokens)
        return output


class AsyncTinkerAdapter:
    def __init__(
        self,
        *,
        tinker_module: Any,
        tokenizer: Any,
        sampling_client: Any,
        system_prompt: str = DEFAULT_ADAPTER_SYSTEM_PROMPT,
        max_tokens: int = 512,
    ):
        self._tinker = tinker_module
        self._tokenizer = tokenizer
        self._sampling_client = sampling_client
        self._system_prompt = system_prompt
        self._max_tokens = max_tokens

    async def compress(self, context: str) -> str:
        from scripts.evaluate_tinker import (
            _build_user_message,
            _cap_output_length,
            _compute_generation_budget,
            _strip_generation_artifacts,
            _truncate_repetition,
        )

        messages = []
        if self._system_prompt:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": _build_user_message(context)})

        prompt_str = self._tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        input_token_ids = self._tokenizer.encode(prompt_str)
        input_tokens = len(input_token_ids)
        max_generation_tokens = _compute_generation_budget(input_tokens, self._max_tokens)
        model_input = self._tinker.ModelInput.from_ints(input_token_ids)
        sampling_params = self._tinker.SamplingParams(
            max_tokens=max_generation_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],
        )

        sample_method = self._sampling_client.sample
        if inspect.iscoroutinefunction(sample_method):
            sample_response = sample_method(
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )
        else:
            sample_response = await asyncio.to_thread(
                sample_method,
                model_input,
                num_samples=1,
                sampling_params=sampling_params,
            )

        if inspect.isawaitable(sample_response):
            sample_result = await cast(Any, sample_response)
        elif callable(getattr(sample_response, "result", None)):
            sample_result = await asyncio.to_thread(sample_response.result)
        else:
            sample_result = sample_response

        if hasattr(sample_result, "sequences") and sample_result.sequences:
            output_token_ids = sample_result.sequences[0].tokens
            raw_output = self._tokenizer.decode(output_token_ids)
        elif getattr(sample_result, "samples", None):
            output_token_ids = sample_result.samples[0].to_ints()
            raw_output = self._tokenizer.decode(output_token_ids)
        elif getattr(sample_result, "text", None):
            raw_output = sample_result.text
        else:
            raw_output = str(sample_result)

        output_text = _strip_generation_artifacts(raw_output.strip())
        output_text = _truncate_repetition(output_text)
        output_text = _cap_output_length(output_text, context)
        return output_text


class DownstreamArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> Any:
        normalized_message = {
            "argument --limit: must be non-negative": "--limit must be non-negative",
            "argument --max-cost: must be non-negative": "--max-cost must be non-negative",
            "argument --truncate-tokens: must be positive": "--truncate-tokens must be positive",
            "argument --extractive-chars: must be positive": "--extractive-chars must be positive",
        }.get(message, message)
        super().error(normalized_message)


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = DownstreamArgumentParser(description="Run downstream evaluation")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--compressor", choices=ALL_COMPRESSORS, required=True)
    parser.add_argument("--task-model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--limit", type=_non_negative_int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cost", type=_non_negative_float, default=None)
    parser.add_argument("--truncate-tokens", type=_positive_int, default=256)
    parser.add_argument("--extractive-chars", type=_positive_int, default=1000)
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--adapter-model", default=None)
    parser.add_argument("--checkpoint-path", default=None)
    return parser


def default_summary_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".summary.json")


def validate_result_row(
    row: Mapping[str, Any], *, source: Path, line_number: int
) -> dict[str, Any]:
    missing_fields = [field for field in RESULT_REQUIRED_FIELDS if field not in row]
    if missing_fields:
        missing = ", ".join(missing_fields)
        raise RuntimeError(
            f"Invalid resumed downstream result row in {source}:{line_number} missing fields: {missing}"
        )
    return dict(row)


def validate_result_compatibility(
    row: Mapping[str, Any],
    *,
    dataset: Path,
    compressor: str,
    task_model: str,
    source: Path,
    line_number: int,
    adapter_path: Path | None = None,
    adapter_model: str | None = None,
    checkpoint_path: str | None = None,
) -> dict[str, Any]:
    expected_values: dict[str, Any] = {
        "dataset": str(dataset),
        "compressor": compressor,
        "task_model": task_model,
    }
    if compressor == "adapter_local":
        expected_values["adapter_path"] = str(adapter_path)
        expected_values["adapter_model"] = cast(Any, adapter_model)
    if compressor == "adapter_tinker":
        expected_values["checkpoint_path"] = cast(Any, checkpoint_path)
    for field, expected in expected_values.items():
        actual = row.get(field)
        if actual != expected:
            raise RuntimeError(
                f"Incompatible resumed downstream result row in {source}:{line_number} field {field}={actual!r} does not match current run {expected!r}"
            )
    return dict(row)


def load_existing_results(
    path: Path,
    *,
    dataset: Path,
    compressor: str,
    task_model: str,
    adapter_path: Path | None = None,
    adapter_model: str | None = None,
    checkpoint_path: str | None = None,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []

    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            raw_row = cast(dict[str, Any], json.loads(line))
            validated_row = validate_result_row(raw_row, source=path, line_number=line_number)
            rows.append(
                validate_result_compatibility(
                    validated_row,
                    dataset=dataset,
                    compressor=compressor,
                    task_model=task_model,
                    adapter_path=adapter_path,
                    adapter_model=adapter_model,
                    checkpoint_path=checkpoint_path,
                    source=path,
                    line_number=line_number,
                )
            )
    return rows


def create_task_client(model_name: str) -> TaskModelClientAdapter:
    return TaskModelClientAdapter(model_name)


def validate_backend_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.compressor == "adapter_local":
        if args.adapter_path is None:
            parser.error("adapter_local requires --adapter-path")
        if args.adapter_model is None:
            parser.error("adapter_local requires --adapter-model")

    if args.compressor == "adapter_tinker" and args.checkpoint_path is None:
        parser.error("adapter_tinker requires --checkpoint-path")


async def create_async_tinker_adapter(
    checkpoint_path: str,
    api_key: str,
    *,
    system_prompt: str = DEFAULT_ADAPTER_SYSTEM_PROMPT,
    max_tokens: int = 512,
) -> AsyncTinkerAdapter:
    import tinker  # type: ignore[import-not-found]

    service_client = tinker.ServiceClient(api_key=api_key)

    async_training_factory = getattr(
        service_client, "create_training_client_from_state_async", None
    )
    if callable(async_training_factory):
        training_client = await cast(Any, async_training_factory(checkpoint_path))
    else:
        training_client = await asyncio.to_thread(
            service_client.create_training_client_from_state,
            checkpoint_path,
        )

    sampling_client_factory = getattr(
        training_client, "save_weights_and_get_sampling_client_async", None
    )
    if callable(sampling_client_factory):
        sampling_client = await cast(Any, sampling_client_factory())
    else:
        sampling_client = await asyncio.to_thread(
            training_client.save_weights_and_get_sampling_client
        )

    tokenizer = sampling_client.get_tokenizer()
    return AsyncTinkerAdapter(
        tinker_module=tinker,
        tokenizer=tokenizer,
        sampling_client=sampling_client,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
    )


async def build_adapter(args: argparse.Namespace) -> Any | None:
    if args.compressor == "adapter_local":
        generator = create_local_generator(args.adapter_model, args.adapter_path)
        return GeneratorBackedAdapter(generator)

    if args.compressor == "adapter_tinker":
        settings = get_settings()
        if not settings.tinker_api_key:
            raise RuntimeError("TINKER_API_KEY is required for adapter_tinker")

        return await create_async_tinker_adapter(args.checkpoint_path, settings.tinker_api_key)

    return None


def estimate_example_cost_upper_bound(
    example: Any,
    *,
    compressor: str,
    task_model: str,
    truncate_tokens_limit: int,
    extractive_chars: int,
    adapter: Any | None = None,
) -> float:
    full_context = render_context(example.retrieved_context)
    if compressor in ADAPTER_COMPRESSORS:
        compressed_context = full_context
    else:
        compressed_context = compress_context(
            full_context,
            compressor,
            truncate_tokens_limit=truncate_tokens_limit,
            extractive_chars=extractive_chars,
            adapter=adapter,
        )
    full_prompt = build_task_prompt(example.query, full_context)
    compressed_prompt = build_task_prompt(example.query, compressed_context)
    full_prompt_tokens = count_tokens(full_prompt)
    compressed_prompt_tokens = count_tokens(compressed_prompt)
    output_token_upper_bound = 1024
    total_input_cost = _estimate_cost_from_tokens(
        full_prompt_tokens + compressed_prompt_tokens,
        task_model,
        is_input=True,
    )
    total_output_cost = _estimate_cost_from_tokens(
        2 * output_token_upper_bound,
        task_model,
        is_input=False,
    )
    return total_input_cost + total_output_cost


async def run(args: argparse.Namespace) -> dict[str, Any]:
    examples = load_examples(args.dataset, limit=args.limit)

    if not args.resume and args.output.exists():
        args.output.unlink()

    existing_results = (
        load_existing_results(
            args.output,
            dataset=args.dataset,
            compressor=args.compressor,
            task_model=args.task_model,
            adapter_path=args.adapter_path,
            adapter_model=args.adapter_model,
            checkpoint_path=args.checkpoint_path,
        )
        if args.resume
        else []
    )
    completed_example_ids = {
        str(row["example_id"])
        for row in existing_results
        if isinstance(row, Mapping) and "example_id" in row
    }
    pending_examples = [example for example in examples if example.id not in completed_example_ids]

    adapter = None
    client = None
    if pending_examples:
        adapter = await build_adapter(args) if args.compressor in ADAPTER_COMPRESSORS else None
        client = create_task_client(args.task_model)

    new_results: list[dict[str, Any]] = []
    spent_so_far = sum(
        float(row["cost_usd_full"]) + float(row["cost_usd_compressed"]) for row in existing_results
    )

    for example in pending_examples:
        if args.max_cost is not None:
            estimated_next_cost = estimate_example_cost_upper_bound(
                example,
                compressor=args.compressor,
                task_model=args.task_model,
                truncate_tokens_limit=args.truncate_tokens,
                extractive_chars=args.extractive_chars,
                adapter=adapter,
            )
            if spent_so_far + estimated_next_cost > args.max_cost:
                break

        example_results = await evaluate_examples(
            [example],
            compressor=args.compressor,
            task_model=args.task_model,
            client=cast(Any, client),
            truncate_tokens_limit=args.truncate_tokens,
            extractive_chars=args.extractive_chars,
            adapter=adapter,
        )
        if not example_results:
            continue

        result = cast(dict[str, Any], example_results[0])
        result["dataset"] = str(args.dataset)
        if args.compressor == "adapter_local":
            result["adapter_path"] = str(args.adapter_path)
            result["adapter_model"] = args.adapter_model
        if args.compressor == "adapter_tinker":
            result["checkpoint_path"] = args.checkpoint_path
        append_results_jsonl(args.output, [result])
        new_results.append(result)
        spent_so_far += float(result["cost_usd_full"]) + float(result["cost_usd_compressed"])

    all_results = existing_results + new_results
    aggregate_summary = dict(aggregate_results(cast(list[Mapping[str, Any]], all_results)))
    examples_completed_total = int(aggregate_summary.get("examples", 0))
    examples_completed_this_run = len(new_results)
    summary: dict[str, Any] = {
        "dataset": str(args.dataset),
        "compressor": args.compressor,
        "task_model": args.task_model,
        "examples_loaded": len(examples),
        "examples_completed_this_run": examples_completed_this_run,
        "examples_completed_total": examples_completed_total,
        "examples_evaluated": examples_completed_this_run,
        "resume": args.resume,
        "max_cost": args.max_cost,
        **aggregate_summary,
    }

    if args.max_cost is not None:
        total_cost = float(summary["total_cost_usd_full"]) + float(
            summary["total_cost_usd_compressed"]
        )
        if total_cost > args.max_cost:
            raise RuntimeError(
                f"Downstream evaluation exceeded --max-cost: {total_cost} > {args.max_cost}"
            )

    summary_output = args.summary_output or default_summary_output_path(args.output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_backend_args(parser, args)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
