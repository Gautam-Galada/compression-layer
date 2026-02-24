#!/usr/bin/env python3
"""Evaluate Tinker-trained compression adapter.

Uses Tinker's SamplingClient to run inference with the trained LoRA weights.

Usage:
    # Quick test with 5 examples
    python scripts/evaluate_tinker.py --limit 5

    # Full evaluation on test set
    python scripts/evaluate_tinker.py

    # With specific checkpoint
    python scripts/evaluate_tinker.py --checkpoint-path tinker://run-id/weights/step-001000
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table

from src.utils.config import get_settings

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class EvalExample:
    """A test example for evaluation."""

    input_text: str
    expected_output: str
    system_prompt: str


@dataclass
class EvalResult:
    """Result of evaluating one example."""

    input_text: str
    expected_output: str
    generated_output: str
    input_tokens: int
    output_tokens: int
    compression_ratio: float
    generation_time_ms: float


def _load_existing_results(path: Path) -> list[EvalResult]:
    """Load existing results from JSONL, stopping at first malformed line."""
    if not path.exists():
        return []

    loaded: list[EvalResult] = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                loaded.append(EvalResult(**record))
            except Exception:
                logger.warning(
                    "Stopped resume parsing at malformed line %d in %s",
                    line_num,
                    path,
                )
                break

    return loaded


def load_test_examples(path: Path, limit: int | None = None) -> list[EvalExample]:
    """Load test examples from chat-format JSONL."""
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            messages = record.get("messages", [])

            system_prompt = ""
            user_content = ""
            assistant_content = ""

            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "system":
                    system_prompt = content
                elif role == "user":
                    user_content = content.replace("Compress:\n", "").strip()
                elif role == "assistant":
                    assistant_content = content

            if user_content and assistant_content:
                examples.append(
                    EvalExample(
                        input_text=user_content,
                        expected_output=assistant_content,
                        system_prompt=system_prompt,
                    )
                )

            if limit and len(examples) >= limit:
                break

    return examples


def _build_user_message(input_text: str) -> str:
    """Build user message in the same format used during training."""
    return f"Compress:\n{input_text}"


def _compute_generation_budget(input_tokens: int, requested_max_tokens: int) -> int:
    """Compute dynamic max generation tokens to keep decoding concise."""
    dynamic_budget = max(24, int(input_tokens * 0.60))
    return min(max(1, requested_max_tokens), dynamic_budget)


def _strip_generation_artifacts(output_text: str) -> str:
    """Remove model artifacts and control tokens from generated output."""
    cleaned = re.sub(r"<think>.*?</think>", "", output_text, flags=re.DOTALL).strip()

    if "<think>" in cleaned:
        before_think, after_think = cleaned.split("<think>", 1)
        after_think = after_think.lstrip("\n").strip()
        cleaned = after_think or before_think.strip()

    cleaned = cleaned.replace("</tool_call>", "")
    cleaned = cleaned.replace("<|im_end|>", "")
    cleaned = cleaned.replace("<|endoftext|>", "")

    return cleaned.strip()


def _truncate_repetition(text: str) -> str:
    """Truncate output when repeated lines or clauses indicate degeneration."""
    normalized = text.strip()
    if not normalized:
        return normalized

    lines = normalized.split("\n")
    if len(lines) >= 4:
        seen_lines: dict[str, int] = {}
        cutoff_idx = len(lines)

        for i, line in enumerate(lines):
            line_stripped = line.strip()
            if len(line_stripped) < 15:
                continue

            if line_stripped in seen_lines and i - seen_lines[line_stripped] > 1:
                cutoff_idx = seen_lines[line_stripped] + 1
                break

            seen_lines[line_stripped] = i

        if cutoff_idx < len(lines):
            normalized = "\n".join(lines[:cutoff_idx]).strip()

    for separator in (" | ", " |>"):
        if separator not in normalized:
            continue

        parts = [part.strip() for part in normalized.split(separator) if part.strip()]
        if len(parts) < 4:
            continue

        seen_parts: dict[str, int] = {}
        cutoff = len(parts)
        for i, part in enumerate(parts):
            key = re.sub(r"\s+", " ", part.lower()).strip()
            if len(key) < 6:
                continue
            if key in seen_parts:
                cutoff = i
                break
            seen_parts[key] = i

        if cutoff < len(parts):
            normalized = separator.join(parts[:cutoff]).strip()
            break

    return normalized


def _truncate_to_boundary(text: str, max_chars: int) -> str:
    """Truncate text near a natural boundary when possible."""
    truncated = text[:max_chars]
    break_at = max(
        truncated.rfind("\n"),
        truncated.rfind(" | "),
        truncated.rfind(" |> "),
        truncated.rfind(". "),
    )

    if break_at > max_chars * 0.6:
        return truncated[:break_at].strip()

    return truncated.strip()


def _cap_output_length(output_text: str, input_text: str) -> str:
    """Cap output length so generated text stays compressive."""
    if not input_text:
        return output_text.strip()

    soft_cap = max(48, int(len(input_text) * 0.75))
    hard_cap = max(48, int(len(input_text) * 0.95))

    capped = output_text.strip()
    if len(capped) > soft_cap:
        capped = _truncate_to_boundary(capped, soft_cap)
    if len(capped) > hard_cap:
        capped = _truncate_to_boundary(capped, hard_cap)
    return capped


def create_tinker_generator(checkpoint_path: str, api_key: str, verbose: bool = False):
    """Create a generator function using Tinker SamplingClient."""
    import tinker

    console.print(f"[cyan]Loading checkpoint: {checkpoint_path}[/cyan]")

    service_client = tinker.ServiceClient(api_key=api_key)

    # Load the trained weights and create sampling client
    training_client = service_client.create_training_client_from_state(checkpoint_path)
    sampling_client = training_client.save_weights_and_get_sampling_client()
    tokenizer = sampling_client.get_tokenizer()

    console.print("[green]Sampling client ready[/green]\n")

    def generate(
        input_text: str, system_prompt: str, max_tokens: int = 512
    ) -> tuple[str, int, int]:
        """Generate compressed output. Returns (output, input_tokens, output_tokens)."""
        # Build chat messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": _build_user_message(input_text)})

        # Apply chat template to get prompt string
        prompt_str = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Tokenize to get input token IDs
        input_token_ids = tokenizer.encode(prompt_str)
        input_tokens = len(input_token_ids)
        max_generation_tokens = _compute_generation_budget(input_tokens, max_tokens)

        # Create ModelInput from token IDs
        model_input = tinker.ModelInput.from_ints(input_token_ids)

        # Create sampling params
        # Use low temperature for deterministic compression
        # Note: Qwen3 may enter "thinking" mode - we strip <think> blocks in post-processing
        sampling_params = tinker.SamplingParams(
            max_tokens=max_generation_tokens,
            temperature=0.0,
            top_p=1.0,
            stop=["<|im_end|>", "<|endoftext|>", "\n\n\n"],  # Also stop on triple newline
        )

        # Sample from the model (num_samples=1)
        sample_result = sampling_client.sample(
            model_input,
            num_samples=1,
            sampling_params=sampling_params,
        ).result()

        # Extract generated text from response
        # sample_result.sequences contains SampledSequence objects
        if hasattr(sample_result, "sequences") and sample_result.sequences:
            output_token_ids = sample_result.sequences[0].tokens
            raw_output = tokenizer.decode(output_token_ids)
        elif getattr(sample_result, "samples", None):
            output_token_ids = getattr(sample_result, "samples")[0].to_ints()
            raw_output = tokenizer.decode(output_token_ids)
        elif getattr(sample_result, "text", None):
            raw_output = getattr(sample_result, "text")
        else:
            # Fallback: try to decode directly
            raw_output = str(sample_result)

        output_text = raw_output.strip()

        if verbose:
            console.print(
                f"[dim]RAW OUTPUT ({len(raw_output)} chars): {repr(raw_output[:300])}...[/dim]"
            )

        output_text = _strip_generation_artifacts(output_text)
        output_text = _truncate_repetition(output_text)
        output_text = _cap_output_length(output_text, input_text)

        # Count output tokens
        output_tokens = len(tokenizer.encode(output_text, add_special_tokens=False))

        return output_text, input_tokens, output_tokens

    return generate


def run_evaluation(
    examples: list[EvalExample],
    generator,
    output_path: Path | None = None,
    start_index: int = 0,
    total_examples: int | None = None,
) -> list[EvalResult]:
    """Run evaluation on all examples."""
    results = []

    total = total_examples if total_examples is not None else len(examples)
    console.print(
        f"[cyan]Evaluating {len(examples)} examples (starting at {start_index + 1}/{total})...[/cyan]\n"
    )

    for i, example in enumerate(examples, start_index + 1):
        start = time.perf_counter()
        generated, input_tokens, output_tokens = generator(
            example.input_text, example.system_prompt
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Calculate compression ratio (output chars / input chars)
        compression_ratio = len(generated) / len(example.input_text) if example.input_text else 0

        result = EvalResult(
            input_text=example.input_text,
            expected_output=example.expected_output,
            generated_output=generated,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            compression_ratio=compression_ratio,
            generation_time_ms=elapsed_ms,
        )
        results.append(result)

        # Print progress
        preview = example.input_text[:40].replace("\n", " ")
        if len(example.input_text) > 40:
            preview += "..."
        console.print(
            f"[{i}/{total}] "
            f"ratio={compression_ratio:.1%} "
            f"in={input_tokens} out={output_tokens} "
            f"time={elapsed_ms:.0f}ms "
            f"[dim]{preview}[/dim]"
        )

        # Save incrementally
        if output_path:
            with open(output_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.__dict__) + "\n")

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print evaluation summary."""
    if not results:
        console.print("[red]No results to summarize[/red]")
        return

    avg_ratio = sum(r.compression_ratio for r in results) / len(results)
    avg_time = sum(r.generation_time_ms for r in results) / len(results)
    avg_input_tokens = sum(r.input_tokens for r in results) / len(results)
    avg_output_tokens = sum(r.output_tokens for r in results) / len(results)
    token_ratio = avg_output_tokens / avg_input_tokens if avg_input_tokens else 0

    table = Table(title="Evaluation Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Examples evaluated", str(len(results)))
    table.add_row("Avg compression ratio (chars)", f"{avg_ratio:.1%}")
    table.add_row("Avg token ratio (out/in)", f"{token_ratio:.1%}")
    table.add_row("Avg input tokens", f"{avg_input_tokens:.1f}")
    table.add_row("Avg output tokens", f"{avg_output_tokens:.1f}")
    table.add_row("Avg generation time", f"{avg_time:.0f} ms")

    console.print()
    console.print(table)


def print_examples(results: list[EvalResult], n: int = 3) -> None:
    """Print a few example compressions."""
    console.print(f"\n[bold]Sample Compressions ({n} examples):[/bold]\n")

    for i, result in enumerate(results[:n], 1):
        console.print(f"[cyan]Example {i}:[/cyan]")
        console.print(f"[dim]Input ({len(result.input_text)} chars):[/dim]")
        console.print(result.input_text[:200] + ("..." if len(result.input_text) > 200 else ""))
        console.print()
        console.print(
            f"[dim]Generated ({len(result.generated_output)} chars, {result.compression_ratio:.1%}):[/dim]"
        )
        console.print(f"[green]{result.generated_output}[/green]")
        console.print()
        console.print(f"[dim]Expected:[/dim]")
        console.print(f"[yellow]{result.expected_output}[/yellow]")
        console.print("-" * 60)
        console.print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tinker-trained compression adapter")

    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Tinker checkpoint path (default: from tinker_run.json)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/training/test.jsonl"),
        help="Path to test data JSONL",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/eval/tinker_eval.jsonl"),
        help="Output path for results",
    )
    parser.add_argument(
        "--show-examples",
        type=int,
        default=3,
        help="Number of example compressions to display",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from existing output file instead of restarting",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show verbose output including raw model responses",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()

    # Check API key
    if not settings.tinker_api_key:
        console.print("[red]Error: TINKER_API_KEY not set[/red]")
        return 1

    # Get checkpoint path
    checkpoint_path = args.checkpoint_path
    if not checkpoint_path:
        # Load from tinker_run.json
        run_state_path = settings.adapters_dir / "tinker" / "tinker_run.json"
        if not run_state_path.exists():
            console.print(f"[red]No training run found at {run_state_path}[/red]")
            return 1

        with open(run_state_path, encoding="utf-8") as f:
            run_state = json.load(f)

        checkpoint_path = run_state.get("latest_checkpoint_path")
        if not checkpoint_path:
            console.print("[red]No checkpoint path in tinker_run.json[/red]")
            return 1

        console.print(f"[dim]Using checkpoint from training run: {checkpoint_path}[/dim]")

    # Load test examples
    if not args.data.exists():
        console.print(f"[red]Test data not found: {args.data}[/red]")
        return 1

    examples = load_test_examples(args.data, limit=args.limit)
    if not examples:
        console.print("[red]No test examples loaded[/red]")
        return 1

    console.print(f"[dim]Loaded {len(examples)} test examples[/dim]")

    existing_results: list[EvalResult] = []
    start_index = 0

    # Prepare output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        if args.resume:
            existing_results = _load_existing_results(args.output)
            start_index = min(len(existing_results), len(examples))
            if start_index > 0:
                console.print(
                    f"[dim]Resuming from {args.output}: {start_index} completed examples[/dim]"
                )
        else:
            args.output.unlink()

    remaining_examples = examples[start_index:]
    if not remaining_examples:
        console.print("[green]Evaluation already complete. No new API calls needed.[/green]")
        print_summary(existing_results)
        if args.show_examples > 0:
            print_examples(existing_results, args.show_examples)
        return 0

    # Create generator
    try:
        generator = create_tinker_generator(
            checkpoint_path, settings.tinker_api_key, verbose=args.verbose
        )
    except Exception as e:
        console.print(f"[red]Failed to create generator: {e}[/red]")
        return 1

    # Run evaluation
    new_results = run_evaluation(
        remaining_examples,
        generator,
        args.output,
        start_index=start_index,
        total_examples=len(examples),
    )
    results = existing_results + new_results

    # Print results
    print_summary(results)
    if args.show_examples > 0:
        print_examples(results, args.show_examples)

    console.print(f"\n[green]Results saved to {args.output}[/green]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
