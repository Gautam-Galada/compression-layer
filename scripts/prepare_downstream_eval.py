#!/usr/bin/env python3
"""CLI skeleton for preparing downstream evaluation benchmarks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.downstream.dataset import save_examples
from src.evaluation.downstream.prepare import get_prepare_registry, get_supported_benchmarks


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare offline downstream evaluation benchmarks")
    parser.add_argument(
        "--benchmark",
        choices=tuple(get_supported_benchmarks()),
        required=True,
    )
    parser.add_argument("--split", required=True)
    parser.add_argument("--limit", type=_non_negative_int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-chunks", type=_non_negative_int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    prepare_fn = get_prepare_registry()[args.benchmark]
    examples = prepare_fn(
        split=args.split,
        limit=args.limit,
        seed=args.seed,
        max_chunks=args.max_chunks,
    )
    save_examples(args.output, examples)


if __name__ == "__main__":
    main()
