#!/usr/bin/env python3
"""Preprocess synthetic compression pairs with heuristic quality filters.

This script removes generation artifacts from synthetic outputs and filters out
low-quality pairs before creating validated files for train/valid/test splitting.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tokenizers import compression_ratio

TAG_PATTERN = re.compile(r"</?think>|</?tool_call>", re.IGNORECASE)
FENCE_LINE_PATTERN = re.compile(r"^```[a-zA-Z0-9_-]*$|^```$")
BLANK_LINES_PATTERN = re.compile(r"\n{3,}")


@dataclass
class PreprocessConfig:
    """Configuration for filtering and cleaning synthetic pairs."""

    max_char_ratio: float = 1.0
    max_token_ratio: float = 1.0


@dataclass
class PreprocessStats:
    """Aggregate statistics from preprocessing."""

    total: int = 0
    passed: int = 0
    rejected: int = 0
    rejected_by_reason: dict[str, int] = field(default_factory=dict)


def clean_compressed_text(text: str) -> str:
    """Remove reasoning/tool artifacts and normalize generated compressed text."""
    cleaned = TAG_PATTERN.sub("", text)
    lines = [line for line in cleaned.splitlines() if not FENCE_LINE_PATTERN.match(line.strip())]
    cleaned = "\n".join(lines).strip()
    cleaned = BLANK_LINES_PATTERN.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_pair(
    pair: dict[str, Any],
    config: PreprocessConfig,
) -> tuple[dict[str, Any] | None, str | None]:
    """Clean and validate one synthetic pair.

    Returns:
        (cleaned_pair, None) when accepted, otherwise (None, rejection_reason).
    """
    verbose = pair.get("verbose")
    compressed = pair.get("compressed")
    domain = pair.get("domain")

    if not isinstance(verbose, str) or not isinstance(compressed, str):
        return None, "invalid_text_fields"
    if domain not in {"nl", "code", "mixed"}:
        return None, "invalid_domain"

    verbose = verbose.strip()
    compressed = clean_compressed_text(compressed)

    if not verbose or not compressed:
        return None, "empty_text"

    char_ratio = len(compressed) / len(verbose)
    if char_ratio >= config.max_char_ratio:
        return None, "char_ratio_ge_1.0"

    token_ratio = compression_ratio(verbose, compressed)
    if token_ratio > config.max_token_ratio:
        return None, "token_ratio_gt_max"

    metadata = pair.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    cleaned_pair = {
        "verbose": verbose,
        "compressed": compressed,
        "domain": domain,
        "language": pair.get("language"),
        "metadata": metadata,
        "validation": {
            "passed": True,
            "method": "heuristic_preprocess_v1",
            "char_ratio": round(char_ratio, 4),
            "token_ratio": round(token_ratio, 4),
        },
    }
    return cleaned_pair, None


def preprocess_file(
    input_path: Path,
    output_path: Path,
    rejected_path: Path | None,
    config: PreprocessConfig,
) -> PreprocessStats:
    """Preprocess one synthetic JSONL file and write clean/rejected outputs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rejected_path:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)

    rejected_by_reason: defaultdict[str, int] = defaultdict(int)
    stats = PreprocessStats()

    with (
        open(input_path, encoding="utf-8") as in_file,
        open(output_path, "w", encoding="utf-8") as out_file,
    ):
        rejected_file = open(rejected_path, "w", encoding="utf-8") if rejected_path else None
        try:
            for line_number, line in enumerate(in_file, start=1):
                if not line.strip():
                    continue

                stats.total += 1

                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    reason = "json_decode_error"
                    stats.rejected += 1
                    rejected_by_reason[reason] += 1
                    if rejected_file:
                        rejected_file.write(
                            json.dumps(
                                {
                                    "reason": reason,
                                    "line_number": line_number,
                                    "raw": line.rstrip("\n"),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                cleaned, reason = clean_pair(record, config)

                if cleaned is not None:
                    stats.passed += 1
                    out_file.write(json.dumps(cleaned, ensure_ascii=False) + "\n")
                    continue

                stats.rejected += 1
                assert reason is not None
                rejected_by_reason[reason] += 1

                if rejected_file:
                    rejected_file.write(
                        json.dumps(
                            {
                                "reason": reason,
                                "line_number": line_number,
                                "pair": record,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        finally:
            if rejected_file:
                rejected_file.close()

    stats.rejected_by_reason = dict(sorted(rejected_by_reason.items(), key=lambda item: item[0]))
    return stats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess synthetic pairs and filter low-quality outputs",
    )
    parser.add_argument("--input", type=Path, required=True, help="Input synthetic JSONL file")
    parser.add_argument("--output", type=Path, required=True, help="Output validated JSONL file")
    parser.add_argument(
        "--rejected",
        type=Path,
        default=None,
        help="Optional JSONL path for rejected samples with reasons",
    )
    parser.add_argument(
        "--max-char-ratio",
        type=float,
        default=1.0,
        help="Reject if len(compressed)/len(verbose) is >= this value (default: 1.0)",
    )
    parser.add_argument(
        "--max-token-ratio",
        type=float,
        default=1.0,
        help="Reject if token ratio compressed/original is > this value (default: 1.0)",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    config = PreprocessConfig(
        max_char_ratio=args.max_char_ratio,
        max_token_ratio=args.max_token_ratio,
    )

    stats = preprocess_file(
        input_path=args.input,
        output_path=args.output,
        rejected_path=args.rejected,
        config=config,
    )

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    if args.rejected:
        print(f"Rejected: {args.rejected}")
    print(f"Total: {stats.total}")
    print(f"Passed: {stats.passed}")
    print(f"Rejected: {stats.rejected}")
    if stats.total:
        print(f"Pass rate: {stats.passed / stats.total * 100:.2f}%")
    if stats.rejected_by_reason:
        print("Rejected by reason:")
        for reason, count in stats.rejected_by_reason.items():
            print(f"  - {reason}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
