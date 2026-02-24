"""Load and materialize Hugging Face chat datasets into local JSONL splits."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def _validate_messages(record: Any, source_split: str, row_index: int) -> list[dict[str, Any]]:
    if not isinstance(record, dict) or "messages" not in record:
        raise ValueError(
            f"Invalid record in split '{source_split}' at row {row_index}: missing messages field"
        )

    messages = record["messages"]
    if not isinstance(messages, list):
        raise ValueError(
            f"Invalid record in split '{source_split}' at row {row_index}: messages must be a list"
        )

    return messages


def materialize_hf_chat_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    split_map: dict[str, str] | None = None,
    force: bool = True,
) -> dict[str, int]:
    """Materialize HF chat dataset splits to local JSONL files.

    Writes destination files in the existing training layout:
    - train.jsonl
    - valid.jsonl
    - test.jsonl
    """
    mapping = split_map or {
        "train": "train",
        "validation": "valid",
        "test": "test",
    }

    dataset = load_dataset(dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {dest_split: 0 for dest_split in mapping.values()}

    for source_split, dest_split in mapping.items():
        if source_split not in dataset:
            raise ValueError(f"Dataset '{dataset_name}' missing split: {source_split}")

        output_path = output_dir / f"{dest_split}.jsonl"
        if output_path.exists() and not force:
            with open(output_path, encoding="utf-8") as f:
                counts[dest_split] = sum(1 for line in f if line.strip())
            continue

        with open(output_path, "w", encoding="utf-8") as f:
            for row_index, record in enumerate(dataset[source_split], start=1):
                messages = _validate_messages(record, source_split, row_index)
                payload = {"messages": messages}
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
                counts[dest_split] += 1

    return counts
