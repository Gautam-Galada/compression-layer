"""Shared downstream dataset models and JSONL IO."""

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SupportedScorer = Literal["qa_exact_f1", "code_reference_match"]


class RetrievedChunk(BaseModel):
    model_config = ConfigDict(extra="forbid")

    doc_id: str
    rank: int
    text: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GoldTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str | None = None
    aliases: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DownstreamExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    benchmark: str
    split: str
    domain: Literal["nl", "code", "mixed"]
    task_type: Literal["qa", "reasoning", "code_gen"]
    query: str
    retrieved_context: list[RetrievedChunk]
    gold: GoldTarget
    scorer: SupportedScorer
    metadata: dict[str, Any] = Field(default_factory=dict)


def save_examples(path: Path, examples: list[DownstreamExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.model_dump(mode="json"), ensure_ascii=True) + "\n")


def load_examples(path: Path, limit: int | None = None) -> list[DownstreamExample]:
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")

    with open(path, encoding="utf-8") as handle:
        examples: list[DownstreamExample] = []
        for line in handle:
            if not line.strip():
                continue
            examples.append(DownstreamExample.model_validate(json.loads(line)))
            if limit is not None and len(examples) >= limit:
                break
        return examples
