from collections.abc import Mapping
from typing import Any, cast

from datasets import load_dataset

from src.evaluation.downstream.dataset import DownstreamExample, GoldTarget, RetrievedChunk
from src.evaluation.downstream.prepare import PREPARE_REGISTRY, PrepareFn

BENCHMARK_NAME = "ds1000"
DATASET_NAME = "xlangai/DS-1000"


def build_ds1000_example(row: Mapping[str, Any], split: str, row_index: int) -> DownstreamExample:
    row_metadata = dict(row.get("metadata") or {})
    source_metadata = {"dataset": DATASET_NAME, "source": DATASET_NAME}
    metadata = {**row_metadata, **source_metadata}
    prompt = str(row.get("prompt") or "")
    code_context = str(row.get("code_context") or "")
    reference_code = str(row.get("reference_code") or "")
    problem_id = metadata.get("problem_id", row_index)
    library = metadata.get("library")
    return DownstreamExample(
        id=f"ds1000::{split}::{problem_id}",
        benchmark=BENCHMARK_NAME,
        split=split,
        domain="code",
        task_type="code_gen",
        query=prompt,
        retrieved_context=[
            RetrievedChunk(
                doc_id=f"ds1000::{problem_id}::context",
                rank=1,
                title=str(library) if library is not None else None,
                text=code_context,
                metadata={"library": library} if library is not None else {},
            )
        ],
        gold=GoldTarget(answer=reference_code, metadata=metadata),
        scorer="code_reference_match",
        metadata=metadata,
    )


def prepare_ds1000(
    split: str,
    limit: int | None,
    seed: int,
    **_: object,
) -> list[DownstreamExample]:
    try:
        dataset = load_dataset(DATASET_NAME, split=split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load xlangai/DS-1000. Ensure the dataset is locally cached and supported by the installed datasets package."
        ) from exc

    if limit == 0:
        return []

    if limit is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))

    return [
        build_ds1000_example(dict(cast(dict[str, Any], row)), split, row_index)
        for row_index, row in enumerate(dataset)
    ]


PREPARE_REGISTRY[BENCHMARK_NAME] = cast(PrepareFn, prepare_ds1000)
