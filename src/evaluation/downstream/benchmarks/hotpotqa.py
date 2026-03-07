from typing import cast

from datasets import load_dataset

from src.evaluation.downstream.dataset import DownstreamExample, GoldTarget, RetrievedChunk
from src.evaluation.downstream.prepare import PREPARE_REGISTRY, PrepareFn

BENCHMARK_NAME = "hotpotqa"
SOURCE_SUBSET = "distractor"


def build_hotpotqa_example(row: dict, split: str) -> DownstreamExample:
    context = row["context"]
    titles = context["title"]
    sentence_groups = context["sentences"]
    retrieved_context = [
        RetrievedChunk(
            doc_id=title,
            rank=index,
            title=title,
            text=" ".join(sentences),
        )
        for index, (title, sentences) in enumerate(
            zip(titles, sentence_groups, strict=False), start=1
        )
    ]
    supporting_facts = row["supporting_facts"]
    evidence = [
        f"{title}::{sent_id}"
        for title, sent_id in zip(
            supporting_facts["title"], supporting_facts["sent_id"], strict=False
        )
    ]
    return DownstreamExample(
        id=str(row["id"]),
        benchmark=BENCHMARK_NAME,
        split=split,
        domain="nl",
        task_type="qa",
        query=str(row["question"]),
        retrieved_context=retrieved_context,
        gold=GoldTarget(answer=str(row["answer"]), evidence=evidence),
        scorer="qa_exact_f1",
        metadata={"source_subset": SOURCE_SUBSET},
    )


def prepare_hotpotqa(
    split: str,
    limit: int | None,
    seed: int,
    **_: object,
) -> list[DownstreamExample]:
    try:
        dataset = load_dataset("hotpotqa/hotpot_qa", SOURCE_SUBSET, split=split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load HotpotQA distractor split. Ensure the dataset is available in the local cache or that network access is enabled for the initial download."
        ) from exc
    if limit is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(limit, len(dataset))))
    return [build_hotpotqa_example(dict(row), split=split) for row in dataset]


PREPARE_REGISTRY[BENCHMARK_NAME] = cast(PrepareFn, prepare_hotpotqa)
