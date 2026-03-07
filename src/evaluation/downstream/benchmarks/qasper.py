import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, cast

from datasets import load_dataset

from src.evaluation.downstream.dataset import DownstreamExample, GoldTarget, RetrievedChunk
from src.evaluation.downstream.prepare import PREPARE_REGISTRY, PrepareFn

BENCHMARK_NAME = "qasper"

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class _ParagraphRecord:
    index: int
    text: str
    section_name: str | None
    occurrence: int


def _normalize_paragraph_text(text: str) -> str:
    return text.strip()


def _build_paragraph_records(
    paragraphs: list[str],
    section_names: list[str | None] | None = None,
) -> list[_ParagraphRecord]:
    normalized_counts: Counter[str] = Counter()
    records: list[_ParagraphRecord] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        normalized_text = _normalize_paragraph_text(str(paragraph))
        if not normalized_text:
            continue
        normalized_counts[normalized_text] += 1
        records.append(
            _ParagraphRecord(
                index=index,
                text=normalized_text,
                section_name=section_names[index - 1] if section_names is not None else None,
                occurrence=normalized_counts[normalized_text],
            )
        )
    return records


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


def _lexical_overlap_score(question: str, paragraph: str) -> tuple[int, int]:
    question_tokens = _tokenize(question)
    paragraph_tokens = _tokenize(paragraph)
    overlap = question_tokens & paragraph_tokens
    return (len(overlap), len(paragraph_tokens))


def _select_qasper_paragraph_records(
    question: str,
    paragraphs: list[_ParagraphRecord],
    evidence: list[str],
    max_chunks: int,
) -> list[_ParagraphRecord]:
    evidence_counts = Counter(
        _normalize_paragraph_text(text) for text in evidence if _normalize_paragraph_text(text)
    )
    selected: list[_ParagraphRecord] = []
    selected_indices: set[int] = set()

    for paragraph in paragraphs:
        remaining = evidence_counts.get(paragraph.text, 0)
        if remaining <= 0 or paragraph.occurrence > remaining:
            continue
        selected.append(paragraph)
        selected_indices.add(paragraph.index)

    if max_chunks <= 0:
        return selected

    ranked_distractors = sorted(
        (
            (_lexical_overlap_score(question, paragraph.text), paragraph.index, paragraph)
            for paragraph in paragraphs
            if paragraph.index not in selected_indices
        ),
        key=lambda item: (-item[0][0], -item[0][1], item[1]),
    )
    for distractors_added, (_, _, paragraph) in enumerate(ranked_distractors, start=1):
        selected.append(paragraph)
        if distractors_added >= max_chunks:
            break
    return selected


def select_qasper_paragraphs(
    question: str,
    paragraphs: list[str],
    evidence: list[str],
    max_chunks: int,
) -> list[str]:
    paragraph_records = _build_paragraph_records(paragraphs)
    return [
        paragraph.text
        for paragraph in _select_qasper_paragraph_records(
            question=question,
            paragraphs=paragraph_records,
            evidence=evidence,
            max_chunks=max_chunks,
        )
    ]


def _flatten_paragraphs(row: dict[str, Any]) -> list[_ParagraphRecord]:
    full_text = row.get("full_text") or {}
    section_names = list(full_text.get("section_name") or [])
    paragraph_groups = list(full_text.get("paragraphs") or [])

    paragraph_texts: list[str] = []
    paragraph_sections: list[str | None] = []
    for section_index, paragraph_group in enumerate(paragraph_groups):
        section_name = section_names[section_index] if section_index < len(section_names) else None
        for paragraph in paragraph_group or []:
            paragraph_texts.append(str(paragraph))
            paragraph_sections.append(section_name)
    return _build_paragraph_records(paragraph_texts, section_names=paragraph_sections)


def _extract_answer_text(answer: dict[str, Any]) -> str | None:
    extractive_spans = [
        str(span).strip() for span in answer.get("extractive_spans") or [] if str(span).strip()
    ]
    if extractive_spans:
        return extractive_spans[0]

    free_form_answer = str(answer.get("free_form_answer") or "").strip()
    if free_form_answer:
        return free_form_answer

    yes_no = answer.get("yes_no")
    if yes_no is True:
        return "yes"
    if yes_no is False:
        return "no"
    return None


def _first_answer(answers_entry: dict[str, Any]) -> dict[str, Any] | None:
    for answer in answers_entry.get("answer") or []:
        answer_dict = dict(answer)
        if answer_dict.get("unanswerable"):
            continue
        if _extract_answer_text(answer_dict) is None:
            continue
        return answer_dict
    return None


def prepare_qasper(
    split: str,
    limit: int | None,
    seed: int,
    max_chunks: int = 6,
    **_: object,
) -> list[DownstreamExample]:
    try:
        dataset = load_dataset("allenai/qasper", split=split)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load allenai/qasper. Ensure the dataset is locally cached and supported by the installed datasets package."
        ) from exc

    if limit == 0:
        return []

    shuffled_dataset = dataset.shuffle(seed=seed) if limit is not None else dataset
    examples: list[DownstreamExample] = []

    for row in shuffled_dataset:
        row_dict = dict(row)
        paper_id = str(row_dict["id"])
        title = str(row_dict.get("title") or "")
        flattened_paragraphs = _flatten_paragraphs(row_dict)

        qas = row_dict.get("qas") or {}
        questions = list(qas.get("question") or [])
        question_ids = list(qas.get("question_id") or [])
        answers = list(qas.get("answers") or [])

        for index, question in enumerate(questions):
            answer_entry = answers[index] if index < len(answers) else {}
            answer = _first_answer(dict(answer_entry))
            if answer is None:
                continue

            evidence = [
                str(paragraph).strip()
                for paragraph in answer.get("evidence") or []
                if str(paragraph).strip() and not str(paragraph).startswith("FLOAT SELECTED")
            ]
            selected_paragraph_records = _select_qasper_paragraph_records(
                question=str(question),
                paragraphs=flattened_paragraphs,
                evidence=evidence,
                max_chunks=max_chunks,
            )
            retrieved_context = [
                RetrievedChunk(
                    doc_id=f"{paper_id}::{paragraph.index}",
                    rank=rank,
                    title=title,
                    text=paragraph.text,
                    metadata={"section_name": paragraph.section_name},
                )
                for rank, paragraph in enumerate(selected_paragraph_records, start=1)
            ]
            answer_text = _extract_answer_text(answer)
            if answer_text is None:
                continue

            question_id = (
                str(question_ids[index]) if index < len(question_ids) else f"{paper_id}::{index}"
            )
            examples.append(
                DownstreamExample(
                    id=question_id,
                    benchmark=BENCHMARK_NAME,
                    split=split,
                    domain="nl",
                    task_type="qa",
                    query=str(question),
                    retrieved_context=retrieved_context,
                    gold=GoldTarget(answer=answer_text, evidence=evidence),
                    scorer="qa_exact_f1",
                    metadata={"paper_id": paper_id},
                )
            )
            if limit is not None and len(examples) >= limit:
                return examples
    return examples


PREPARE_REGISTRY[BENCHMARK_NAME] = cast(PrepareFn, prepare_qasper)
