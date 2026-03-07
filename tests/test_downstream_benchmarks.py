import inspect

import pytest

from src.evaluation.downstream.benchmarks import ds1000, hotpotqa, qasper
from src.evaluation.downstream.benchmarks.ds1000 import build_ds1000_example
from src.evaluation.downstream.benchmarks.hotpotqa import build_hotpotqa_example
from src.evaluation.downstream.benchmarks.qasper import select_qasper_paragraphs


def test_build_hotpotqa_example_uses_context_and_supporting_facts() -> None:
    row = {
        "id": "abc",
        "question": "Who won?",
        "answer": "Alice",
        "context": {
            "title": ["Doc A", "Doc B"],
            "sentences": [["Alice won."], ["Bob lost."]],
        },
        "supporting_facts": {"title": ["Doc A"], "sent_id": [0]},
    }

    example = build_hotpotqa_example(row, split="validation")

    assert example.benchmark == "hotpotqa"
    assert example.gold.answer == "Alice"
    assert example.retrieved_context[0].title == "Doc A"
    assert example.gold.evidence == ["Doc A::0"]


def test_build_ds1000_example_uses_prompt_and_code_context() -> None:
    row = {
        "prompt": "Use pandas to reorder rows.",
        "code_context": "import pandas as pd\nimport numpy as np",
        "reference_code": "df.iloc[List]",
        "metadata": {"problem_id": 1, "library": "Pandas"},
    }

    example = build_ds1000_example(row, split="test", row_index=0)

    assert example.domain == "code"
    assert example.task_type == "code_gen"
    assert example.retrieved_context[0].text.startswith("import pandas")


def test_prepare_ds1000_builds_code_examples_and_registers_preparer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def shuffle(self, seed: int) -> "FakeDataset":
            assert seed == 9
            return self

        def select(self, indices: range) -> "FakeDataset":
            selected = FakeDataset()
            selected.rows = [self.rows[index] for index in indices]
            return selected

        def __init__(self) -> None:
            self.rows = [
                {
                    "prompt": "Use pandas to reorder rows.",
                    "code_context": "import pandas as pd",
                    "reference_code": "df.iloc[List]",
                    "metadata": {"problem_id": 1, "library": "Pandas"},
                }
            ]

        def __len__(self) -> int:
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

    monkeypatch.setattr(ds1000, "load_dataset", lambda *args, **kwargs: FakeDataset())

    examples = ds1000.prepare_ds1000(split="test", limit=1, seed=9)

    assert ds1000.BENCHMARK_NAME in ds1000.PREPARE_REGISTRY
    assert len(examples) == 1
    assert examples[0].benchmark == "ds1000"
    assert examples[0].scorer == "code_reference_match"
    assert examples[0].gold.answer == "df.iloc[List]"
    assert examples[0].metadata["library"] == "Pandas"
    assert examples[0].metadata["dataset"] == "xlangai/DS-1000"
    assert examples[0].metadata["source"] == "xlangai/DS-1000"
    assert examples[0].gold.metadata["problem_id"] == 1
    assert examples[0].gold.metadata["dataset"] == "xlangai/DS-1000"
    assert examples[0].gold.metadata["source"] == "xlangai/DS-1000"


def test_prepare_hotpotqa_clamps_oversized_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeDataset:
        def __init__(self) -> None:
            self.rows = [
                {
                    "id": "1",
                    "question": "Q1",
                    "answer": "A1",
                    "context": {"title": ["Doc 1"], "sentences": [["One"]]},
                    "supporting_facts": {"title": ["Doc 1"], "sent_id": [0]},
                },
                {
                    "id": "2",
                    "question": "Q2",
                    "answer": "A2",
                    "context": {"title": ["Doc 2"], "sentences": [["Two"]]},
                    "supporting_facts": {"title": ["Doc 2"], "sent_id": [0]},
                },
            ]

        def shuffle(self, seed: int) -> "FakeDataset":
            assert seed == 7
            return self

        def select(self, indices: range) -> "FakeDataset":
            selected = FakeDataset()
            selected.rows = [self.rows[index] for index in indices]
            return selected

        def __len__(self) -> int:
            return len(self.rows)

        def __iter__(self):
            return iter(self.rows)

    monkeypatch.setattr(hotpotqa, "load_dataset", lambda *args, **kwargs: FakeDataset())

    examples = hotpotqa.prepare_hotpotqa(split="validation", limit=5, seed=7)

    assert [example.id for example in examples] == ["1", "2"]


def test_prepare_hotpotqa_preserves_full_context_with_extra_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def __iter__(self):
            return iter(
                [
                    {
                        "id": "1",
                        "question": "Who won?",
                        "answer": "Alice",
                        "context": {
                            "title": ["Doc A", "Doc B", "Doc C"],
                            "sentences": [["A"], ["B"], ["C"]],
                        },
                        "supporting_facts": {"title": ["Doc A"], "sent_id": [0]},
                    }
                ]
            )

    monkeypatch.setattr(hotpotqa, "load_dataset", lambda *args, **kwargs: FakeDataset())

    examples = hotpotqa.prepare_hotpotqa(split="validation", limit=None, seed=42, max_chunks=2)

    assert len(examples) == 1
    assert [chunk.title for chunk in examples[0].retrieved_context] == ["Doc A", "Doc B", "Doc C"]


def test_prepare_hotpotqa_has_required_public_signature() -> None:
    signature = inspect.signature(hotpotqa.prepare_hotpotqa)
    parameters = list(signature.parameters.values())

    assert [parameter.name for parameter in parameters] == ["split", "limit", "seed", "_"]
    assert [parameter.kind for parameter in parameters] == [
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.VAR_KEYWORD,
    ]
    assert parameters[0].default is inspect.Parameter.empty
    assert parameters[1].default is inspect.Parameter.empty
    assert parameters[2].default is inspect.Parameter.empty


def test_select_qasper_paragraphs_keeps_evidence_and_adds_distractors() -> None:
    question = "What is the seed lexicon?"
    paragraphs = [
        "This paragraph is unrelated.",
        "The seed lexicon contains 60 positive and 60 negative predicates.",
        "More unrelated text.",
    ]
    evidence = ["The seed lexicon contains 60 positive and 60 negative predicates."]

    selected = select_qasper_paragraphs(question, paragraphs, evidence, max_chunks=2)

    assert evidence[0] in selected
    assert len(selected) == 3


def test_select_qasper_paragraphs_preserves_all_evidence_beyond_max_chunks() -> None:
    question = "What evidence supports the answer?"
    paragraphs = [
        "Evidence paragraph one.",
        "Evidence paragraph two.",
        "Evidence paragraph three.",
        "Relevant distractor paragraph.",
        "Unrelated distractor paragraph.",
    ]
    evidence = [
        "Evidence paragraph one.",
        "Evidence paragraph two.",
        "Evidence paragraph three.",
    ]

    selected = select_qasper_paragraphs(question, paragraphs, evidence, max_chunks=2)

    assert selected[:3] == evidence
    assert len(selected) == 5


def test_prepare_qasper_flattens_questions_and_keeps_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def shuffle(self, seed: int) -> "FakeDataset":
            assert seed == 11
            return self

        def __iter__(self):
            return iter(
                [
                    {
                        "id": "paper-1",
                        "title": "Paper Title",
                        "abstract": "Paper abstract.",
                        "full_text": {
                            "section_name": ["Intro", "Method"],
                            "paragraphs": [
                                ["This paragraph is unrelated."],
                                [
                                    "The seed lexicon contains 60 positive and 60 negative predicates.",
                                    "More unrelated text.",
                                ],
                            ],
                        },
                        "qas": {
                            "question": ["What is the seed lexicon?"],
                            "question_id": ["q1"],
                            "answers": [
                                {
                                    "answer": [
                                        {
                                            "unanswerable": False,
                                            "extractive_spans": [
                                                "60 positive and 60 negative predicates"
                                            ],
                                            "free_form_answer": "",
                                            "yes_no": None,
                                            "evidence": [
                                                "The seed lexicon contains 60 positive and 60 negative predicates."
                                            ],
                                        }
                                    ]
                                }
                            ],
                        },
                    }
                ]
            )

    monkeypatch.setattr(qasper, "load_dataset", lambda *args, **kwargs: FakeDataset())

    examples = qasper.prepare_qasper("validation", limit=1, seed=11, max_chunks=2)

    assert len(examples) == 1
    assert examples[0].id == "q1"
    assert examples[0].benchmark == "qasper"
    assert examples[0].domain == "nl"
    assert examples[0].task_type == "qa"
    assert examples[0].scorer == "qa_exact_f1"
    assert examples[0].gold.answer == "60 positive and 60 negative predicates"
    assert examples[0].gold.evidence == [
        "The seed lexicon contains 60 positive and 60 negative predicates."
    ]
    assert len(examples[0].retrieved_context) == 3
    assert any(
        chunk.text == "The seed lexicon contains 60 positive and 60 negative predicates."
        for chunk in examples[0].retrieved_context
    )


def test_prepare_qasper_preserves_all_evidence_even_when_it_exceeds_max_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def shuffle(self, seed: int) -> "FakeDataset":
            assert seed == 13
            return self

        def __iter__(self):
            return iter(
                [
                    {
                        "id": "paper-2",
                        "title": "Paper Title",
                        "full_text": {
                            "section_name": ["Intro", "Method"],
                            "paragraphs": [
                                ["Evidence paragraph one.", "Evidence paragraph two."],
                                [
                                    "Evidence paragraph three.",
                                    "Relevant distractor paragraph.",
                                    "Unrelated distractor paragraph.",
                                ],
                            ],
                        },
                        "qas": {
                            "question": ["What evidence supports the answer?"],
                            "question_id": ["q2"],
                            "answers": [
                                {
                                    "answer": [
                                        {
                                            "unanswerable": False,
                                            "extractive_spans": ["Answer"],
                                            "free_form_answer": "",
                                            "yes_no": None,
                                            "evidence": [
                                                "Evidence paragraph one.",
                                                "Evidence paragraph two.",
                                                "Evidence paragraph three.",
                                            ],
                                        }
                                    ]
                                }
                            ],
                        },
                    }
                ]
            )

    monkeypatch.setattr(qasper, "load_dataset", lambda *args, **kwargs: FakeDataset())

    examples = qasper.prepare_qasper("validation", limit=1, seed=13, max_chunks=2)

    assert [chunk.text for chunk in examples[0].retrieved_context[:3]] == [
        "Evidence paragraph one.",
        "Evidence paragraph two.",
        "Evidence paragraph three.",
    ]
    assert len(examples[0].retrieved_context) == 5


def test_prepare_qasper_tracks_duplicate_paragraph_text_by_position(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeDataset:
        def shuffle(self, seed: int) -> "FakeDataset":
            assert seed == 17
            return self

        def __iter__(self):
            return iter(
                [
                    {
                        "id": "paper-3",
                        "title": "Paper Title",
                        "full_text": {
                            "section_name": ["Intro", "Method"],
                            "paragraphs": [
                                ["Repeated paragraph.", "Question distractor paragraph."],
                                ["Repeated paragraph.", "Another distractor paragraph."],
                            ],
                        },
                        "qas": {
                            "question": ["Which repeated paragraph is evidence?"],
                            "question_id": ["q3"],
                            "answers": [
                                {
                                    "answer": [
                                        {
                                            "unanswerable": False,
                                            "extractive_spans": ["Repeated"],
                                            "free_form_answer": "",
                                            "yes_no": None,
                                            "evidence": [
                                                "Repeated paragraph.",
                                                "Repeated paragraph.",
                                            ],
                                        }
                                    ]
                                }
                            ],
                        },
                    }
                ]
            )

    monkeypatch.setattr(qasper, "load_dataset", lambda *args, **kwargs: FakeDataset())

    examples = qasper.prepare_qasper("validation", limit=1, seed=17, max_chunks=1)

    assert [chunk.doc_id for chunk in examples[0].retrieved_context[:2]] == [
        "paper-3::1",
        "paper-3::3",
    ]
    assert [chunk.metadata["section_name"] for chunk in examples[0].retrieved_context[:2]] == [
        "Intro",
        "Method",
    ]


def test_select_qasper_paragraphs_normalizes_whitespace_and_picks_first_duplicate() -> None:
    question = "Which repeated paragraph is evidence?"
    paragraphs = [
        "  Repeated paragraph.  ",
        "Repeated paragraph.",
        "Question distractor paragraph.",
    ]
    evidence = ["Repeated paragraph."]

    selected = select_qasper_paragraphs(question, paragraphs, evidence, max_chunks=1)

    assert selected == ["Repeated paragraph.", "Repeated paragraph."]
