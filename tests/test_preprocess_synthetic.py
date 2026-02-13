import json

from scripts.preprocess_synthetic import PreprocessConfig, clean_pair, preprocess_file


def test_clean_pair_removes_think_and_tool_tags() -> None:
    pair = {
        "verbose": "Artificial intelligence improves quality control in aerospace manufacturing.",
        "compressed": "<think>\n<tool_call>\nAI improves aerospace QC",
        "domain": "nl",
        "metadata": {},
    }

    cleaned, reason = clean_pair(pair, PreprocessConfig(max_token_ratio=2.0))

    assert reason is None
    assert cleaned is not None
    assert cleaned["compressed"] == "AI improves aerospace QC"


def test_clean_pair_rejects_char_expansion() -> None:
    pair = {
        "verbose": "Short source text.",
        "compressed": "This compressed output is now clearly longer than the source text.",
        "domain": "nl",
        "metadata": {},
    }

    cleaned, reason = clean_pair(pair, PreprocessConfig(max_token_ratio=2.0))

    assert cleaned is None
    assert reason == "char_ratio_ge_1.0"


def test_preprocess_file_writes_clean_and_rejected_samples(tmp_path) -> None:
    input_path = tmp_path / "synthetic.jsonl"
    output_path = tmp_path / "validated.jsonl"
    rejected_path = tmp_path / "rejected.jsonl"

    records = [
        {
            "verbose": "Artificial intelligence improves quality control in aerospace manufacturing.",
            "compressed": "<think>\nAI improves aerospace QC",
            "domain": "nl",
            "metadata": {},
        },
        {
            "verbose": "Short source text.",
            "compressed": "This compressed output is now clearly longer than the source text.",
            "domain": "nl",
            "metadata": {},
        },
        {
            "verbose": "def add(a, b):\n    return a + b",
            "compressed": "def add(a, b):\n    return a + b",
            "domain": "code",
            "metadata": {},
        },
    ]

    input_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

    stats = preprocess_file(
        input_path=input_path,
        output_path=output_path,
        rejected_path=rejected_path,
        config=PreprocessConfig(max_token_ratio=2.0),
    )

    assert stats.total == 3
    assert stats.passed == 1
    assert stats.rejected == 2
    assert stats.rejected_by_reason["char_ratio_ge_1.0"] == 2

    kept_lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(kept_lines) == 1
    kept = json.loads(kept_lines[0])
    assert kept["compressed"] == "AI improves aerospace QC"
    assert kept["validation"]["passed"] is True

    rejected_lines = rejected_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(rejected_lines) == 2
    rejected = [json.loads(line) for line in rejected_lines]
    assert all(item["reason"] == "char_ratio_ge_1.0" for item in rejected)
