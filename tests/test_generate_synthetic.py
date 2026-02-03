import json

import pytest

from scripts.generate_synthetic import count_existing, load_corpus, write_pairs


def test_load_corpus_filters_and_limits(tmp_path) -> None:
    input_path = tmp_path / "input.jsonl"
    payloads = [
        {"text": "short"},
        {"content": "This is long enough to keep."},
        {"code": "def foo():\n    return 1\n" * 5},
    ]
    input_path.write_text(
        "\n".join(json.dumps(item) for item in payloads) + "\n",
        encoding="utf-8",
    )

    texts = load_corpus(input_path, limit=1)

    assert len(texts) == 1
    assert "long enough" in texts[0] or "def foo" in texts[0]


def test_count_existing_counts_non_empty_lines(tmp_path) -> None:
    output_path = tmp_path / "output.jsonl"
    output_path.write_text("{}\n\n{}\n", encoding="utf-8")

    assert count_existing(output_path) == 2


def test_write_pairs_writes_per_batch_on_failure(tmp_path) -> None:
    output_path = tmp_path / "output.jsonl"
    texts = [f"text {idx} long enough to keep" for idx in range(5)]

    class StubGenerator:
        def __init__(self) -> None:
            self.calls = 0

        def compress_batch(self, batch, show_progress: bool = True):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("boom")
            return [f"compressed {idx}" for idx in range(len(batch))]

    generator = StubGenerator()

    with pytest.raises(RuntimeError):
        write_pairs(
            texts,
            generator,
            output_path,
            domain="nl",
            batch_size=2,
            mode="w",
        )

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    assert all(item["domain"] == "nl" for item in payloads)
