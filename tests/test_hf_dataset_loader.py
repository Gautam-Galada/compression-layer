import json
from pathlib import Path

import pytest

from src.training.hf_dataset_loader import materialize_hf_chat_dataset


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_materialize_hf_dataset_writes_train_valid_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_dataset = {
        "train": [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "Compress:\nhello"},
                    {"role": "assistant", "content": "h"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "Compress:\nworld"},
                    {"role": "assistant", "content": "w"},
                ]
            },
        ],
        "validation": [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "Compress:\nval"},
                    {"role": "assistant", "content": "v"},
                ]
            }
        ],
        "test": [
            {
                "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "Compress:\ntest"},
                    {"role": "assistant", "content": "t"},
                ]
            }
        ],
    }

    monkeypatch.setattr("src.training.hf_dataset_loader.load_dataset", lambda name: fake_dataset)

    counts = materialize_hf_chat_dataset("Sudhendra/semantic-compression-sft", tmp_path)

    assert counts == {"train": 2, "valid": 1, "test": 1}
    assert (tmp_path / "train.jsonl").exists()
    assert (tmp_path / "valid.jsonl").exists()
    assert (tmp_path / "test.jsonl").exists()
    assert len(_read_jsonl(tmp_path / "train.jsonl")) == 2


def test_materialize_hf_dataset_maps_validation_to_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_dataset = {
        "train": [],
        "validation": [{"messages": [{"role": "user", "content": "u"}]}],
        "test": [],
    }
    monkeypatch.setattr("src.training.hf_dataset_loader.load_dataset", lambda name: fake_dataset)

    materialize_hf_chat_dataset("Sudhendra/semantic-compression-sft", tmp_path)

    assert (tmp_path / "valid.jsonl").exists()
    assert not (tmp_path / "validation.jsonl").exists()


def test_materialize_hf_dataset_rejects_missing_messages_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_dataset = {
        "train": [{"text": "invalid"}],
        "validation": [],
        "test": [],
    }
    monkeypatch.setattr("src.training.hf_dataset_loader.load_dataset", lambda name: fake_dataset)

    with pytest.raises(ValueError, match="messages"):
        materialize_hf_chat_dataset("Sudhendra/semantic-compression-sft", tmp_path)
