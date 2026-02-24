from pathlib import Path
from types import SimpleNamespace

import scripts.train_tinker as train_tinker_cli


def _write_minimal_split_files(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    line = '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"Compress:\\nu"},{"role":"assistant","content":"a"}]}\n'
    (data_dir / "train.jsonl").write_text(line, encoding="utf-8")
    (data_dir / "valid.jsonl").write_text(line, encoding="utf-8")
    (data_dir / "test.jsonl").write_text(line, encoding="utf-8")


def test_train_cli_materializes_hf_dataset_before_training(tmp_path: Path, monkeypatch) -> None:
    settings = SimpleNamespace(
        tinker_api_key="test-key",
        data_dir=tmp_path / "data",
        adapters_dir=tmp_path / "models" / "adapters",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)

    calls: dict[str, object] = {}

    def fake_materialize(dataset_name: str, output_dir: Path) -> dict[str, int]:
        calls["dataset_name"] = dataset_name
        calls["output_dir"] = output_dir
        _write_minimal_split_files(output_dir)
        return {"train": 1, "valid": 1, "test": 1}

    captured: dict[str, object] = {}

    def fake_run_train(config, _settings) -> int:
        captured["dataset_path"] = config.dataset_path
        return 0

    monkeypatch.setattr(train_tinker_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        train_tinker_cli,
        "materialize_hf_chat_dataset",
        fake_materialize,
        raising=False,
    )
    monkeypatch.setattr(train_tinker_cli, "run_train", fake_run_train)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train_tinker.py",
            "--output",
            str(tmp_path / "models" / "adapters" / "tinker"),
        ],
    )

    exit_code = train_tinker_cli.main()

    assert exit_code == 0
    assert calls["dataset_name"] == "Sudhendra/semantic-compression-sft"
    assert isinstance(calls["output_dir"], Path)
    assert captured["dataset_path"] == calls["output_dir"]


def test_train_cli_uses_hf_default_dataset_name(tmp_path: Path, monkeypatch) -> None:
    settings = SimpleNamespace(
        tinker_api_key="test-key",
        data_dir=tmp_path / "data",
        adapters_dir=tmp_path / "models" / "adapters",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)

    seen_dataset_name: dict[str, str] = {}

    def fake_materialize(dataset_name: str, output_dir: Path) -> dict[str, int]:
        seen_dataset_name["name"] = dataset_name
        _write_minimal_split_files(output_dir)
        return {"train": 1, "valid": 1, "test": 1}

    monkeypatch.setattr(train_tinker_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        train_tinker_cli,
        "materialize_hf_chat_dataset",
        fake_materialize,
        raising=False,
    )
    monkeypatch.setattr(train_tinker_cli, "run_train", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("sys.argv", ["train_tinker.py"])

    exit_code = train_tinker_cli.main()

    assert exit_code == 0
    assert seen_dataset_name["name"] == "Sudhendra/semantic-compression-sft"


def test_train_cli_local_backend_does_not_require_tinker_api_key(
    tmp_path: Path, monkeypatch
) -> None:
    settings = SimpleNamespace(
        tinker_api_key="",
        data_dir=tmp_path / "data",
        adapters_dir=tmp_path / "models" / "adapters",
    )
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.adapters_dir.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(train_tinker_cli, "get_settings", lambda: settings)
    monkeypatch.setattr(
        train_tinker_cli,
        "materialize_hf_chat_dataset",
        lambda dataset_name, output_dir: (
            _write_minimal_split_files(output_dir),
            {"train": 1, "valid": 1, "test": 1},
        )[-1],
        raising=False,
    )
    monkeypatch.setattr(train_tinker_cli, "run_train", lambda *_args, **_kwargs: 0)
    monkeypatch.setattr("sys.argv", ["train_tinker.py", "--backend", "local"])

    exit_code = train_tinker_cli.main()

    assert exit_code == 0
