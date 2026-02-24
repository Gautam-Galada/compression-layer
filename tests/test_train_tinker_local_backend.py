import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.training.train_tinker as train_tinker


def _write_train_file(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"Compress:\\nu"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )


def test_train_on_tinker_local_backend_dispatches_to_mlx(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    _write_train_file(data_dir)

    output_dir = tmp_path / "out"
    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
    )
    config.backend = "local"

    called = {"value": False}

    def fake_local_backend(cfg: train_tinker.TinkerTrainingConfig):
        called["value"] = True
        return train_tinker.TinkerTrainingResult(success=True, adapter_path=cfg.output_dir)

    monkeypatch.setattr(train_tinker, "_train_with_local_mlx_backend", fake_local_backend)

    result = train_tinker.train_on_tinker(config)

    assert result.success is True
    assert called["value"] is True


def test_local_backend_writes_tinker_run_and_metrics_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    _write_train_file(data_dir)
    output_dir = tmp_path / "out"

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=1,
    )

    def fake_run(cmd, stdout=None, stderr=None, text=True):  # noqa: ARG001
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapters.safetensors").write_text("weights", encoding="utf-8")
        if stdout is not None:
            stdout.write("Iter 1: Train loss 0.1234\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_tinker.subprocess, "run", fake_run)
    monkeypatch.setattr(train_tinker, "check_mlx_available", lambda: True)

    result = train_tinker._train_with_local_mlx_backend(config)

    assert result.success is True
    assert (output_dir / "tinker_run.json").exists()
    assert (output_dir / "run.json").exists()
    assert (output_dir / "train.log").exists()
    assert (output_dir / "metrics.jsonl").exists()


def test_local_backend_uses_resume_adapter_file_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    _write_train_file(data_dir)
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    resume_adapter = output_dir / "adapters-000100.safetensors"
    resume_adapter.write_text("weights", encoding="utf-8")
    (output_dir / "tinker_run.json").write_text(
        json.dumps({"latest_checkpoint_path": str(resume_adapter)}),
        encoding="utf-8",
    )

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        epochs=1,
        batch_size=1,
        resume_from_checkpoint=True,
    )

    captured_cmd: dict[str, list[str]] = {}

    def fake_run(cmd, stdout=None, stderr=None, text=True):  # noqa: ARG001
        captured_cmd["cmd"] = list(cmd)
        (output_dir / "adapters.safetensors").write_text("weights", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(train_tinker.subprocess, "run", fake_run)
    monkeypatch.setattr(train_tinker, "check_mlx_available", lambda: True)

    result = train_tinker._train_with_local_mlx_backend(config)

    assert result.success is True
    assert "--resume-adapter-file" in captured_cmd["cmd"]
    resume_index = captured_cmd["cmd"].index("--resume-adapter-file")
    assert captured_cmd["cmd"][resume_index + 1] == str(resume_adapter)
