"""Tests for the Tinker training CLI."""

from pathlib import Path

import pytest

from scripts.train_tinker import main


def test_cli_status_reads_run_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "run-123"
    (tmp_path / "runs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "runs" / f"{run_id}.json").write_text("{}")
    exit_code = main(["--status", run_id, "--output", str(tmp_path)])
    assert exit_code == 0
