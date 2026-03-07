import argparse
from types import SimpleNamespace
from typing import cast

import pytest

from scripts.prepare_downstream_eval import build_parser
from src.evaluation.downstream import prepare
from src.evaluation.downstream.prepare import get_prepare_registry, get_supported_benchmarks


def _get_action(parser: argparse.ArgumentParser, dest: str) -> argparse.Action:
    for action in parser._actions:
        if action.dest == dest:
            return action
    raise AssertionError(f"missing action for {dest}")


def test_prepare_cli_accepts_hotpotqa_args() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--benchmark",
            "hotpotqa",
            "--split",
            "validation",
            "--limit",
            "10",
            "--output",
            "data/eval/downstream/hotpotqa_validation_10.jsonl",
        ]
    )

    assert args.benchmark == "hotpotqa"
    assert args.split == "validation"
    assert args.limit == 10


def test_prepare_cli_uses_supported_benchmark_names() -> None:
    parser = build_parser()
    registry_names = tuple(sorted(get_prepare_registry()))
    choices = _get_action(parser, "benchmark").choices

    assert choices is not None
    assert tuple(choices) == registry_names


def test_supported_benchmarks_come_from_registered_preparers(monkeypatch) -> None:
    monkeypatch.setattr(prepare, "load_benchmark_modules", lambda: None)
    monkeypatch.setattr(prepare, "discover_benchmark_names", lambda: ("qasper",))
    monkeypatch.setattr(
        prepare,
        "PREPARE_REGISTRY",
        cast(dict[str, prepare.PrepareFn], {"qasper": cast(prepare.PrepareFn, object())}),
    )

    assert get_supported_benchmarks() == ("qasper",)


def test_get_prepare_registry_fails_when_discovered_module_is_unregistered(monkeypatch) -> None:
    discovered = ("ds1000", "helper", "hotpotqa", "qasper")

    monkeypatch.setattr(prepare, "load_benchmark_modules", lambda: None)
    monkeypatch.setattr(
        prepare,
        "discover_benchmark_names",
        lambda: discovered,
    )
    monkeypatch.setattr(
        prepare,
        "PREPARE_REGISTRY",
        cast(
            dict[str, prepare.PrepareFn],
            {name: cast(prepare.PrepareFn, object()) for name in discovered if name != "helper"},
        ),
    )

    with pytest.raises(ValueError, match="helper"):
        get_prepare_registry()


def test_get_prepare_registry_ignores_helper_modules_without_benchmark_marker(monkeypatch) -> None:
    modules = {
        "src.evaluation.downstream.benchmarks": SimpleNamespace(__path__=["benchmarks"]),
        "src.evaluation.downstream.benchmarks.ds1000": SimpleNamespace(BENCHMARK_NAME="ds1000"),
        "src.evaluation.downstream.benchmarks.hotpotqa": SimpleNamespace(BENCHMARK_NAME="hotpotqa"),
        "src.evaluation.downstream.benchmarks.qasper": SimpleNamespace(BENCHMARK_NAME="qasper"),
        "src.evaluation.downstream.benchmarks.common": SimpleNamespace(),
    }

    monkeypatch.setattr(
        prepare,
        "iter_modules",
        lambda _paths: [
            SimpleNamespace(name="ds1000"),
            SimpleNamespace(name="hotpotqa"),
            SimpleNamespace(name="qasper"),
            SimpleNamespace(name="common"),
        ],
    )
    monkeypatch.setattr(prepare, "import_module", lambda name: modules[name])

    assert prepare.discover_benchmark_names() == ("ds1000", "hotpotqa", "qasper")


def test_prepare_cli_rejects_negative_limit(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--benchmark",
                "hotpotqa",
                "--split",
                "validation",
                "--limit",
                "-1",
                "--output",
                "data/eval/downstream/hotpotqa_validation.jsonl",
            ]
        )

    assert "--limit" in capsys.readouterr().err


def test_prepare_cli_rejects_negative_max_chunks(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--benchmark",
                "hotpotqa",
                "--split",
                "validation",
                "--max-chunks",
                "-1",
                "--output",
                "data/eval/downstream/hotpotqa_validation.jsonl",
            ]
        )

    assert "--max-chunks" in capsys.readouterr().err
