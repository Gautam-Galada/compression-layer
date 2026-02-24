from pathlib import Path
from types import SimpleNamespace

import scripts.evaluate_tinker as evaluate_tinker


def test_main_local_backend_uses_hf_loader_and_local_generator(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "eval.jsonl"
    data_path = tmp_path / "training" / "test.jsonl"

    args = SimpleNamespace(
        checkpoint_path=None,
        adapter_path=tmp_path / "adapters" / "tinker",
        data=data_path,
        limit=1,
        output=output_path,
        show_examples=0,
        resume=True,
        verbose=False,
        backend="local",
        model="mlx-community/Qwen3-4B-Instruct-2507-8bit",
        hf_dataset="Sudhendra/semantic-compression-sft",
    )

    monkeypatch.setattr(evaluate_tinker, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluate_tinker,
        "get_settings",
        lambda: SimpleNamespace(
            tinker_api_key="",
            adapters_dir=tmp_path / "adapters",
            data_dir=tmp_path,
        ),
    )

    calls: dict[str, object] = {}

    def fake_materialize(dataset_name: str, output_dir: Path) -> dict[str, int]:
        calls["dataset_name"] = dataset_name
        calls["output_dir"] = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "test.jsonl").write_text(
            '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"Compress:\\nhello"},{"role":"assistant","content":"h"}]}\n',
            encoding="utf-8",
        )
        (output_dir / "train.jsonl").write_text("", encoding="utf-8")
        (output_dir / "valid.jsonl").write_text("", encoding="utf-8")
        return {"train": 0, "valid": 0, "test": 1}

    monkeypatch.setattr(evaluate_tinker, "materialize_hf_chat_dataset", fake_materialize)

    examples = [
        evaluate_tinker.EvalExample(input_text="hello", expected_output="h", system_prompt="s")
    ]
    monkeypatch.setattr(evaluate_tinker, "load_test_examples", lambda *_args, **_kwargs: examples)

    def fake_create_local_generator(model: str, adapter_path: Path, verbose: bool = False):
        calls["local_model"] = model
        calls["adapter_path"] = adapter_path
        return lambda *_args, **_kwargs: ("h", 10, 3)

    monkeypatch.setattr(evaluate_tinker, "create_local_generator", fake_create_local_generator)
    monkeypatch.setattr(
        evaluate_tinker,
        "create_tinker_generator",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("should not be called")),
    )
    monkeypatch.setattr(
        evaluate_tinker,
        "run_evaluation",
        lambda *_args, **_kwargs: [
            evaluate_tinker.EvalResult(
                input_text="hello",
                expected_output="h",
                generated_output="h",
                input_tokens=10,
                output_tokens=3,
                compression_ratio=0.1,
                generation_time_ms=5.0,
            )
        ],
    )
    monkeypatch.setattr(evaluate_tinker, "print_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate_tinker, "print_examples", lambda *_args, **_kwargs: None)

    exit_code = evaluate_tinker.main()

    assert exit_code == 0
    assert calls["dataset_name"] == "Sudhendra/semantic-compression-sft"
    assert calls["local_model"] == "mlx-community/Qwen3-4B-Instruct-2507-8bit"


def test_local_backend_does_not_require_tinker_api_key(tmp_path: Path, monkeypatch) -> None:
    args = SimpleNamespace(
        checkpoint_path=None,
        adapter_path=tmp_path / "adapters" / "tinker",
        data=tmp_path / "training" / "test.jsonl",
        limit=1,
        output=tmp_path / "eval.jsonl",
        show_examples=0,
        resume=True,
        verbose=False,
        backend="local",
        model="mlx-community/Qwen3-4B-Instruct-2507-8bit",
        hf_dataset="Sudhendra/semantic-compression-sft",
    )

    monkeypatch.setattr(evaluate_tinker, "parse_args", lambda: args)
    monkeypatch.setattr(
        evaluate_tinker,
        "get_settings",
        lambda: SimpleNamespace(
            tinker_api_key="", adapters_dir=tmp_path / "adapters", data_dir=tmp_path
        ),
    )
    monkeypatch.setattr(
        evaluate_tinker,
        "materialize_hf_chat_dataset",
        lambda *_args, **_kwargs: (
            args.data.parent.mkdir(parents=True, exist_ok=True),
            args.data.write_text(
                '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"Compress:\\nhello"},{"role":"assistant","content":"h"}]}\n',
                encoding="utf-8",
            ),
            {"train": 0, "valid": 0, "test": 1},
        )[-1],
    )
    monkeypatch.setattr(
        evaluate_tinker,
        "load_test_examples",
        lambda *_args, **_kwargs: [
            evaluate_tinker.EvalExample(input_text="hello", expected_output="h", system_prompt="s")
        ],
    )
    monkeypatch.setattr(
        evaluate_tinker,
        "create_local_generator",
        lambda *_args, **_kwargs: (lambda *_a, **_k: ("h", 10, 3)),
    )
    monkeypatch.setattr(
        evaluate_tinker,
        "run_evaluation",
        lambda *_args, **_kwargs: [
            evaluate_tinker.EvalResult(
                input_text="hello",
                expected_output="h",
                generated_output="h",
                input_tokens=10,
                output_tokens=3,
                compression_ratio=0.1,
                generation_time_ms=5.0,
            )
        ],
    )
    monkeypatch.setattr(evaluate_tinker, "print_summary", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(evaluate_tinker, "print_examples", lambda *_args, **_kwargs: None)

    exit_code = evaluate_tinker.main()

    assert exit_code == 0
