from scripts.evaluate_tinker import (
    _build_user_message,
    _cap_output_length,
    _compute_generation_budget,
    _load_existing_results,
    _strip_generation_artifacts,
    _truncate_repetition,
)


def test_build_user_message_uses_training_format() -> None:
    assert _build_user_message("hello") == "Compress:\nhello"


def test_compute_generation_budget_tightens_decoding() -> None:
    assert _compute_generation_budget(input_tokens=100, requested_max_tokens=512) == 60
    assert _compute_generation_budget(input_tokens=400, requested_max_tokens=512) == 240
    assert _compute_generation_budget(input_tokens=10, requested_max_tokens=512) == 24
    assert _compute_generation_budget(input_tokens=400, requested_max_tokens=128) == 128


def test_strip_generation_artifacts_removes_tags() -> None:
    raw = "<think>scratch</think>final answer</tool_call><|im_end|>"
    assert _strip_generation_artifacts(raw) == "final answer"


def test_truncate_repetition_stops_after_duplicate_clause() -> None:
    text = "a = one | b = two | c = three | b = two | c = three | d = four"
    assert _truncate_repetition(text) == "a = one | b = two | c = three"


def test_cap_output_length_prevents_expansion() -> None:
    input_text = "x" * 100
    output_text = "a" * 250
    capped = _cap_output_length(output_text, input_text)
    assert len(capped) <= 95


def test_load_existing_results_reads_valid_jsonl_rows(tmp_path) -> None:
    path = tmp_path / "eval.jsonl"
    path.write_text(
        '{"input_text":"a","expected_output":"x","generated_output":"y","input_tokens":10,"output_tokens":4,"compression_ratio":0.4,"generation_time_ms":123.0}\n'
        '{"input_text":"b","expected_output":"m","generated_output":"n","input_tokens":12,"output_tokens":5,"compression_ratio":0.42,"generation_time_ms":124.0}\n',
        encoding="utf-8",
    )

    loaded = _load_existing_results(path)
    assert len(loaded) == 2
    assert loaded[0].input_text == "a"
    assert loaded[1].output_tokens == 5


def test_load_existing_results_stops_at_partial_row(tmp_path) -> None:
    path = tmp_path / "eval_partial.jsonl"
    path.write_text(
        '{"input_text":"a","expected_output":"x","generated_output":"y","input_tokens":10,"output_tokens":4,"compression_ratio":0.4,"generation_time_ms":123.0}\n'
        '{"input_text":"b","expected_output":"m"',
        encoding="utf-8",
    )

    loaded = _load_existing_results(path)
    assert len(loaded) == 1
    assert loaded[0].input_text == "a"
