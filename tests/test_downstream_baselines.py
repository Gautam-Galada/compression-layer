from src.evaluation.downstream.baselines import truncate_tokens


def test_truncate_tokens_shortens_text() -> None:
    text = "one two three four five six"
    truncated = truncate_tokens(text, max_tokens=3)
    assert truncated != text
    assert truncated.split() == ["one", "two", "three"]
