from src.evaluation.downstream.scoring import (
    code_reference_exact_match,
    code_reference_match,
    exact_match,
    token_f1,
)


def test_exact_match_normalizes_case_and_whitespace() -> None:
    assert exact_match(" Alice ", "alice") == 1.0


def test_token_f1_rewards_partial_overlap() -> None:
    assert token_f1("alice won", "alice") > 0.0


def test_code_reference_match_accepts_normalized_substring_match() -> None:
    assert code_reference_match("result = df.iloc[list]", "df.iloc[List]") == 1.0


def test_code_reference_match_falls_back_to_token_f1() -> None:
    assert code_reference_match("df sort_values", "df.iloc[List]") == token_f1(
        "df sort_values", "df.iloc[List]"
    )


def test_code_reference_exact_match_is_binary() -> None:
    assert code_reference_exact_match("df sort_values", "df.iloc[List]") == 0.0
