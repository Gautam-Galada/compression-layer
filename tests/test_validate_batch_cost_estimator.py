import pytest

from scripts.validate_batch import estimate_validation_cost
from src.utils.costs import MODEL_PRICING


def test_gemini_flash_pricing_matches_latest_published_rate() -> None:
    assert MODEL_PRICING["gemini-2.0-flash"] == (0.15, 0.60)


def test_estimator_uses_selected_models_not_global_average() -> None:
    cost = estimate_validation_cost(
        num_pairs=100,
        model_ids=["gemini-2.0-flash"],
        num_tasks=2,
        avg_input_tokens_per_call=500,
        avg_output_tokens_per_call=250,
    )

    # 100 pairs * 2 tasks * 2 prompts (verbose + compressed) = 400 calls
    # Input: 400 * 500 = 200,000 tokens at $0.15 / 1M  => $0.03
    # Output: 400 * 250 = 100,000 tokens at $0.60 / 1M => $0.06
    assert cost == pytest.approx(0.09)


def test_estimator_includes_llm_judge_cost_when_enabled() -> None:
    no_judge = estimate_validation_cost(
        num_pairs=200,
        model_ids=["claude-sonnet-4-20250514", "gpt-4o-mini", "gemini-2.0-flash"],
        num_tasks=2,
        avg_input_tokens_per_call=500,
        avg_output_tokens_per_call=250,
        use_llm_judge=False,
    )

    with_judge = estimate_validation_cost(
        num_pairs=200,
        model_ids=["claude-sonnet-4-20250514", "gpt-4o-mini", "gemini-2.0-flash"],
        num_tasks=2,
        avg_input_tokens_per_call=500,
        avg_output_tokens_per_call=250,
        use_llm_judge=True,
        llm_judge_model="claude-sonnet-4-20250514",
        avg_judge_input_tokens_per_call=700,
        avg_judge_output_tokens_per_call=120,
    )

    assert with_judge > no_judge
