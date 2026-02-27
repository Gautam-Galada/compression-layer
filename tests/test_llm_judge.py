"""Tests for LLM judge response parsing."""

from src.validation.llm_judge import EquivalenceVerdict, LLMJudge


def _judge() -> LLMJudge:
    """Create a judge instance without initializing API clients."""
    return LLMJudge.__new__(LLMJudge)


def test_parse_response_allows_trailing_comma_in_object() -> None:
    """Parser should recover from trailing commas in JSON objects."""
    response = """{
  "verdict": "equivalent",
  "confidence": 0.93,
  "reasoning": "Same facts and conclusion",
  "missing_from_compressed": [],
  "missing_from_verbose": [],
}"""

    result = _judge()._parse_response(response)

    assert result.verdict == EquivalenceVerdict.EQUIVALENT
    assert result.confidence == 0.93


def test_parse_response_allows_trailing_comma_in_array() -> None:
    """Parser should recover from trailing commas in arrays."""
    response = """{
  "verdict": "partial",
  "confidence": 0.8,
  "reasoning": "Minor detail missing",
  "missing_from_compressed": ["detail A",],
  "missing_from_verbose": []
}"""

    result = _judge()._parse_response(response)

    assert result.verdict == EquivalenceVerdict.PARTIAL
    assert result.missing_from_compressed == ["detail A"]
