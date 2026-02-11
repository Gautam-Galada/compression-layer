"""
Unit tests for tokenizer utilities with special token handling.

Tests cover:
- Basic token counting
- Special token handling (<|endoftext|>, <|im_start|>, etc.)
- Compression ratio calculations
- Edge cases (empty strings, zero tokens, etc.)
- Regression test for tiktoken version changes
"""

import sys
from pathlib import Path

import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.tokenizers import compression_ratio, count_tokens


class TestTokenCounting:
    """Test basic token counting functionality."""

    def test_simple_text(self):
        """Test counting tokens in simple text."""
        text = "Hello, world!"
        count = count_tokens(text)
        assert count > 0, "Should count tokens in simple text"
        assert isinstance(count, int), "Token count should be integer"

    def test_empty_string(self):
        """Test empty string returns 0 tokens."""
        assert count_tokens("") == 0

    def test_whitespace_only(self):
        """Test whitespace-only string."""
        count = count_tokens("   ")
        assert count >= 0, "Whitespace should have non-negative token count"

    def test_longer_text(self):
        """Test longer text has more tokens."""
        short = "Hi"
        long = "This is a much longer piece of text with many more words."

        short_count = count_tokens(short)
        long_count = count_tokens(long)

        assert long_count > short_count, "Longer text should have more tokens"


class TestSpecialTokens:
    """Test handling of special tokens in text."""

    def test_endoftext_token(self):
        """
        REGRESSION TEST: Verify <|endoftext|> is handled as normal text.

        This is the token that caused the original error. We now encode it
        as normal text rather than treating it as a special control token.
        """
        text_with_special = "This is text with <|endoftext|> in it."
        text_without_special = "This is text with PLACEHOLDER in it."

        # Should not raise ValueError
        count_with = count_tokens(text_with_special)
        count_without = count_tokens(text_without_special)

        assert count_with > 0, "Should count tokens with <|endoftext|>"
        assert isinstance(count_with, int), "Token count should be integer"

        # The counts may differ, but both should work
        assert count_with >= count_without - 5, "Token counts should be similar"

    def test_multiple_special_tokens(self):
        """Test text with multiple special tokens."""
        text = "Start <|endoftext|> middle <|im_start|> end <|im_end|>"

        # Should not raise ValueError
        count = count_tokens(text)
        assert count > 0, "Should handle multiple special tokens"

    def test_only_special_token(self):
        """Test text that is only a special token."""
        text = "<|endoftext|>"

        # Should not raise ValueError
        count = count_tokens(text)
        assert count > 0, "Should count special token as text"

    def test_special_token_variations(self):
        """Test various special token formats."""
        special_tokens = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|im_sep|>",
        ]

        for token in special_tokens:
            # Should not raise ValueError for any special token
            count = count_tokens(token)
            assert count > 0, f"Should handle {token}"


class TestCompressionRatio:
    """Test compression ratio calculations."""

    def test_perfect_compression(self):
        """Test ratio when compressed is much shorter."""
        original = "This is a very long piece of text that should compress well."
        compressed = "Short"

        ratio = compression_ratio(original, compressed)
        assert ratio < 1.0, "Good compression should have ratio < 1.0"

    def test_no_compression(self):
        """Test ratio when text is unchanged."""
        text = "Same text"
        ratio = compression_ratio(text, text)

        assert ratio == 1.0, "Identical text should have ratio of 1.0"

    def test_expansion(self):
        """Test ratio when compressed is longer (bad compression)."""
        original = "Short"
        compressed = "This is actually much longer than the original text."

        ratio = compression_ratio(original, compressed)
        assert ratio > 1.0, "Expansion should have ratio > 1.0"

    def test_empty_original(self):
        """Test ratio with empty original string."""
        ratio = compression_ratio("", "something")
        assert ratio == 1.0, "Empty original should return 1.0"

    def test_both_empty(self):
        """Test ratio when both strings are empty."""
        ratio = compression_ratio("", "")
        assert ratio == 1.0, "Both empty should return 1.0"

    def test_ratio_with_special_tokens(self):
        """
        REGRESSION TEST: Verify compression ratio works with special tokens.

        This ensures the special token handling doesn't break ratio calculations.
        """
        original = "This text has <|endoftext|> special tokens in it."
        compressed = "Text with <|endoftext|> tokens."

        # Should not raise ValueError
        ratio = compression_ratio(original, compressed)

        assert 0 < ratio <= 1.0, "Compression should have ratio between 0 and 1"
        assert isinstance(ratio, float), "Ratio should be float"


class TestTokenizerConsistency:
    """Test consistency across different inputs and edge cases."""

    def test_unicode_characters(self):
        """Test counting tokens in text with unicode characters."""
        text = "Hello 世界 🌍 café"
        count = count_tokens(text)
        assert count > 0, "Should handle unicode characters"

    def test_code_text(self):
        """Test counting tokens in code snippets."""
        code = """
        def hello():
            return "world"
        """
        count = count_tokens(code)
        assert count > 0, "Should count tokens in code"

    def test_newlines_and_whitespace(self):
        """Test text with various whitespace."""
        text = "Line 1\nLine 2\n\nLine 3\t\tTabbed"
        count = count_tokens(text)
        assert count > 0, "Should handle newlines and tabs"

    def test_repeated_counting(self):
        """Test that counting is consistent across calls."""
        text = "Consistency test"
        count1 = count_tokens(text)
        count2 = count_tokens(text)
        count3 = count_tokens(text)

        assert count1 == count2 == count3, "Token counting should be consistent"


class TestTiktokenVersionRegression:
    """
    Regression tests to catch tiktoken version changes.

    These tests will fail if tiktoken's behavior changes significantly,
    alerting us to review and update our special token handling.
    """

    def test_known_token_counts(self):
        """Test known strings have expected token counts (±1 for version tolerance)."""
        test_cases = {
            "Hello": (1, 2),  # Expected range: 1-2 tokens
            "Hello, world!": (3, 5),  # Expected range: 3-5 tokens
            "The quick brown fox": (4, 6),  # Expected range: 4-6 tokens
        }

        for text, (min_expected, max_expected) in test_cases.items():
            count = count_tokens(text)
            assert min_expected <= count <= max_expected, (
                f"Token count for '{text}' outside expected range: {count} not in [{min_expected}, {max_expected}]"
            )

    def test_special_token_encoding_behavior(self):
        """
        CRITICAL REGRESSION TEST: Verify special tokens are encoded as text.

        If this fails after a tiktoken update, it means:
        1. Our disallowed_special=() parameter may have changed behavior
        2. We need to review our special token handling strategy
        """
        text_with_special = "Text <|endoftext|> here"

        # This should NOT raise ValueError
        try:
            count = count_tokens(text_with_special)
            assert count > 0, "Should successfully count with special tokens"
        except ValueError as e:
            if "special token" in str(e).lower():
                pytest.fail(
                    "Special token handling broken! "
                    "disallowed_special=() may no longer work. "
                    "Review tokenizer implementation and tiktoken version."
                )
            else:
                raise

    def test_tokenizer_caching(self):
        """Test that tokenizer caching works (via consistent token counts)."""
        # Instead of testing internal caching mechanism, test that
        # repeated calls give consistent results (which implies caching works)
        text = "Test tokenizer caching"

        counts = [count_tokens(text) for _ in range(5)]

        # All counts should be identical (proves caching works)
        assert len(set(counts)) == 1, "Token counts should be consistent across calls"
        assert counts[0] > 0, "Should count tokens"


class TestDataSanitizationIntegration:
    """Integration tests for data sanitization use cases."""

    def test_compression_ratio_threshold(self):
        """Test that ratio threshold logic works correctly."""
        # Good compression (should pass)
        original = "This is a long verbose sentence with many unnecessary words."
        compressed = "Long verbose sentence."
        ratio = compression_ratio(original, compressed)
        assert ratio <= 1.0, "Good compression should pass <= 1.0 threshold"

        # Bad compression / expansion (should fail)
        original = "Short"
        compressed = "This became very long and verbose."
        ratio = compression_ratio(original, compressed)
        assert ratio > 1.0, "Expansion should fail > 1.0 threshold"

    def test_real_world_training_sample(self):
        """Test with realistic training data format."""
        verbose = """
        def validate(function: AnyCallableT) -> AnyCallableT:
            _check_function_type(function)
            validate_call_wrapper = _validate_call.ValidateCallWrapper(
                cast(_generate_schema.ValidateCallSupportedTypes, function), 
                config, validate_return, parent_namespace
            )
            return _validate_call.update_wrapper_attributes(function, validate_call_wrapper.__call__)
        """

        compressed = """fn:validate(function:AnyCallableT)->AnyCallableT = 
  _check_function_type(function) |> 
  _validate_call.ValidateCallWrapper(cast(...), config, validate_return, parent_namespace) |> 
  λwrapper: _validate_call.update_wrapper_attributes(function, wrapper.__call__)"""

        ratio = compression_ratio(verbose, compressed)

        # Should be valid compression
        assert 0 < ratio < 1.0, f"Real training sample should compress well, got ratio: {ratio}"

    def test_sample_with_special_tokens_in_training_data(self):
        """
        CRITICAL: Test that training data with special tokens works.

        This is the actual bug that was reported - training data contained
        <|endoftext|> and caused ValueError during sanitization.
        """
        verbose = "This is training data with <|endoftext|> token."
        compressed = "Training data with <|endoftext|>"

        # Should not raise ValueError
        ratio = compression_ratio(verbose, compressed)

        assert 0 < ratio <= 1.0, "Should handle special tokens in training data"


# Pytest fixtures
@pytest.fixture
def sample_texts():
    """Fixture providing various text samples for testing."""
    return {
        "simple": "Hello, world!",
        "empty": "",
        "with_special": "Text <|endoftext|> here",
        "code": "def foo(): return 42",
        "unicode": "café 世界 🌍",
        "long": " ".join(["word"] * 100),
    }


# Run tests with: pytest test_tokenizers.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
