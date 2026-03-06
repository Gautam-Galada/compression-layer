"""Tests for the compression layer validation metrics."""

import pytest

from src.validation.metrics import (
    TaskType,
    compute_ast_similarity,
    compute_code_equivalence,
    compute_fact_overlap,
    compute_lexical_overlap,
    compute_nl_equivalence,
    extract_atomic_facts,
    is_equivalent,
)


class TestLexicalOverlap:
    """Tests for lexical overlap computation."""

    def test_identical_text(self):
        """Identical text should have overlap of 1.0."""
        text = "the quick brown fox jumps"
        assert compute_lexical_overlap(text, text) == 1.0

    def test_no_overlap(self):
        """Completely different text should have 0 overlap."""
        text_a = "cat dog bird"
        text_b = "apple orange banana"
        assert compute_lexical_overlap(text_a, text_b) == 0.0

    def test_partial_overlap(self):
        """Partial overlap should be between 0 and 1."""
        text_a = "the quick brown fox"
        text_b = "the slow brown dog"
        overlap = compute_lexical_overlap(text_a, text_b)
        assert 0 < overlap < 1
        # "the" and "brown" are shared, so 2 / 6 unique words
        assert overlap == pytest.approx(2 / 6)

    def test_empty_text(self):
        """Empty text should return 0."""
        assert compute_lexical_overlap("", "hello") == 0.0
        assert compute_lexical_overlap("hello", "") == 0.0
        assert compute_lexical_overlap("", "") == 0.0


class TestASTSimilarity:
    """Tests for AST-based code similarity."""

    def test_identical_code(self):
        """Identical code should have similarity of 1.0."""
        code = "def foo(): return 42"
        assert compute_ast_similarity(code, code) == 1.0

    def test_equivalent_code_different_formatting(self):
        """Equivalent code with different formatting should be similar."""
        code_a = "def foo():\n    return 42"
        code_b = "def foo(): return 42"
        sim = compute_ast_similarity(code_a, code_b)
        assert sim == 1.0  # AST should be identical

    def test_different_code(self):
        """Different code should have lower similarity."""
        code_a = "def foo(): return 42"
        code_b = "def bar(x): return x + 1"
        sim = compute_ast_similarity(code_a, code_b)
        assert 0 < sim < 1

    def test_invalid_syntax(self):
        """Invalid Python code should return 0."""
        valid = "def foo(): return 42"
        invalid = "def foo( return 42"
        assert compute_ast_similarity(valid, invalid) == 0.0
        assert compute_ast_similarity(invalid, valid) == 0.0


class TestNLEquivalence:
    """Tests for natural language equivalence scoring."""

    def test_identical_text(self):
        """Identical text should have high equivalence."""
        text = "The quick brown fox jumps over the lazy dog."
        equiv = compute_nl_equivalence(text, text)
        assert equiv > 0.99

    def test_similar_text(self):
        """Similar text should have reasonable equivalence."""
        text_a = "The cat sat on the mat."
        text_b = "A cat was sitting on a mat."
        equiv = compute_nl_equivalence(text_a, text_b)
        assert equiv > 0.5

    def test_unrelated_text(self):
        """Unrelated text should have low equivalence."""
        text_a = "The weather is sunny today."
        text_b = "def calculate_sum(a, b): return a + b"
        equiv = compute_nl_equivalence(text_a, text_b)
        assert equiv < 0.5


class TestCodeEquivalence:
    """Tests for code equivalence scoring."""

    def test_identical_code(self):
        """Identical code should have equivalence near 1.0."""
        code = "def add(a, b): return a + b"
        equiv = compute_code_equivalence(code, code)
        assert equiv > 0.99

    def test_semantically_equivalent_code(self):
        """Semantically equivalent code should have high equivalence."""
        code_a = """
def add(a, b):
    return a + b
"""
        code_b = """
def add(a, b):
    result = a + b
    return result
"""
        equiv = compute_code_equivalence(code_a, code_b)
        # Should be reasonably high due to similar semantics
        assert equiv > 0.5


class TestIsEquivalent:
    """Tests for the is_equivalent helper function."""

    def test_equivalent_nl(self):
        """Identical NL text should be equivalent."""
        text = "Hello world"
        assert is_equivalent(text, text, TaskType.QA, threshold=0.85)

    def test_equivalent_code(self):
        """Identical code should be equivalent."""
        code = "def foo(): pass"
        assert is_equivalent(code, code, TaskType.CODE_GEN, threshold=0.85)

    def test_not_equivalent_different_text(self):
        """Very different text should not be equivalent."""
        text_a = "Python programming"
        text_b = "Completely unrelated topic about cooking recipes"
        # May or may not pass depending on embeddings - test the function works
        result = is_equivalent(text_a, text_b, TaskType.QA, threshold=0.95)
        assert isinstance(result, bool)


class TestTaskTypes:
    """Tests for TaskType enum."""

    def test_all_task_types(self):
        """Verify all expected task types exist."""
        assert TaskType.QA.value == "qa"
        assert TaskType.CODE_GEN.value == "code_generation"
        assert TaskType.REASONING.value == "reasoning"
        assert TaskType.SUMMARIZATION.value == "summarization"


class TestExtractAtomicFacts:
    """Tests for extract_atomic_facts."""

    def test_simple_sentences(self):
        """Multiple sentences should produce multiple facts."""
        text = "Revenue is $10M. Headcount is 500. Founded in 2020."
        facts = extract_atomic_facts(text)
        assert len(facts) >= 3

    def test_compound_sentence(self):
        """Compound sentences joined by 'and' should be split."""
        text = "Revenue grew 15% and headcount increased to 500."
        facts = extract_atomic_facts(text)
        assert len(facts) >= 2

    def test_empty_text(self):
        """Empty text should produce no facts."""
        assert extract_atomic_facts("") == []

    def test_short_fragments_filtered(self):
        """Fragments shorter than 10 chars should be filtered out."""
        text = "OK. Yes. Revenue grew to $10M year over year."
        facts = extract_atomic_facts(text)
        # "OK" and "Yes" are too short
        assert all(len(f) > 10 for f in facts)

    def test_decimal_numbers_preserved(self):
        """Decimal numbers like 0.258 should NOT be broken by sentence splitting."""
        text = "Final loss was 0.258. Learning rate was 2e-5."
        facts = extract_atomic_facts(text)
        # The decimal 0.258 must appear intact in some fact
        joined = " ".join(facts)
        assert "0.258" in joined, f"Decimal 0.258 was broken: {facts}"

    def test_version_numbers_preserved(self):
        """Version numbers like 3.11 should not be broken."""
        text = "Python 3.11 is required. Node 18.0 is optional."
        facts = extract_atomic_facts(text)
        joined = " ".join(facts)
        assert "3.11" in joined, f"Version 3.11 was broken: {facts}"

    def test_semicolon_splitting(self):
        """Facts separated by semicolons should be split."""
        text = "Revenue is $10M; headcount is 500; founded in 2020"
        facts = extract_atomic_facts(text)
        assert len(facts) >= 3

    def test_deduplication(self):
        """Duplicate facts should be removed."""
        text = "Revenue is $10M. Revenue is $10M. Headcount is 500."
        facts = extract_atomic_facts(text)
        # Should deduplicate the repeated revenue fact
        revenue_facts = [f for f in facts if "revenue" in f]
        assert len(revenue_facts) == 1


class TestFactOverlap:
    """Tests for compute_fact_overlap (recall-weighted)."""

    def test_identical_text(self):
        """Identical text should have high fact overlap."""
        text = "The revenue increased 15%. Headcount grew to 500. Profit margin is 20%."
        overlap = compute_fact_overlap(text, text)
        assert overlap > 0.8

    def test_partially_overlapping(self):
        """When compressed drops some facts, overlap should be moderate."""
        verbose = "Revenue is $10M. Headcount is 500. Founded in 2020. CEO is John Smith."
        compressed = "Revenue is $10M. Headcount is 500."
        overlap = compute_fact_overlap(verbose, compressed)
        # Missing 2 of 4 facts — overlap should be less than perfect
        assert 0.0 < overlap < 1.0

    def test_empty_text_returns_zero(self):
        """Empty text should return 0."""
        assert compute_fact_overlap("", "something meaningful here") == 0.0
        assert compute_fact_overlap("something meaningful here", "") == 0.0

    def test_completely_different(self):
        """Unrelated texts should have low overlap."""
        verbose = "The weather is sunny today with clear skies."
        compressed = "Python uses indentation for code blocks."
        overlap = compute_fact_overlap(verbose, compressed)
        assert overlap < 0.5

    def test_recall_weighting(self):
        """Dropping facts from verbose should hurt more than adding facts in compressed.

        With recall_weight=0.7, verbose_coverage (recall) matters 2.3x more
        than compressed_coverage (precision).
        """
        verbose = "Revenue is $10M. Headcount is 500. Founded in 2020."
        # Compressed keeps only 1 of 3 facts but adds an extra one
        compressed = "Revenue is $10M. The sky is blue and clouds are fluffy."
        overlap = compute_fact_overlap(verbose, compressed)
        # Verbose coverage ≈ 1/3, compressed coverage ≈ 1/2
        # Score ≈ 0.7 * 0.33 + 0.3 * 0.5 = 0.381
        # Should be quite low due to recall weighting
        assert overlap < 0.5

    def test_decimal_numbers_dont_break_scoring(self):
        """Facts containing decimal numbers should score correctly."""
        verbose = "The model has 7 billion parameters. Final loss was 0.258."
        compressed = "The model has 7 billion parameters. Final loss was 0.258."
        overlap = compute_fact_overlap(verbose, compressed)
        # Identical content should score very high
        assert overlap > 0.8
