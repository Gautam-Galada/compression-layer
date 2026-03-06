"""Tests for the validation harness with mocked API calls."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.validation.harness import (
    BatchValidationStats,
    CompressionPair,
    ValidationHarness,
    ValidationResult,
)
from src.validation.metrics import TaskType
from src.validation.models import ModelType


@pytest.fixture
def mock_model_client():
    """Create a mock ModelClient that returns predictable responses."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="This is a test response.")
    return mock


@pytest.fixture
def sample_pair():
    """Create a sample compression pair for testing."""
    return CompressionPair(
        verbose="The user named John Smith is a senior software engineer at Google.",
        compressed="John Smith | sr SWE @ Google",
        domain="nl",
    )


@pytest.fixture
def sample_code_pair():
    """Create a sample code compression pair for testing."""
    return CompressionPair(
        verbose="""
def calculate_total(items):
    total = 0
    for item in items:
        total += item['price']
    return total
""",
        compressed="fn:calculate_total(items) = sum(i.price for i in items)",
        domain="code",
    )


def _make_harness_mocks(
    complete_return="Test response",
    semantic_sim=0.85,
    fact_overlap=0.70,
):
    """Helper: create standard mocks for harness tests.

    Returns (patches_context_manager_args, mock_client, mock_calc).
    """
    mock_client = MagicMock()
    mock_client.complete = AsyncMock(return_value=complete_return)

    mock_calc = MagicMock()
    mock_calc.compute_semantic_similarity = MagicMock(return_value=semantic_sim)

    return mock_client, mock_calc, fact_overlap


class TestCompressionPair:
    """Tests for CompressionPair model."""

    def test_create_pair(self, sample_pair):
        """Test creating a compression pair."""
        assert sample_pair.verbose.startswith("The user")
        assert sample_pair.compressed.startswith("John Smith")
        assert sample_pair.domain == "nl"

    def test_pair_with_metadata(self):
        """Test creating a pair with metadata."""
        pair = CompressionPair(
            verbose="test verbose",
            compressed="test compressed",
            domain="code",
            metadata={"source": "test", "language": "python"},
        )
        assert pair.metadata["language"] == "python"


class TestValidationResult:
    """Tests for ValidationResult dataclass."""

    def test_token_reduction_percent(self):
        """Test token reduction calculation."""
        result = ValidationResult(
            verbose_tokens=100,
            compressed_tokens=40,
            compression_ratio=0.4,
            equivalence_scores={ModelType.CLAUDE_SONNET: 0.9},
            min_equivalence=0.9,
            passed=True,
        )
        assert result.token_reduction_percent == 60.0

    def test_passed_result(self):
        """Test that passed is correctly set."""
        result = ValidationResult(
            verbose_tokens=100,
            compressed_tokens=50,
            compression_ratio=0.5,
            equivalence_scores={
                ModelType.CLAUDE_SONNET: 0.92,
                ModelType.GPT4O_MINI: 0.88,
            },
            min_equivalence=0.88,
            passed=True,
        )
        assert result.passed
        assert result.min_equivalence == 0.88

    def test_result_includes_gate_scores(self):
        """Test that ValidationResult accepts per-gate score fields."""
        result = ValidationResult(
            verbose_tokens=100,
            compressed_tokens=40,
            compression_ratio=0.4,
            equivalence_scores={ModelType.CLAUDE_SONNET: 0.9},
            min_equivalence=0.7,
            passed=True,
            embedding_scores={ModelType.CLAUDE_SONNET: 0.85},
            fact_overlap_scores={ModelType.CLAUDE_SONNET: 0.70},
        )
        assert result.embedding_scores[ModelType.CLAUDE_SONNET] == 0.85
        assert result.fact_overlap_scores[ModelType.CLAUDE_SONNET] == 0.70


class TestValidationHarness:
    """Tests for ValidationHarness class."""

    @pytest.mark.asyncio
    async def test_harness_initialization(self):
        """Test harness initializes with default models and gate thresholds."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client_class.return_value = MagicMock()
            mock_calc_class.return_value = MagicMock()
            harness = ValidationHarness()

            assert len(harness.models) == 3
            assert ModelType.CLAUDE_SONNET in harness.models
            # 3-gate thresholds (no legacy threshold)
            assert harness.embedding_threshold == 0.60
            assert harness.fact_overlap_threshold == 0.55
            assert harness.judge_threshold == 0.75

    @pytest.mark.asyncio
    async def test_harness_custom_models(self):
        """Test harness with custom model list."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client_class.return_value = MagicMock()
            mock_calc_class.return_value = MagicMock()
            harness = ValidationHarness(
                models=[ModelType.GPT4O_MINI],
            )

            assert len(harness.models) == 1

    @pytest.mark.asyncio
    async def test_validate_pair_mocked(self, sample_pair):
        """Test validation with mocked API calls produces correct result shape."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.70),
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Test response about John Smith")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.85)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            result = await harness.validate_pair(sample_pair)

            assert isinstance(result, ValidationResult)
            assert result.verbose_tokens > 0
            assert result.compressed_tokens > 0
            assert result.compression_ratio < 1.0
            assert ModelType.CLAUDE_SONNET in result.equivalence_scores
            # New gate fields
            assert result.embedding_scores is not None
            assert result.fact_overlap_scores is not None
            assert ModelType.CLAUDE_SONNET in result.embedding_scores
            assert ModelType.CLAUDE_SONNET in result.fact_overlap_scores

    @pytest.mark.asyncio
    async def test_quick_validate_mocked(self):
        """Test quick_validate convenience method."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.70),
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Identical response")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.95)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.GPT4O_MINI],
                tasks=[TaskType.QA],
            )

            passed = await harness.quick_validate(
                verbose="Test verbose text",
                compressed="Test compressed",
                domain="nl",
            )

            assert isinstance(passed, bool)

    @pytest.mark.asyncio
    async def test_validate_batch_mocked(self, sample_pair):
        """Test batch validation."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.70),
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Batch response")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.95)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            pairs = [sample_pair, sample_pair]
            stats = await harness.validate_batch(pairs, concurrency=2)

            assert isinstance(stats, BatchValidationStats)
            assert stats.total_pairs == 2
            assert len(stats.results) == 2


class TestMaxTokensCap:
    """Tests that frontier model calls use capped max_tokens."""

    @pytest.mark.asyncio
    async def test_complete_called_with_max_tokens_200(self, sample_pair):
        """Verify model completions are called with max_tokens=200."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.70),
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Short response")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.85)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            await harness.validate_pair(sample_pair)

            # Every complete() call should use max_tokens=200
            for call in mock_client.complete.call_args_list:
                _, kwargs = call
                if "max_tokens" in kwargs:
                    assert kwargs["max_tokens"] == 200, (
                        f"Expected max_tokens=200, got {kwargs['max_tokens']}"
                    )

    def test_validation_max_tokens_constant(self):
        """Verify the class constant is 200."""
        assert ValidationHarness.VALIDATION_MAX_TOKENS == 200


class TestFactOverlapIntegration:
    """Tests that fact_overlap is computed during validation."""

    @pytest.mark.asyncio
    async def test_fact_overlap_scores_populated(self, sample_pair):
        """Verify fact_overlap_scores are included in ValidationResult."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.70) as mock_fo,
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Test response about facts")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.85)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            result = await harness.validate_pair(sample_pair)

            assert result.fact_overlap_scores is not None
            assert ModelType.CLAUDE_SONNET in result.fact_overlap_scores
            assert result.fact_overlap_scores[ModelType.CLAUDE_SONNET] == 0.70
            # Verify compute_fact_overlap was called
            assert mock_fo.called

    @pytest.mark.asyncio
    async def test_embedding_scores_populated(self, sample_pair):
        """Verify embedding_scores are included in ValidationResult."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.70),
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Test response")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.82)
            mock_calc_class.return_value = mock_calc

            harness = ValidationHarness(
                models=[ModelType.CLAUDE_SONNET],
                tasks=[TaskType.QA],
            )

            result = await harness.validate_pair(sample_pair)

            assert result.embedding_scores is not None
            assert ModelType.CLAUDE_SONNET in result.embedding_scores
            assert result.embedding_scores[ModelType.CLAUDE_SONNET] == 0.82


class TestThreeGateLogic:
    """Tests for 3-gate pass/fail system."""

    async def _run_with_scores(
        self,
        embedding_sim: float,
        fact_overlap_val: float,
        use_judge: bool = False,
        judge_score: float | None = None,
        **harness_kwargs,
    ) -> ValidationResult:
        """Helper to run validation with controlled gate scores."""
        pair = CompressionPair(
            verbose="The revenue grew 15% year over year to $10M.",
            compressed="Rev +15% YoY → $10M",
            domain="nl",
        )
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=fact_overlap_val),
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Revenue increased 15%")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=embedding_sim)
            mock_calc_class.return_value = mock_calc

            if use_judge and judge_score is not None:
                # Mock the LLM judge
                with patch("src.validation.harness.LLMJudge") as mock_judge_class:
                    mock_judge = MagicMock()
                    mock_judge.evaluate = AsyncMock(return_value=MagicMock())
                    mock_judge.verdict_to_score = MagicMock(return_value=judge_score)
                    mock_judge_class.return_value = mock_judge

                    # Also mock the compute method for the combined score path
                    mock_scores = MagicMock()
                    mock_scores.combined_score = judge_score
                    mock_calc.compute = MagicMock(return_value=mock_scores)

                    harness = ValidationHarness(
                        models=[ModelType.CLAUDE_SONNET],
                        tasks=[TaskType.QA],
                        use_llm_judge=True,
                        **harness_kwargs,
                    )
                    return await harness.validate_pair(pair)
            else:
                harness = ValidationHarness(
                    models=[ModelType.CLAUDE_SONNET],
                    tasks=[TaskType.QA],
                    use_llm_judge=False,
                    **harness_kwargs,
                )
                return await harness.validate_pair(pair)

    @pytest.mark.asyncio
    async def test_all_gates_pass_no_judge(self):
        """Pair passes when embedding and fact_overlap both exceed thresholds (no judge)."""
        result = await self._run_with_scores(embedding_sim=0.80, fact_overlap_val=0.70)
        assert result.passed is True
        # min_equivalence = min(0.80, 0.70) = 0.70
        assert result.min_equivalence == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_embedding_gate_fails(self):
        """Pair fails when embedding similarity is below 0.60."""
        result = await self._run_with_scores(embedding_sim=0.50, fact_overlap_val=0.70)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_fact_overlap_gate_fails(self):
        """Pair fails when fact overlap is below 0.55."""
        result = await self._run_with_scores(embedding_sim=0.80, fact_overlap_val=0.40)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_both_gates_fail(self):
        """Pair fails when both gates fail."""
        result = await self._run_with_scores(embedding_sim=0.50, fact_overlap_val=0.40)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_all_three_gates_pass(self):
        """Pair passes when all three gates pass (with judge)."""
        result = await self._run_with_scores(
            embedding_sim=0.80, fact_overlap_val=0.70, use_judge=True, judge_score=0.85
        )
        assert result.passed is True
        # min(0.80, 0.70, 0.85) = 0.70
        assert result.min_equivalence == pytest.approx(0.70)

    @pytest.mark.asyncio
    async def test_judge_gate_fails(self):
        """Pair fails when judge score is below 0.75, even if others pass."""
        result = await self._run_with_scores(
            embedding_sim=0.80, fact_overlap_val=0.70, use_judge=True, judge_score=0.60
        )
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_thresholds(self):
        """Custom gate thresholds are respected."""
        # With default thresholds, embedding=0.55 would fail (< 0.60)
        # But with custom embedding_threshold=0.50, it passes
        result = await self._run_with_scores(
            embedding_sim=0.55,
            fact_overlap_val=0.70,
            embedding_threshold=0.50,
        )
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_min_equivalence_is_min_of_gates_no_judge(self):
        """min_equivalence equals min of active gate scores (no judge)."""
        result = await self._run_with_scores(embedding_sim=0.90, fact_overlap_val=0.60)
        assert result.min_equivalence == pytest.approx(0.60)

    @pytest.mark.asyncio
    async def test_min_equivalence_is_min_of_gates_with_judge(self):
        """min_equivalence equals min of all three gate scores (with judge)."""
        result = await self._run_with_scores(
            embedding_sim=0.90, fact_overlap_val=0.80, use_judge=True, judge_score=0.76
        )
        assert result.min_equivalence == pytest.approx(0.76)


class TestBatchValidationStats:
    """Tests for BatchValidationStats."""

    def test_stats_calculation(self):
        """Test batch stats are correctly computed."""
        results = [
            ValidationResult(
                verbose_tokens=100,
                compressed_tokens=50,
                compression_ratio=0.5,
                equivalence_scores={ModelType.CLAUDE_SONNET: 0.9},
                min_equivalence=0.9,
                passed=True,
            ),
            ValidationResult(
                verbose_tokens=200,
                compressed_tokens=80,
                compression_ratio=0.4,
                equivalence_scores={ModelType.CLAUDE_SONNET: 0.8},
                min_equivalence=0.8,
                passed=False,  # Below 0.85 threshold
            ),
        ]

        stats = BatchValidationStats(
            total_pairs=2,
            passed_pairs=1,
            failed_pairs=1,
            avg_compression_ratio=0.45,
            avg_equivalence=0.85,
            min_equivalence=0.8,
            pass_rate=0.5,
            results=results,
        )

        assert stats.pass_rate == 0.5
        assert stats.avg_compression_ratio == 0.45


class TestIntegration:
    """Integration tests (can be skipped without API keys)."""

    @pytest.mark.asyncio
    @pytest.mark.skip(reason="Requires API keys")
    async def test_real_validation(self, sample_pair):
        """Test real validation with actual API calls."""
        harness = ValidationHarness(
            models=[ModelType.GPT4O_MINI],  # Cheapest option
        )

        result = await harness.validate_pair(sample_pair)

        assert result.compression_ratio < 1.0
        print(f"Compression ratio: {result.compression_ratio:.2%}")
        print(f"Equivalence scores: {result.equivalence_scores}")
        print(f"Passed: {result.passed}")
