"""Cross-model validation harness for compression pairs."""

import asyncio
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel

from ..utils.caching import SemanticCache
from ..utils.tokenizers import count_tokens
from .llm_judge import LLMJudge
from .metrics import EquivalenceCalculator, TaskType, compute_equivalence, compute_fact_overlap
from .models import ModelClient, ModelType

logger = logging.getLogger(__name__)


class CompressionPair(BaseModel):
    """A pair of verbose and compressed text for validation."""

    verbose: str
    compressed: str
    domain: str  # "nl" | "code" | "mixed"
    metadata: dict[str, str] | None = None


@dataclass
class ValidationResult:
    """Result of validating a single compression pair."""

    verbose_tokens: int
    compressed_tokens: int
    compression_ratio: float
    equivalence_scores: dict[ModelType, float]
    min_equivalence: float
    passed: bool
    llm_judge_used: bool = False
    llm_judge_scores: dict[ModelType, float] | None = None
    # Per-gate scores for the 3-gate system
    embedding_scores: dict[ModelType, float] | None = None
    fact_overlap_scores: dict[ModelType, float] | None = None

    @property
    def token_reduction_percent(self) -> float:
        """Percentage of tokens reduced."""
        return (1 - self.compression_ratio) * 100


@dataclass
class BatchValidationStats:
    """Statistics from batch validation."""

    total_pairs: int
    passed_pairs: int
    failed_pairs: int
    avg_compression_ratio: float
    avg_equivalence: float
    min_equivalence: float
    pass_rate: float
    results: list[ValidationResult] = field(default_factory=list)


# Default task prompts for validation.
#
# These prompts are designed to test FACTUAL PRECISION — whether the context
# preserves specific facts, numbers, parameter names, and code details.
# A good compression must let the model extract the same facts as the verbose
# original.  Generic "summarize" or "key points" prompts would test gist
# retention, which is too lenient for production middleware.
DEFAULT_TASK_PROMPTS: dict[TaskType, str] = {
    TaskType.QA: """Read the context below and list every specific fact it contains.
Include all numbers, percentages, names, dates, quantities, and technical terms.
Be exhaustive — do not summarize or paraphrase. One fact per line.

Context:
{context}

Facts:""",
    TaskType.REASONING: """Read the information below and answer these questions using ONLY the information provided.
1. What specific quantities, metrics, or measurements are mentioned?
2. What entities (people, organizations, systems, models) are named?
3. What causal or conditional relationships are stated?
List each answer with the exact values from the text. Do not infer beyond what is stated.

Information:
{context}

Answers:""",
    TaskType.CODE_GEN: """Read the code or technical specification below and list:
1. Every function/class/variable name defined
2. Every parameter and its type or default value
3. Every return type or return value
4. Every numeric constant, threshold, or configuration value
Be precise — use the exact names and values from the source.

Source:
{context}

Details:""",
}


class ValidationHarness:
    """
    Cross-model validation harness for compression pairs.

    Validates that compressed text produces equivalent model outputs
    across multiple LLMs (Claude, GPT, Gemini).

    The validation pipeline uses a 3-gate scoring system:
    1. Embedding similarity (MiniLM) - catches gross topic drift / garbled outputs
    2. Fact overlap - catches dropped facts, numbers, parameter names
    3. LLM judge (optional) - catches subtle reasoning gaps, quality issues

    A pair passes only if ALL active gates exceed their thresholds.
    The final equivalence_score is min(all gate scores).
    """

    # Cap frontier model output tokens to fit within MiniLM-L6-v2's 256 token window.
    # This prevents silent truncation that corrupts embedding similarity scores.
    VALIDATION_MAX_TOKENS = 200

    def __init__(
        self,
        models: list[ModelType] | None = None,
        tasks: list[TaskType] | None = None,
        cache: SemanticCache | None = None,
        use_llm_judge: bool = False,  # Optional LLM judge for more accuracy
        llm_judge_model: ModelType = ModelType.CLAUDE_SONNET,
        # 3-gate thresholds
        embedding_threshold: float = 0.60,
        fact_overlap_threshold: float = 0.55,
        judge_threshold: float = 0.75,
    ):
        """
        Initialize the validation harness.

        Args:
            models: List of models to validate against (defaults to all)
            tasks: Task types to test (defaults to QA + REASONING)
            cache: Optional cache for API responses
            use_llm_judge: Whether to use LLM-as-judge for equivalence (costs more but more accurate)
            llm_judge_model: Model to use for LLM judge (default: Claude Sonnet)
            embedding_threshold: Gate 1 - min embedding similarity to pass (default: 0.60)
            fact_overlap_threshold: Gate 2 - min fact overlap to pass (default: 0.55)
            judge_threshold: Gate 3 - min LLM judge score to pass (default: 0.75)
        """
        if models is None:
            models = [ModelType.CLAUDE_SONNET, ModelType.GPT4O_MINI, ModelType.GEMINI_FLASH]

        self.models = models
        # Auto-select tasks based on domain if not specified
        if tasks is None:
            # Will be set in validate_pair based on pair.domain
            self.tasks = None
        else:
            self.tasks = tasks
        self.cache = cache

        # 3-gate thresholds
        self.embedding_threshold = embedding_threshold
        self.fact_overlap_threshold = fact_overlap_threshold
        self.judge_threshold = judge_threshold

        # Initialize clients
        self.clients = {m: ModelClient(m, cache=cache) for m in models}

        # LLM Judge (optional)
        self.use_llm_judge = use_llm_judge
        self.llm_judge = LLMJudge(judge_model=llm_judge_model) if use_llm_judge else None

        # Equivalence calculator with pure semantic similarity
        self.metrics = EquivalenceCalculator(
            semantic_weight=1.0,  # Pure semantic
            lexical_weight=0.0,  # No lexical (hurts valid compressions)
        )

    async def validate_pair(
        self,
        pair: CompressionPair,
        task_prompts: dict[TaskType, str] | None = None,
    ) -> ValidationResult:
        """
        Validate a single compression pair across all models and tasks.

        Uses a 3-gate scoring system:
        - Gate 1 (embedding): MiniLM cosine similarity >= embedding_threshold
        - Gate 2 (fact_overlap): Atomic fact coverage >= fact_overlap_threshold
        - Gate 3 (judge, optional): LLM judge score >= judge_threshold

        A pair passes only if ALL active gates pass.

        Args:
            pair: The compression pair to validate
            task_prompts: Optional custom prompts (use {context} placeholder)

        Returns:
            ValidationResult with per-gate scores and pass/fail status
        """
        prompts = task_prompts or DEFAULT_TASK_PROMPTS
        # Per-model combined scores (legacy field, now min of gates)
        scores: dict[ModelType, float] = {}
        llm_judge_scores: dict[ModelType, float] | None = {} if self.use_llm_judge else None
        embedding_scores: dict[ModelType, float] = {}
        fact_overlap_scores: dict[ModelType, float] = {}

        # Auto-select tasks based on domain if not set
        if self.tasks is None:
            if pair.domain == "code":
                tasks = [TaskType.CODE_GEN, TaskType.REASONING]
            else:
                tasks = [TaskType.QA, TaskType.REASONING]
        else:
            tasks = self.tasks

        async def eval_model(
            model_type: ModelType,
        ) -> tuple[ModelType, float, float, float, float | None]:
            """Evaluate equivalence for a single model across all tasks.

            Returns (model_type, avg_combined, avg_embedding, avg_fact_overlap, avg_judge).
            """
            client = self.clients[model_type]
            task_scores: list[float] = []
            judge_task_scores: list[float] = []
            embedding_sims: list[float] = []
            fact_scores: list[float] = []

            for task_type in tasks:
                prompt_template = prompts.get(task_type, prompts[TaskType.QA])

                # Get outputs for verbose and compressed inputs
                verbose_prompt = prompt_template.format(context=pair.verbose)
                compressed_prompt = prompt_template.format(context=pair.compressed)

                verbose_out = await client.complete(
                    verbose_prompt, max_tokens=self.VALIDATION_MAX_TOKENS
                )
                compressed_out = await client.complete(
                    compressed_prompt, max_tokens=self.VALIDATION_MAX_TOKENS
                )

                # Gate 1: Embedding similarity
                embedding_sim = self.metrics.compute_semantic_similarity(
                    verbose_out, compressed_out
                )
                embedding_sims.append(embedding_sim)

                # Gate 2: Fact overlap
                fact_score = compute_fact_overlap(
                    verbose_out, compressed_out, calculator=self.metrics
                )
                fact_scores.append(fact_score)

                # Gate 3: LLM judge (optional)
                llm_score = None
                if self.use_llm_judge and self.llm_judge:
                    task_desc = f"{task_type.value} task: extracting and comparing information"
                    judge_result = await self.llm_judge.evaluate(
                        task_description=task_desc,
                        verbose_output=verbose_out,
                        compressed_output=compressed_out,
                    )
                    llm_score = self.llm_judge.verdict_to_score(judge_result)
                    judge_task_scores.append(llm_score)

                # Combined score for legacy equivalence_scores field
                if llm_score is not None:
                    combined = self.metrics.compute(
                        verbose_out,
                        compressed_out,
                        llm_judge_score=llm_score,
                    )
                    task_scores.append(combined.combined_score)
                else:
                    score = await compute_equivalence(verbose_out, compressed_out, task_type)
                    task_scores.append(score)

            # Average across tasks
            avg_combined = sum(task_scores) / len(task_scores)
            avg_embedding = sum(embedding_sims) / len(embedding_sims)
            avg_fact = sum(fact_scores) / len(fact_scores)
            avg_judge = (
                sum(judge_task_scores) / len(judge_task_scores) if judge_task_scores else None
            )

            return model_type, avg_combined, avg_embedding, avg_fact, avg_judge

        # Run all models in parallel
        results = await asyncio.gather(*[eval_model(m) for m in self.models])

        for model_type, combined, embedding, fact, judge_score in results:
            scores[model_type] = combined
            embedding_scores[model_type] = embedding
            fact_overlap_scores[model_type] = fact
            if llm_judge_scores is not None and judge_score is not None:
                llm_judge_scores[model_type] = judge_score

        # 3-gate pass/fail logic
        min_embedding = min(embedding_scores.values())
        min_fact_overlap = min(fact_overlap_scores.values())

        gate1_pass = min_embedding >= self.embedding_threshold
        gate2_pass = min_fact_overlap >= self.fact_overlap_threshold

        if self.use_llm_judge and llm_judge_scores:
            min_judge = min(llm_judge_scores.values())
            gate3_pass = min_judge >= self.judge_threshold
            all_gates_pass = gate1_pass and gate2_pass and gate3_pass
            min_equiv = min(min_embedding, min_fact_overlap, min_judge)
        else:
            all_gates_pass = gate1_pass and gate2_pass
            min_equiv = min(min_embedding, min_fact_overlap)

        # Calculate token metrics
        verbose_tokens = count_tokens(pair.verbose)
        compressed_tokens = count_tokens(pair.compressed)

        return ValidationResult(
            verbose_tokens=verbose_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compressed_tokens / verbose_tokens if verbose_tokens > 0 else 1.0,
            equivalence_scores=scores,
            min_equivalence=min_equiv,
            passed=all_gates_pass,
            llm_judge_used=self.use_llm_judge,
            llm_judge_scores=llm_judge_scores if llm_judge_scores else None,
            embedding_scores=embedding_scores,
            fact_overlap_scores=fact_overlap_scores,
        )

    async def validate_batch(
        self,
        pairs: list[CompressionPair],
        task_prompts: dict[TaskType, str] | None = None,
        concurrency: int = 10,
    ) -> BatchValidationStats:
        """
        Validate a batch of compression pairs with concurrency control.

        Args:
            pairs: List of compression pairs to validate
            task_prompts: Optional custom prompts
            concurrency: Maximum concurrent validations

        Returns:
            BatchValidationStats with aggregated results
        """
        sem = asyncio.Semaphore(concurrency)

        async def bounded(pair: CompressionPair) -> ValidationResult:
            async with sem:
                return await self.validate_pair(pair, task_prompts)

        results = await asyncio.gather(*[bounded(p) for p in pairs])

        # Aggregate statistics
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        avg_ratio = sum(r.compression_ratio for r in results) / len(results)
        avg_equiv = sum(r.min_equivalence for r in results) / len(results)
        min_equiv = min(r.min_equivalence for r in results)

        return BatchValidationStats(
            total_pairs=len(pairs),
            passed_pairs=passed,
            failed_pairs=failed,
            avg_compression_ratio=avg_ratio,
            avg_equivalence=avg_equiv,
            min_equivalence=min_equiv,
            pass_rate=passed / len(pairs),
            results=results,
        )

    async def quick_validate(
        self,
        verbose: str,
        compressed: str,
        domain: str = "nl",
    ) -> bool:
        """
        Quick validation check for a single pair.

        Args:
            verbose: Original verbose text
            compressed: Compressed version
            domain: Content domain ("nl", "code", "mixed")

        Returns:
            True if the pair passes validation
        """
        pair = CompressionPair(verbose=verbose, compressed=compressed, domain=domain)
        result = await self.validate_pair(pair)
        return result.passed


async def validate_compression(
    verbose: str,
    compressed: str,
    domain: str = "nl",
    use_llm_judge: bool = False,
    embedding_threshold: float = 0.60,
    fact_overlap_threshold: float = 0.55,
    judge_threshold: float = 0.75,
) -> ValidationResult:
    """
    Convenience function to validate a single compression.

    Args:
        verbose: Original text
        compressed: Compressed text
        domain: Content domain
        use_llm_judge: Whether to use LLM judge for more accurate equivalence
        embedding_threshold: Gate 1 threshold (default: 0.60)
        fact_overlap_threshold: Gate 2 threshold (default: 0.55)
        judge_threshold: Gate 3 threshold (default: 0.75)

    Returns:
        ValidationResult
    """
    harness = ValidationHarness(
        use_llm_judge=use_llm_judge,
        embedding_threshold=embedding_threshold,
        fact_overlap_threshold=fact_overlap_threshold,
        judge_threshold=judge_threshold,
    )
    pair = CompressionPair(verbose=verbose, compressed=compressed, domain=domain)
    return await harness.validate_pair(pair)
