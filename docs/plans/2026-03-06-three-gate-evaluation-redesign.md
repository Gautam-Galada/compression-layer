# 3-Gate Evaluation Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-score equivalence pipeline with a 3-gate system (embedding, fact overlap, LLM judge) where all gates must pass for a pair to be marked equivalent.

**Architecture:** Each compression pair is evaluated through three independent scoring layers. Embedding similarity (MiniLM) catches gross topic drift, fact overlap catches dropped facts/numbers, and LLM judge catches subtle reasoning gaps. A pair passes only if ALL three gates exceed their thresholds. The final `equivalence_score` is `min(all three)`.

**Tech Stack:** Python 3.11+, sentence-transformers (all-MiniLM-L6-v2), pytest, asyncio

---

## Gate Thresholds

| Gate | Metric | Threshold | Role |
|------|--------|-----------|------|
| 1 | Embedding similarity (`all-MiniLM-L6-v2`) | >= 0.60 | Safety net (catches garbled/off-topic) |
| 2 | Fact overlap (`compute_fact_overlap`) | >= 0.55 | Factual precision gate |
| 3 | LLM judge score | >= 0.75 | Primary quality signal |

**Pass condition:** ALL three gates must pass. `equivalence_score = min(gate1, gate2, gate3)`.

**No-judge mode:** When `use_llm_judge=False`, only gates 1 and 2 apply. Pass condition: both must pass. `equivalence_score = min(gate1, gate2)`.

---

### Task 1: Cap `max_tokens=200` on Frontier Model Calls in Harness

**Why:** `all-MiniLM-L6-v2` truncates at 256 wordpiece tokens. With `max_tokens=1024`, frontier model outputs get silently truncated before embedding comparison, producing misleading similarity scores.

**Files:**
- Modify: `src/validation/harness.py:85-110` (ValidationHarness.__init__ and DEFAULT_TASK_PROMPTS area)
- Modify: `src/validation/harness.py:130-170` (validate_pair, where client.complete is called)
- Test: `tests/test_harness.py`

**Step 1: Write the failing test**

Add to `tests/test_harness.py`:

```python
class TestMaxTokensCap:
    """Tests that frontier model calls use capped max_tokens."""

    @pytest.mark.asyncio
    async def test_complete_called_with_max_tokens_200(self, sample_pair):
        """Verify model completions are called with max_tokens=200."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
        ):
            mock_client = MagicMock()
            mock_client.complete = AsyncMock(return_value="Short response")
            mock_client_class.return_value = mock_client

            mock_calc = MagicMock()
            mock_calc.compute_semantic_similarity = MagicMock(return_value=0.95)
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
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_harness.py::TestMaxTokensCap -v`
Expected: FAIL (currently calls complete() without explicit max_tokens, defaulting to 1024)

**Step 3: Write minimal implementation**

In `src/validation/harness.py`:
1. Add a class constant `VALIDATION_MAX_TOKENS = 200`
2. Pass `max_tokens=self.VALIDATION_MAX_TOKENS` to both `client.complete()` calls in `eval_model()`

```python
# In ValidationHarness class, after __init__:
VALIDATION_MAX_TOKENS = 200  # Cap to fit within MiniLM-L6-v2 256 token window

# In eval_model(), change the two complete() calls:
verbose_out = await client.complete(verbose_prompt, max_tokens=self.VALIDATION_MAX_TOKENS)
compressed_out = await client.complete(compressed_prompt, max_tokens=self.VALIDATION_MAX_TOKENS)
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_harness.py::TestMaxTokensCap -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All existing tests still pass

---

### Task 2: Wire `compute_fact_overlap` into ValidationHarness

**Why:** `compute_fact_overlap()` exists in `metrics.py` but is never called during validation. We need it as gate 2.

**Files:**
- Modify: `src/validation/harness.py` (import and call compute_fact_overlap in eval_model)
- Modify: `src/validation/harness.py` (add fact_overlap to ValidationResult)
- Test: `tests/test_harness.py`

**Step 1: Update ValidationResult to include fact_overlap_scores**

In `src/validation/harness.py`, add to `ValidationResult`:

```python
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
    # NEW: per-model fact overlap and embedding scores for 3-gate system
    embedding_scores: dict[ModelType, float] | None = None
    fact_overlap_scores: dict[ModelType, float] | None = None
```

**Step 2: Write the failing test**

Add to `tests/test_harness.py`:

```python
class TestFactOverlapIntegration:
    """Tests that fact_overlap is computed during validation."""

    @pytest.mark.asyncio
    async def test_fact_overlap_scores_populated(self, sample_pair):
        """Verify fact_overlap_scores are included in ValidationResult."""
        with (
            patch("src.validation.harness.ModelClient") as mock_client_class,
            patch("src.validation.harness.EquivalenceCalculator") as mock_calc_class,
            patch("src.validation.harness.compute_fact_overlap", return_value=0.7) as mock_fo,
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
            assert result.fact_overlap_scores[ModelType.CLAUDE_SONNET] == 0.7
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_harness.py::TestFactOverlapIntegration -v`
Expected: FAIL (fact_overlap_scores not yet computed)

**Step 4: Implement fact_overlap computation in eval_model**

In `src/validation/harness.py`:

1. Add import: `from .metrics import compute_fact_overlap`
2. In `eval_model()`, compute fact_overlap for each task alongside the existing score
3. Average fact_overlap across tasks per model
4. Collect into result

```python
# In eval_model inner function, after computing verbose_out and compressed_out:
fact_score = compute_fact_overlap(verbose_out, compressed_out, calculator=self.metrics)
fact_scores.append(fact_score)

# Also track embedding similarity separately:
embedding_sim = self.metrics.compute_semantic_similarity(verbose_out, compressed_out)
embedding_sims.append(embedding_sim)
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_harness.py::TestFactOverlapIntegration -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All tests pass

---

### Task 3: Implement 3-Gate Pass/Fail Logic

**Why:** Currently pass/fail is `min_equiv >= threshold`. We need: all three gates (embedding >= 0.60, fact_overlap >= 0.55, judge >= 0.75) must pass independently.

**Files:**
- Modify: `src/validation/harness.py` (ValidationHarness.__init__ for thresholds, validate_pair for logic)
- Test: `tests/test_harness.py`

**Step 1: Add gate thresholds to ValidationHarness.__init__**

```python
def __init__(
    self,
    models: list[ModelType] | None = None,
    equivalence_threshold: float = 0.72,  # kept for backward compat, used as fallback
    tasks: list[TaskType] | None = None,
    cache: SemanticCache | None = None,
    use_llm_judge: bool = False,
    llm_judge_model: ModelType = ModelType.CLAUDE_SONNET,
    # 3-gate thresholds
    embedding_threshold: float = 0.60,
    fact_overlap_threshold: float = 0.55,
    judge_threshold: float = 0.75,
):
```

**Step 2: Write failing tests for 3-gate logic**

```python
class TestThreeGateLogic:
    """Tests for 3-gate pass/fail system."""

    @pytest.mark.asyncio
    async def test_all_gates_pass(self):
        """Pair passes when all three gates exceed thresholds."""
        # embedding=0.80 >= 0.60, fact=0.70 >= 0.55, judge=0.85 >= 0.75
        # -> passed=True, equivalence_score=min(0.80, 0.70, 0.85)=0.70
        ...

    @pytest.mark.asyncio
    async def test_embedding_gate_fails(self):
        """Pair fails when embedding similarity is below 0.60."""
        # embedding=0.50 < 0.60 -> fails even if others pass
        ...

    @pytest.mark.asyncio
    async def test_fact_overlap_gate_fails(self):
        """Pair fails when fact overlap is below 0.55."""
        # fact=0.40 < 0.55 -> fails even if others pass
        ...

    @pytest.mark.asyncio
    async def test_judge_gate_fails(self):
        """Pair fails when judge score is below 0.75."""
        # judge=0.60 < 0.75 -> fails even if others pass
        ...

    @pytest.mark.asyncio
    async def test_no_judge_mode_two_gates(self):
        """Without judge, only embedding and fact_overlap gates apply."""
        # embedding=0.80 >= 0.60, fact=0.70 >= 0.55 -> passed=True
        ...

    @pytest.mark.asyncio
    async def test_no_judge_mode_fact_fails(self):
        """Without judge, fact_overlap failure -> pair fails."""
        # embedding=0.80, fact=0.40 < 0.55 -> fails
        ...
```

**Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_harness.py::TestThreeGateLogic -v`

**Step 4: Implement 3-gate logic in validate_pair**

Replace the pass/fail logic at the end of `validate_pair()`:

```python
# 3-gate pass/fail
min_embedding = min(embedding_scores_dict.values())
min_fact_overlap = min(fact_overlap_scores_dict.values())

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

return ValidationResult(
    ...
    min_equivalence=min_equiv,
    passed=all_gates_pass,
    ...
)
```

**Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_harness.py::TestThreeGateLogic -v`

**Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`

---

### Task 4: Log All Three Metrics in Output JSONL

**Why:** We need all three gate scores per model in the output for analysis and debugging.

**Files:**
- Modify: `scripts/validate_batch.py:181-215` (save_single_result)
- Test: `tests/test_validate_batch_cost_estimator.py` (or new test file)

**Step 1: Update save_single_result to include new fields**

```python
def save_single_result(
    pair: CompressionPair,
    result: ValidationResult,
    output_path: Path,
    save_all: bool = False,
) -> bool:
    ...
    data["validation"] = {
        "passed": result.passed,
        "min_equivalence": result.min_equivalence,
        "compression_ratio": result.compression_ratio,
        "equivalence_scores": {
            model.value: score for model, score in result.equivalence_scores.items()
        },
        "llm_judge_used": result.llm_judge_used,
        # NEW: per-gate scores
        "embedding_scores": {
            model.value: score for model, score in result.embedding_scores.items()
        } if result.embedding_scores else None,
        "fact_overlap_scores": {
            model.value: score for model, score in result.fact_overlap_scores.items()
        } if result.fact_overlap_scores else None,
    }

    # Include LLM judge scores if available
    if result.llm_judge_scores:
        data["validation"]["llm_judge_scores"] = {
            model.value: score for model, score in result.llm_judge_scores.items()
        }
    ...
```

**Step 2: Write test verifying JSONL output contains gate scores**

**Step 3: Implement and verify**

Run: `python -m pytest tests/ -v`

---

### Task 5: Update validate_batch.py CLI

**Why:** Users need to be able to configure gate thresholds from the CLI.

**Files:**
- Modify: `scripts/validate_batch.py` (argument parser + harness initialization)

**Step 1: Add CLI arguments**

```python
parser.add_argument(
    "--embedding-threshold",
    type=float,
    default=0.60,
    help="Minimum embedding similarity to pass gate 1 (default: 0.60)",
)
parser.add_argument(
    "--fact-overlap-threshold",
    type=float,
    default=0.55,
    help="Minimum fact overlap to pass gate 2 (default: 0.55)",
)
parser.add_argument(
    "--judge-threshold",
    type=float,
    default=0.75,
    help="Minimum LLM judge score to pass gate 3 (default: 0.75)",
)
```

**Step 2: Pass thresholds to ValidationHarness**

```python
harness = ValidationHarness(
    models=models,
    equivalence_threshold=args.threshold,
    tasks=tasks,
    cache=cache,
    use_llm_judge=args.use_llm_judge,
    embedding_threshold=args.embedding_threshold,
    fact_overlap_threshold=args.fact_overlap_threshold,
    judge_threshold=args.judge_threshold,
)
```

**Step 3: Update cost estimate for max_tokens=200**

In `estimate_validation_cost()`, lower `avg_output_tokens_per_call` default from 250 to 150 (since we cap at 200 tokens).

**Step 4: Run full test suite**

Run: `python -m pytest tests/ -v`

---

### Task 6: Update Existing Tests

**Why:** Existing tests mock different behavior and may break with the new gate logic.

**Files:**
- Modify: `tests/test_harness.py` (update existing mocked tests for new fields)
- Modify: `tests/test_metrics.py` (add test for compute_fact_overlap)

**Step 1: Update existing harness tests**

The existing `test_validate_pair_mocked` and similar tests need to:
1. Also mock `compute_fact_overlap` (since it's now called in validate_pair)
2. Check that `embedding_scores` and `fact_overlap_scores` are populated

**Step 2: Add fact_overlap unit tests to test_metrics.py**

```python
class TestFactOverlap:
    """Tests for compute_fact_overlap."""

    def test_identical_text(self):
        """Identical text should have high fact overlap."""
        text = "The revenue increased 15%. Headcount grew to 500. Profit margin is 20%."
        overlap = compute_fact_overlap(text, text)
        assert overlap > 0.8

    def test_missing_facts(self):
        """When compressed drops facts, overlap should decrease."""
        verbose = "Revenue: $10M. Headcount: 500. Founded 2020. CEO: John Smith."
        compressed = "Revenue: $10M. Headcount: 500."
        overlap = compute_fact_overlap(verbose, compressed)
        assert overlap < 0.8  # Missing 2 of 4 facts

    def test_empty_text(self):
        """Empty text should return 0."""
        assert compute_fact_overlap("", "something") == 0.0
        assert compute_fact_overlap("something", "") == 0.0
```

**Step 3: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: ALL tests pass

---

## Final Verification

After all tasks are complete:

1. Run: `python -m pytest tests/ -v` — all tests pass
2. Run: `python -m pytest tests/ -v --tb=short` — no warnings
3. Run linter: `ruff check src/ scripts/ tests/`
4. Dry-run cost estimate to verify lower cost:
   ```
   python scripts/validate_batch.py \
     -i models/eval/pairs_sample300_seed42_qwen_step004500.jsonl \
     -o /dev/null \
     --estimate-only
   ```

---

## Summary of Changed Files

| File | Changes |
|------|---------|
| `src/validation/harness.py` | Add `VALIDATION_MAX_TOKENS=200`, `embedding_threshold`, `fact_overlap_threshold`, `judge_threshold` to `__init__`; compute fact_overlap + embedding per model-task in `eval_model`; 3-gate pass/fail logic; new fields on `ValidationResult` |
| `src/validation/metrics.py` | No changes needed (compute_fact_overlap already exists) |
| `scripts/validate_batch.py` | Add CLI args for gate thresholds; pass to harness; log new fields in JSONL; lower default output tokens in cost estimate |
| `tests/test_harness.py` | New test classes: `TestMaxTokensCap`, `TestFactOverlapIntegration`, `TestThreeGateLogic`; update existing mocked tests |
| `tests/test_metrics.py` | New `TestFactOverlap` class |
