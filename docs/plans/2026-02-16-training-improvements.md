# Training Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix overfitting, normalize loss values to per-token cross-entropy, and add a Tinker training visualizer.

**Architecture:** Three independent workstreams: (1) config + early stopping changes to `train_tinker.py`, (2) per-token loss normalization in the training loop and validation, (3) a standalone Tinker visualizer script that parses `metrics.jsonl`.

**Tech Stack:** Python, YAML config, matplotlib for visualization, pytest for tests.

---

## Context

Training run on Qwen3-8B with LoRA r=64 showed severe overfitting:
- Best val loss ~93.7 at epoch 1 end (step 4962)
- Val loss diverged to 130.4 by epoch 3 end
- Root causes: over-parameterized LoRA (r=64), no dropout, no mid-epoch eval, no early stopping
- Loss values are sum-reduced (100-400 range), not per-token (~2-5 range)

Research-backed recommendations for Qwen3-8B:
- LoRA r=16, alpha=32, dropout=0.05 (from Axolotl Qwen3 config, QLoRA paper)
- epochs=2 with early stopping (from Raschka, QLoRA paper, empirical data)
- eval_interval_steps=250, early_stopping_patience=5
- lr=2e-4 stays (confirmed correct)

---

### Task 1: Update training.yaml config

**Files:**
- Modify: `configs/training.yaml`

**Step 1: Update cloud LoRA and training params**

Change the `cloud` section:

```yaml
cloud:
  model: "Qwen/Qwen3-8B"
  lora:
    rank: 16           # Down from 64; r=16 is sweet spot for 8B models (Axolotl Qwen3 config)
    alpha: 32           # 2x rank convention
    dropout: 0.05       # Mild regularization (was 0.0)
    target_modules:
      - q_proj
      - k_proj
      - v_proj
      - o_proj
      - gate_proj
      - up_proj
      - down_proj
  training:
    epochs: 2                       # Down from 3; expect early stopping at ~1-1.5 epochs
    batch_size: 4
    learning_rate: 2.0e-4
    warmup_ratio: 0.03
    max_seq_length: 2048
    log_interval_steps: 10
    checkpoint_interval_steps: 250
    eval_interval_steps: 250        # Was 0 (disabled!); now eval every 250 steps
    eval_at_epoch_end: true
    checkpoint_ttl_seconds: null
    resume_from_checkpoint: true
    # Early stopping
    early_stopping_patience: 5      # NEW: stop after 5 evals with no improvement
    early_stopping_threshold: 0.01  # NEW: min improvement to count as "better" (absolute)
```

**Step 2: Verify config loads correctly**

Run: `python -c "from src.training.train_tinker import load_config_from_yaml; from pathlib import Path; c = load_config_from_yaml(Path('configs/training.yaml')); print(f'rank={c.lora.rank}, alpha={c.lora.alpha}, dropout={c.lora.dropout}, epochs={c.epochs}, eval_interval={c.eval_interval_steps}')"`

Expected: `rank=16, alpha=32, dropout=0.05, epochs=2, eval_interval=250`

Note: This will fail until Task 2 adds the `dropout` and early stopping fields to the config loading code.

**Step 3: Commit**

```bash
git add configs/training.yaml
git commit -m "config: tune LoRA for Qwen3-8B (r=16, dropout=0.05, early stopping)"
```

---

### Task 2: Add early stopping config and logic to train_tinker.py

**Files:**
- Modify: `src/training/train_tinker.py`
- Modify: `configs/training.yaml` (already done in Task 1)

**Step 2.1: Add early stopping fields to TinkerTrainingConfig**

In `TinkerTrainingConfig` dataclass (after `resume_from_checkpoint`):

```python
    # Early stopping
    early_stopping_patience: int = 0  # 0 = disabled; number of evals with no improvement before stopping
    early_stopping_threshold: float = 0.01  # minimum improvement to count as "better"
```

**Step 2.2: Add dropout field to TinkerLoRAConfig loading**

In `load_config_from_yaml`, ensure `dropout` is read from YAML:

```python
    return TinkerTrainingConfig(
        ...
        lora=TinkerLoRAConfig(
            rank=lora_config.get("rank", 16),
            alpha=lora_config.get("alpha", 32),
            dropout=lora_config.get("dropout", 0.05),
            target_modules=lora_config.get(...),
        ),
        ...
        early_stopping_patience=training_config.get("early_stopping_patience", 0),
        early_stopping_threshold=training_config.get("early_stopping_threshold", 0.01),
    )
```

**Step 2.3: Implement early stopping in the training loop**

In `_train_with_service_client_sdk`, before the epoch loop, add early stopping state:

```python
    # Early stopping state
    best_val_loss: float | None = None
    evals_without_improvement = 0
    stopped_early = False
```

After each validation evaluation (both mid-epoch and epoch-end), add early stopping check:

```python
    if val_loss is not None and config.early_stopping_patience > 0:
        if best_val_loss is None or val_loss < best_val_loss - config.early_stopping_threshold:
            best_val_loss = val_loss
            evals_without_improvement = 0
        else:
            evals_without_improvement += 1
            logger.info(
                "No val improvement for %d/%d evals (best=%.4f, current=%.4f)",
                evals_without_improvement, config.early_stopping_patience,
                best_val_loss, val_loss,
            )
            if evals_without_improvement >= config.early_stopping_patience:
                logger.info("Early stopping triggered at step %d", current_step)
                stopped_early = True
```

Break out of the epoch loop when `stopped_early` is True. Also break out of the inner batch loop.

**Step 2.4: Update TinkerLoRAConfig defaults to match new recommendations**

```python
@dataclass
class TinkerLoRAConfig:
    rank: int = 16      # was 64
    alpha: int = 32     # was 128
    dropout: float = 0.05  # was 0.0
    ...
```

**Step 2.5: Run existing tests**

Run: `pytest tests/test_train_tinker.py -v`
Expected: All existing tests pass.

**Step 2.6: Commit**

```bash
git add src/training/train_tinker.py
git commit -m "feat: add early stopping support to Tinker training"
```

---

### Task 3: Normalize loss to per-token cross-entropy

**Files:**
- Modify: `src/training/train_tinker.py`
- Modify: `src/training/tinker_data.py`

**Step 3.1: Track completion token count in batch iteration**

The weights list in `render_chat_example` has 1s for completion tokens and 0s for prompt tokens. We need to count the 1s to normalize loss.

In `_iter_training_batches`, track the completion token count alongside total token count:

```python
def _iter_training_batches(
    train_file: Path,
    tokenizer: Any,
    tinker_module: ModuleType,
    batch_size: int,
) -> Iterator[tuple[list[Any], int, int]]:
    """Yield SDK-ready training batches with token counts from chat JSONL data.
    
    Returns (batch, total_tokens, completion_tokens) tuples.
    """
    batch: list[Any] = []
    batch_tokens = 0
    batch_completion_tokens = 0
    
    # ... inside the loop:
    local_datum = render_chat_example(messages, tokenizer)
    sdk_datum, token_count = _to_sdk_datum(local_datum, tinker_module)
    completion_count = sum(local_datum.loss_fn_inputs["weights"])
    batch.append(sdk_datum)
    batch_tokens += token_count
    batch_completion_tokens += completion_count
    
    if len(batch) >= batch_size:
        yield batch, batch_tokens, batch_completion_tokens
        batch = []
        batch_tokens = 0
        batch_completion_tokens = 0
    
    if batch:
        yield batch, batch_tokens, batch_completion_tokens
```

**Step 3.2: Normalize training loss in the training loop**

After extracting `step_loss`, normalize by completion tokens:

```python
    step_loss = _extract_loss(metrics)
    if step_loss is not None and completion_tokens > 0:
        step_loss_per_token = step_loss / completion_tokens
    else:
        step_loss_per_token = step_loss
```

Log both values:

```python
    _append_train_log_line(
        train_log_path,
        f"Iter {current_step}: Train loss {step_loss_per_token:.4f} "
        f"| Total loss {step_loss:.4f} "
        f"| Tokens/sec {tokens_per_sec:.1f} | Peak mem 0.0 GB",
    )
```

**Step 3.3: Normalize validation loss**

In `_run_validation`, normalize each batch's val loss by its completion token count:

```python
def _run_validation(
    training_client, valid_file, tokenizer, tinker_module, batch_size,
) -> tuple[float | None, int]:
    total_loss = 0.0
    total_completion_tokens = 0
    val_batches = 0
    
    for batch, _, completion_tokens in _iter_training_batches(...):
        forward_result = training_client.forward(batch, "cross_entropy").result()
        metrics = getattr(forward_result, "metrics", {}) or {}
        val_loss = _extract_loss(metrics)
        if val_loss is not None:
            total_loss += val_loss
            total_completion_tokens += completion_tokens
        val_batches += 1
    
    if total_completion_tokens == 0:
        return None, val_batches
    return total_loss / total_completion_tokens, val_batches
```

**Step 3.4: Update all callers of `_iter_training_batches` and `_run_validation`**

Ensure the training loop unpacks the new 3-tuple:

```python
for batch, batch_tokens, completion_tokens in _iter_training_batches(...):
```

**Step 3.5: Run tests**

Run: `pytest tests/test_train_tinker.py tests/test_tinker_data.py -v`
Expected: All pass (existing tests should still work).

**Step 3.6: Commit**

```bash
git add src/training/train_tinker.py src/training/tinker_data.py
git commit -m "feat: normalize training/validation loss to per-token cross-entropy"
```

---

### Task 4: Build Tinker training visualizer

**Files:**
- Create: `scripts/visualize_tinker_training.py`
- Create: `tests/test_visualize_tinker_training.py`

**Step 4.1: Write the visualizer script**

The script should:
1. Parse `metrics.jsonl` (structured JSON, one per line, with `type: "train"` or `type: "val"`)
2. Plot train loss vs val loss over steps
3. Mark epoch boundaries with vertical lines
4. Mark the best val loss point
5. Mark early stopping point if applicable
6. Save as PNG to the output directory

Key design decisions:
- Use `metrics.jsonl` not `train.log` (structured data is more reliable)
- Support both sum-reduced and per-token loss (detect from field names)
- Show smoothed train loss (EMA) alongside raw values (train loss is noisy)
- Two subplot layout: (1) loss curves, (2) tokens/sec throughput

CLI interface:
```
python scripts/visualize_tinker_training.py [--metrics-path PATH] [--output PATH] [--dpi DPI]
```

Default metrics path: `models/adapters/tinker/metrics.jsonl`

**Step 4.2: Write tests for the log parser**

Test the JSONL parsing with mock data, not the plotting.

**Step 4.3: Run tests**

Run: `pytest tests/test_visualize_tinker_training.py -v`

**Step 4.4: Generate visualization from existing training run**

Run: `python scripts/visualize_tinker_training.py`
Expected: PNG saved to `models/adapters/tinker/training_curves.png`

**Step 4.5: Commit**

```bash
git add scripts/visualize_tinker_training.py tests/test_visualize_tinker_training.py
git commit -m "feat: add Tinker training curve visualizer"
```

---

### Task 5: Add tests for early stopping logic

**Files:**
- Modify: `tests/test_train_tinker.py`

**Step 5.1: Test early stopping triggers correctly**

```python
def test_early_stopping_triggers_after_patience():
    """Verify early stopping fires after N evals with no improvement."""
    ...

def test_early_stopping_resets_on_improvement():
    """Verify counter resets when val loss improves."""
    ...

def test_early_stopping_disabled_when_patience_zero():
    """Verify no early stopping when patience=0."""
    ...
```

**Step 5.2: Run all tests**

Run: `pytest tests/ -v --tb=short`
Expected: All pass.

**Step 5.3: Commit**

```bash
git add tests/test_train_tinker.py
git commit -m "test: add early stopping and loss normalization tests"
```

---

### Task 6: Update config defaults and documentation

**Files:**
- Modify: `configs/training.yaml` (update cost estimates section)

**Step 6.1: Update cost estimates in training.yaml**

With epochs=2 and early stopping likely at ~1.5 epochs, costs drop:
- Previous: 3 epochs * 19,845 * 500 tokens * $0.40/M = ~$11.91
- New: ~1.5 epochs * 19,845 * 500 tokens * $0.40/M = ~$5.95

**Step 6.2: Final test run**

Run: `pytest tests/ -v --tb=short`
Expected: All pass.

**Step 6.3: Commit**

```bash
git add configs/training.yaml
git commit -m "docs: update cost estimates for 2-epoch training with early stopping"
```
