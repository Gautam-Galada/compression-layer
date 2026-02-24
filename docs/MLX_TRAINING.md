# MLX Local Training Runbook

This guide walks through downloading the base model, running local MLX LoRA
training, and monitoring logs/checkpoints on Apple Silicon.

## Prerequisites

- Activate the project venv:

```bash
source .venv/bin/activate
```

- Ensure MLX is installed:

```bash
pip install -U mlx-lm
```

- Authenticate with Hugging Face if needed:

```bash
hf auth login
```

## Download the base model

Recommended local model (M4 Pro 24GB):

```bash
hf download mlx-community/Qwen3-4B-Instruct-2507-8bit
```

## Run training

Default run (uses config defaults):

```bash
python scripts/train_local.py --train
```

Override the model explicitly (optional):

```bash
python scripts/train_local.py --train --model mlx-community/Qwen3-4B-Instruct-2507-8bit
```

Run the Tinker workflow locally (same CLI flow, local backend):

```bash
python scripts/train_tinker.py \
  --backend local \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/adapters/tinker
```

## Evaluate the latest adapter

If you trained with the default run storage, evaluation will use
`models/runs/mlx/latest/adapter` automatically.

```bash
python scripts/train_local.py --evaluate
```

To evaluate a specific run, pass the adapter path explicitly:

```bash
python scripts/train_local.py --evaluate --adapter-path models/runs/mlx/<timestamp>/adapter
```

Evaluate with the Tinker evaluation workflow locally:

```bash
python scripts/evaluate_tinker.py \
  --backend local \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --adapter-path models/adapters/tinker \
  --output models/eval/tinker_eval_local.jsonl
```

## Where outputs are stored

Each run writes to a timestamped directory:

```
models/runs/mlx/<timestamp>/
```

Contents include:
- `run.json` (metadata)
- `train.log` (stdout)
- `train.err` (stderr)
- `adapter/` (LoRA checkpoints)

The latest run is available at:

```
models/runs/mlx/latest
```

## Monitor logs during training

Follow training output:

```bash
tail -f models/runs/mlx/latest/train.log
```

Follow stderr (warnings/errors):

```bash
tail -f models/runs/mlx/latest/train.err
```

## Quick sanity checks

- `train.log` updates every `steps_per_report` steps.
- `adapter/` gains `.safetensors` files at each `save_every` interval.
- If training fails, check `train.err` and the CLI error message for details.

## Notes on checkpointing

- **Gradient checkpointing** reduces memory usage during training. It does **not**
  control whether LoRA adapters are saved to disk.
- Adapter checkpoints are saved according to `save_every` and live under
  `models/runs/mlx/<timestamp>/adapter/`.

## Data preprocessing pipeline

Before training, raw synthetic data must be preprocessed and sanitized. The full
pipeline is:

### Step 1: Preprocess synthetic pairs

Strip generation artifacts (`<think>`, `<tool_call>` tags) and filter by
compression ratio:

```bash
# NL pairs (stricter filtering: 0.95 char/token ratio)
python scripts/preprocess_synthetic.py \
  --input data/synthetic/nl_v2.jsonl \
  --output data/validated/nl_pairs.jsonl \
  --rejected data/validated/rejected_nl_pairs.jsonl \
  --max-char-ratio 0.95 \
  --max-token-ratio 0.95

# Code pairs (default 1.0 thresholds)
python scripts/preprocess_synthetic.py \
  --input data/synthetic/code_v2.jsonl \
  --output data/validated/code_pairs.jsonl \
  --rejected data/validated/rejected_code_pairs.jsonl \
  --max-char-ratio 1.0 \
  --max-token-ratio 1.0
```

### Step 2: Format into train/valid/test splits

```bash
python scripts/format_training_data.py \
  --input data/validated \
  --output data/training \
  --train-ratio 0.8 --valid-ratio 0.1 --test-ratio 0.1 \
  --seed 42
```

### Step 3: Sanitize all splits

```bash
for split in train valid test; do
  python scripts/data_sanitization.py \
    --input "data/training/${split}.jsonl" \
    --sanitized "data/training/sanitized_${split}.jsonl" \
    --unsanitized "data/training/unsanitized_${split}.jsonl"
done

# Promote sanitized files
for split in train valid test; do
  mv "data/training/sanitized_${split}.jsonl" "data/training/${split}.jsonl"
done
```

### Current data state (v2)

| File | Count |
|------|-------|
| `data/training/train.jsonl` | 19,845 |
| `data/training/valid.jsonl` | 2,473 |
| `data/training/test.jsonl` | 2,497 |

## Post-training: MLflow/DagsHub logging

After a training run completes, log metrics and artifacts to DagsHub/MLflow using
`scripts/mlflow_logger.py`. This is **not** called automatically by the training
scripts -- you must run it manually.

### Prerequisites

```bash
# Ensure dagshub and mlflow are installed
pip install dagshub mlflow

# Or install via optional dependency group
pip install -e ".[mlflow]"
```

### Usage

```bash
# Log the latest MLX run (auto-detected)
python scripts/mlflow_logger.py \
  --experiment-name "compression-v2" \
  --dagshub-owner Sudhendra \
  --dagshub-repo compression-layer

# Log a specific run directory
python scripts/mlflow_logger.py \
  --run-dir models/runs/mlx/2026-01-30_17-14-36 \
  --experiment-name "compression-v1" \
  --dagshub-owner Sudhendra \
  --dagshub-repo compression-layer
```

The logger reads `run.json` and `train.log` from the run directory and logs:
- **Params**: model, git_sha, lora_rank, batch_size, learning_rate, etc.
- **Metrics**: train_loss, val_loss, tokens_per_sec, peak_mem_gb (per step)
- **Artifacts**: run.json, train.log, adapter weights, loss curve plot

### Environment variables

You can set defaults via environment variables instead of CLI flags:

```bash
DAGSHUB_OWNER=Sudhendra
DAGSHUB_REPO=compression-layer
MLFLOW_TRACKING_URI=https://dagshub.com/Sudhendra/compression-layer.mlflow
```

**Note**: The default `--dagshub-owner` in the script is `Gautam-Galada` (the
contributor who wrote it). Override with `--dagshub-owner Sudhendra` or set the
`DAGSHUB_OWNER` env var.
