# Setup Guide

## Overview

This project uses a **hybrid workflow**:
- **Tinker** (cloud) — Production training runs
- **MLX** (local M4 Pro) — Quick iteration, inference, validation

## Prerequisites

- **Local**: MacBook M4 Pro 24GB, macOS 15+
- **Cloud**: Tinker account (https://tinker.thinkingmachines.ai)
- **Python**: 3.11+

---

## Part 1: Local Environment (MLX)

### 1.1 Base Setup

```bash
# Clone project
git clone <repo-url>
cd compression-layer

# Create environment
python3 -m venv .venv
source .venv/bin/activate

# Install base deps
pip install -e ".[dev]"
```

### 1.2 Install MLX (Apple Silicon)

```bash
# MLX for local training/inference
pip install -U mlx-lm

# Verify
python -c "import mlx; print('MLX OK')"
```

### 1.3 Download Qwen Models (Local)

```bash
# Login to HuggingFace
pip install "huggingface_hub[cli]"
huggingface-cli login

# Download 4-bit Qwen for local use
huggingface-cli download mlx-community/Qwen2.5-7B-Instruct-4bit

# Test inference
python -m mlx_lm.generate \
  --model mlx-community/Qwen2.5-7B-Instruct-4bit \
  --prompt "Compress this: The user is a software engineer at Google" \
  --max-tokens 100
```

### 1.4 Local Training (Small Scale)

For quick iteration with Qwen3-4B:

```bash
# Fine-tune with MLX LoRA (recommended wrapper script)
python scripts/train_local.py --train

# Or run mlx_lm directly
python -m mlx_lm.lora \
  --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
  --train \
  --data ./data/training \
  --iters 500 \
  --batch-size 2 \
  --lora-rank 8

# Test adapter
python -m mlx_lm.generate \
  --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
  --adapter-path models/runs/mlx/latest/adapter \
  --prompt "Compress: ..."
```

**Memory usage on M4 Pro 24GB:**
| Model | QLoRA Memory | Speed |
|-------|--------------|-------|
| Qwen3-4B | ~8GB | ~150 tok/s |
| Qwen2.5-7B | ~12GB | ~100 tok/s |

### 1.5 Run Outputs

Local MLX runs are stored per run under:

```
models/runs/mlx/<timestamp>
```

Each run directory captures run metadata (`run.json`), training logs (`train.log`,
`train.err`), and adapter outputs under `adapter/`.
The newest run is always accessible via the `models/runs/mlx/latest` symlink when
using `scripts/train_local.py`.

---

## Part 2: Cloud Training (Tinker)

### 2.1 Tinker Setup

```bash
# Install Tinker SDK
pip install tinker

# Set API key
export TINKER_API_KEY=your_key_here
```

Or add to `.env`:
```bash
TINKER_API_KEY=tk_...
```

### 2.2 Training with Tinker

Use the CLI wrapper for end-to-end training:

```bash
python scripts/train_tinker.py \
  --config configs/training.yaml \
  --output models/adapters/tinker
```

ServiceClient mode writes artifacts directly in `--output`:

- `tinker_run.json` (resume state + checkpoint history)
- `run.json` (MLflow-compatible run metadata)
- `train.log` (train/val lines parsed by `scripts/mlflow_logger.py`)
- `metrics.jsonl` (structured metric events)

Auto-resume is enabled by default and uses `latest_checkpoint_path` from
`tinker_run.json`.

### 2.3 Tinker CLI Workflow

```bash
# Start training with config defaults
python scripts/train_tinker.py \
  --config configs/training.yaml \
  --output models/adapters/tinker

# Customize training telemetry/checkpoints
python scripts/train_tinker.py \
  --config configs/training.yaml \
  --output models/adapters/tinker \
  --log-interval-steps 10 \
  --checkpoint-interval-steps 250 \
  --eval-interval-steps 100

# Disable auto-resume for a fresh run in same output dir
python scripts/train_tinker.py \
  --config configs/training.yaml \
  --output models/adapters/tinker \
  --no-resume

# Check status (legacy Client API mode)
python scripts/train_tinker.py --status <job-id>

# Inspect service-mode run artifacts
cat models/adapters/tinker/tinker_run.json
cat models/adapters/tinker/run.json

# Send artifacts to MLflow/DagsHub
python scripts/mlflow_logger.py \
  --run-dir models/adapters/tinker \
  --experiment-name "compression-v2" \
  --dagshub-owner Sudhendra \
  --dagshub-repo compression-layer
```

Additional useful flags:

- `--no-eval-at-epoch-end`
- `--checkpoint-ttl-seconds <seconds>`

### 2.4 Cost Estimation

| Model | Per 1M Tokens | 10K pairs (~5M tok) | 50K pairs (~25M tok) |
|-------|---------------|---------------------|----------------------|
| Qwen3-4B | $0.22 | ~$1.10 | ~$5.50 |
| Qwen3-8B | $0.40 | ~$2.00 | ~$10.00 |
| Qwen3-30B-A3B | $0.36 | ~$1.80 | ~$9.00 |

**5 training runs**: $10-50 total

---

## Part 3: API Keys

Create `.env`:

```bash
# Frontier models (validation)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...

# HuggingFace (model downloads)
HF_TOKEN=hf_...
HF_HUB_ENABLE_HF_TRANSFER=1

# Tinker (cloud training)
TINKER_API_KEY=tk_...

# DagsHub/MLflow (experiment tracking, optional)
DAGSHUB_OWNER=Sudhendra
DAGSHUB_REPO=compression-layer
# MLFLOW_TRACKING_URI is auto-derived from owner/repo if omitted
```

---

## Part 4: Recommended Workflow

### Phase 1-2: Local (MLX)
- Generate seed pairs via Claude API
- Run validation harness
- Quick experiments with Qwen3-4B locally

### Phase 3-4: Cloud (Tinker)
- Train Qwen3-8B on validated dataset
- Iterate on hyperparameters
- ~$20-50 total cost

### Phase 5-6: Local (MLX)
- Download trained adapter
- Run inference locally on M4 Pro
- Integrate with memory system

---

## Quick Commands

```bash
# Local inference
python -m mlx_lm.generate --model mlx-community/Qwen3-4B-Instruct-2507-8bit --prompt "..."

# Local training (wrapper script with run storage)
python scripts/train_local.py --train

# Cloud training (production)
python scripts/train_tinker.py --config configs/training.yaml --output models/adapters/tinker

# Post-training MLflow logging
python scripts/mlflow_logger.py --experiment-name "compression-v2" --dagshub-owner Sudhendra

# Data preprocessing pipeline
python scripts/preprocess_synthetic.py --input data/synthetic/nl_v2.jsonl --output data/validated/nl_pairs.jsonl
python scripts/format_training_data.py --input data/validated --output data/training
python scripts/data_sanitization.py --input data/training/train.jsonl --sanitized data/training/sanitized_train.jsonl --unsanitized data/training/unsanitized_train.jsonl

# Run validation
python scripts/validate_batch.py --input data/seed/pairs.jsonl
```

---

## Troubleshooting

### MLX "out of memory"
- Use smaller model: `Qwen3-4B` instead of `8B`
- Reduce batch size to 1
- Use 4-bit quantized models from `mlx-community/`

### Tinker job failed
- Check dataset format (JSONL with `text` or `messages` field)
- Verify API key in `.env` or shell: `TINKER_API_KEY`
- Inspect run metadata: `models/adapters/tinker/tinker_run.json`
- Inspect MLflow metadata/logs: `models/adapters/tinker/run.json`, `models/adapters/tinker/train.log`
- Re-run same command to resume from latest checkpoint (default)
- Add `--no-resume` to force a fresh run in an existing output directory

### Slow local inference
- Ensure using 4-bit model: `*-4bit`
- Increase wired memory limit (macOS 15+):
  ```bash
  sudo sysctl iogpu.wired_limit_mb=20000
  ```
