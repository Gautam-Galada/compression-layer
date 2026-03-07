# Semantic Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Dataset on HF](https://img.shields.io/badge/HuggingFace-Dataset-yellow.svg)](https://huggingface.co/datasets/Sudhendra/semantic-compression-sft)

**Can we fine-tune small LLMs to compress context while preserving semantic equivalence across models?**

This project trains LoRA adapters on small (3B–8B) language models to rewrite
verbose LLM context into compressed representations that produce equivalent
reasoning outputs from Claude, GPT, and Gemini — reducing token usage while
maintaining answer quality.

> **Read the full write-up:** [Semantic Compression — Research Overview](https://x.com/wthagi/status/2028228181212451286)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/pipeline-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/pipeline-light.svg">
    <img alt="Pipeline: Raw Corpora → Synthetic Generation → Preprocessing → SFT Training → Compression Eval → Equivalence Eval" src="assets/pipeline-light.svg" width="100%">
  </picture>
</p>

---

## Key Results

Summary of trained adapter evaluations on the full test set (2,497 examples from
[`Sudhendra/semantic-compression-sft`](https://huggingface.co/datasets/Sudhendra/semantic-compression-sft)):

### Compression (Token Ratio)

| Model | Params | Backend | LoRA Config | Avg Token Ratio | Avg Input Tokens | Avg Output Tokens |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: |
| Nanbeige4.1-3B (8-bit) | 3B | MLX (local) | rank 8, lr 1e-4, 500 iters | **34.8%** | 276.3 | 103.2 |
| Qwen3-8B | 8B | Tinker (cloud) | rank 16, lr 2e-4, 4962 steps | **38.6%** | 237.5 | 98.1 |

> **Token ratio** = `output_tokens / input_tokens`. Lower is better. A ratio of 34.8% means
> the compressed output uses ~35% of the original token count.

### Equivalence (Cross-Model Scoring)

Equivalence is measured using a **3-gate system**. Both the original verbose text
and the compressed output are sent to three frontier models (Claude, GPT, Gemini),
which perform fact-extraction tasks. The outputs are compared through three gates:

| Gate | Metric | Threshold | What it catches |
| :--- | :--- | ---: | :--- |
| 1 | Embedding similarity (MiniLM-L6-v2) | >= 0.60 | Gross topic drift, garbled outputs |
| 2 | Fact overlap (atomic fact coverage) | >= 0.55 | Dropped facts, numbers, parameters |
| 3 | LLM judge | >= 0.75 | Subtle reasoning gaps, quality issues |

A sample **passes** only when the minimum score across all three models exceeds
the threshold on **every active gate**.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/equivalence-eval-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="assets/equivalence-eval-light.svg">
    <img alt="Equivalence Evaluation: 3-Gate Scoring — Compression pair → Frontier model (Claude/GPT/Gemini) fact extraction → 3-gate comparison (embedding, fact overlap, LLM judge) → per-model aggregation → final pass/fail verdict" src="assets/equivalence-eval-light.svg" width="100%">
  </picture>
</p>

| Model | Sample (seed 42) | Pass Rate | Avg Min-Equiv | Median Min-Equiv |
| :--- | ---: | ---: | ---: | ---: |
| Qwen3-8B (step 4500) | 300 | **2.0%** | 0.390 | 0.413 |
| Nanbeige 3B (iter 500) | 300 | **1.3%** | 0.402 | 0.434 |

<details>
<summary>Qwen3-8B gate-level breakdown</summary>

**Gate failure rates** (how often each gate was the bottleneck):

| Gate | Fail Rate | Avg Score |
| :--- | ---: | ---: |
| Embedding (>= 0.60) | 15.3% | 0.782 |
| Fact overlap (>= 0.55) | 76.0% | 0.535 |
| LLM judge (>= 0.75) | 96.3% | 0.584 |

Without the LLM judge (gates 1+2 only), the pass rate would be **24.0%**.

**Per-model avg scores** (across 300 samples):

| Model | Embedding | Fact Overlap | LLM Judge |
| :--- | ---: | ---: | ---: |
| Claude Sonnet | 0.783 | 0.539 | 0.572 |
| GPT-4o-mini | 0.795 | 0.525 | 0.607 |
| Gemini Flash | 0.768 | 0.542 | 0.574 |

**Per-domain results** (3-gate, all models):

| Domain | n | Pass Rate | Avg Min-Equiv | Median Min-Equiv |
| :--- | ---: | ---: | ---: | ---: |
| NL | 95 | 2.1% | 0.320 | 0.304 |
| Mixed | 171 | 1.8% | 0.415 | 0.444 |
| Code | 34 | 2.9% | 0.459 | 0.450 |

</details>

<details>
<summary>Nanbeige 3B gate-level breakdown</summary>

**Gate failure rates** (how often each gate was the bottleneck):

| Gate | Fail Rate | Avg Score |
| :--- | ---: | ---: |
| Embedding (>= 0.60) | 12.7% | 0.804 |
| Fact overlap (>= 0.55) | 74.0% | 0.556 |
| LLM judge (>= 0.75) | 97.0% | 0.596 |

Without the LLM judge (gates 1+2 only), the pass rate would be **25.7%**.

**Per-model avg scores** (across 300 samples):

| Model | Embedding | Fact Overlap | LLM Judge |
| :--- | ---: | ---: | ---: |
| Claude Sonnet | 0.805 | 0.553 | 0.594 |
| GPT-4o-mini | 0.817 | 0.548 | 0.618 |
| Gemini Flash | 0.790 | 0.566 | 0.577 |

**Per-domain results** (3-gate, all models):

| Domain | n | Pass Rate | Avg Min-Equiv | Median Min-Equiv |
| :--- | ---: | ---: | ---: | ---: |
| NL | 95 | 3.2% | 0.346 | 0.350 |
| Mixed | 171 | 0.6% | 0.419 | 0.444 |
| Code | 34 | 0.0% | 0.476 | 0.456 |

</details>

> Samples are domain-stratified
> (95 NL / 171 mixed / 34 code) to match the test set distribution.

<details>
<summary>Training run details</summary>

**Nanbeige 3B — 500 iterations (MLX)**
- Model: `mlx-community/Nanbeige4.1-3B-8bit`
- LoRA: rank 8, alpha 16, 16 layers
- Config: 500 iters, batch size 4, lr 1e-4
- Eval artifact: `models/eval/ratio_nanbeige_iter500.jsonl`

**Qwen3-8B (Tinker cloud)**
- Model: `Qwen/Qwen3-8B`
- LoRA: rank 16, alpha 32, dropout 0.05
- Config: 2 epochs, batch size 4, lr 2e-4
- Status: completed, early-stopped at step 4962 (planned 9924)
- Best checkpoint: step 4500 (val_loss = 0.2579)
- Train examples: 19,845

</details>

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/Sudhendra/compression-layer.git
cd compression-layer

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Copy and configure environment variables
cp .env.example .env
# Edit .env with your API keys

# Run tests
pytest tests/ -v
```

### Environment Variables

Create a `.env` file (see `.env.example`):

```bash
ANTHROPIC_API_KEY=sk-ant-...   # For Claude equivalence eval
OPENAI_API_KEY=sk-...          # For GPT equivalence eval
GOOGLE_API_KEY=...             # For Gemini equivalence eval
HF_TOKEN=hf_...               # For HuggingFace dataset access
TINKER_API_KEY=tk_...          # For Tinker cloud training/inference
```

---

## Reproduce Evaluation

### Cheap Downstream Eval

For the low-cost downstream workflow used during development, see
[`docs/downstream-eval.md`](docs/downstream-eval.md).

- `scripts/prepare_downstream_eval.py` is the current dataset-build entry point for
  `hotpotqa`, `qasper`, and `ds1000`.
- `scripts/evaluate_downstream.py` now runs paired downstream evaluations for the
  current baselines (`identity`, `truncate`, `extractive`) and explicit learned
  backends (`adapter_local`, `adapter_tinker`), writing per-example results plus
  an aggregate summary.

### 1. Compression Eval (Token Ratio)

Measures how much the adapter compresses input tokens.

**Local MLX backend** (for Nanbeige / local models):

```bash
python scripts/evaluate_tinker.py \
  --backend local \
  --model mlx-community/Nanbeige4.1-3B-8bit \
  --adapter-path models/runs/mlx/2026-03-01_20-53-41--iter-500/adapter \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/eval/ratio_nanbeige_iter500.jsonl \
  --resume
```

**Tinker cloud backend** (for Qwen3-8B):

```bash
python scripts/evaluate_tinker.py \
  --backend tinker \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --checkpoint-path "tinker://161b1f39-3e50-53c0-9d75-f5ce804db7eb:train:0/weights/step-004500" \
  --output models/eval/ratio_qwen3-8b_tinker_step004500.jsonl \
  --show-examples 0 \
  --resume
```

### 2. Convert to Validation Pairs

Transforms compression eval output into the format needed for equivalence testing:

```bash
python - <<'PY'
import json
from pathlib import Path
from src.inference.domain_classifier import DomainClassifier

src = Path("models/eval/ratio_qwen3-8b_tinker_step004500.jsonl")
dst = Path("models/eval/ratio_qwen3-8b_pairs.jsonl")
clf = DomainClassifier()

with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        verbose = row["input_text"]
        compressed = row["generated_output"]
        domain = clf.classify(verbose).value
        fout.write(json.dumps({
            "verbose": verbose,
            "compressed": compressed,
            "domain": domain
        }) + "\n")
PY
```

### 3. Equivalence Eval (Cross-Model Judge)

Tests whether compressed outputs produce equivalent reasoning across Claude, GPT,
and Gemini:

```bash
python scripts/validate_batch.py \
  --input models/eval/pairs_sample300_seed42_qwen_step004500.jsonl \
  --output models/eval/equiv_qwen_step004500_judge.jsonl \
  --models claude gpt gemini \
  --use-llm-judge \
  --save-all \
  --concurrency 3 \
  --resume
```

---

## How It Works

### Pipeline Overview

1. **Raw Corpora** — Source material from code repositories and natural language
   documents
2. **Synthetic Generation** — A trained adapter generates compression pairs from
   raw inputs, bootstrapping the training data
3. **Preprocessing** — Pairs are cleaned and filtered by compression ratio to
   remove degenerate outputs
4. **SFT Training** — LoRA fine-tuning on the filtered pairs teaches the model to
   compress while preserving semantics
5. **Compression Eval** — Token ratio measurement (`output_tokens / input_tokens`)
   across the held-out test set
6. **Equivalence Eval** — The compressed text is fed to Claude, GPT, and Gemini;
   an LLM judge scores whether the outputs are semantically equivalent to outputs
   from the original verbose text

For full data pipeline reproduction commands, see
[`docs/data-pipeline.md`](docs/data-pipeline.md).

### Training

Training is supported on two backends:

- **MLX (local)** — Apple Silicon, quantized models (3B–4B). Uses `mlx-lm` for
  LoRA fine-tuning.
- **Tinker (cloud)** — Remote GPU training for larger models (8B+). Manages
  checkpoints, metrics, and early stopping.

```bash
# Local MLX training
python scripts/train_tinker.py \
  --backend local \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --local-model mlx-community/Qwen3-4B-Instruct-2507-8bit \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/adapters/local_run

# Cloud Tinker training
python scripts/train_tinker.py \
  --backend tinker \
  --model Qwen/Qwen3-8B \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/adapters/tinker
```

---

## Project Structure

```
compression-layer/
├── src/
│   ├── validation/        # Cross-model equivalence testing
│   ├── generation/        # Compression pair generation
│   ├── training/          # Tinker + MLX training pipelines
│   ├── inference/         # Compression inference (local + cloud)
│   ├── evaluation/        # Adapter evaluation logic
│   └── utils/             # Tokenizers, caching, cost tracking
├── scripts/               # CLI entry points
│   ├── train_tinker.py    # Training (local MLX / cloud Tinker)
│   ├── evaluate_tinker.py # Compression ratio evaluation
│   ├── validate_batch.py  # Equivalence evaluation
│   └── ...
├── data/                  # Corpora and datasets (gitignored)
├── models/                # Checkpoints and eval artifacts (gitignored)
├── configs/               # YAML configurations
├── docs/                  # Documentation and plans
├── tests/                 # Test suite
└── assets/                # README images and diagrams
```

---

## Contributing

We welcome contributions, especially in these areas:

- **New domains** — Extending compression beyond code and natural language (e.g.,
  structured data, mathematical notation)
- **Model experiments** — Training adapters on different base models or with
  alternative LoRA configurations
- **Evaluation methodology** — Improving the equivalence scoring system,
  adding new judge models, or refining the pass/fail threshold
- **Dataset expansion** — Contributing to the
  [HuggingFace dataset](https://huggingface.co/datasets/Sudhendra/semantic-compression-sft)
  with new high-quality compression pairs

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup instructions, branch strategy,
and development workflow.

---

## License

[MIT](LICENSE)
