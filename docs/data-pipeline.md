# Data Pipeline

Full commands for the synthetic data bootstrapping pipeline. For an overview of
the pipeline stages, see the main [README](../README.md).

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Ensure you have a `.env` file with your API keys (see `.env.example`).

## 1. Generate Synthetic Pairs

Use a trained local adapter to generate compression pairs from raw corpora:

```bash
# Code domain
python scripts/generate_synthetic.py \
  --input data/raw/code.jsonl \
  --domain code \
  --adapter models/runs/mlx/latest/adapter \
  --output data/synthetic/code_v2.jsonl \
  --resume

# Natural language domain
python scripts/generate_synthetic.py \
  --input data/raw/nl_docs.jsonl \
  --domain nl \
  --adapter models/runs/mlx/latest/adapter \
  --output data/synthetic/nl_v2.jsonl \
  --resume
```

## 2. Clean and Ratio-Filter

Remove low-quality pairs and enforce compression ratio constraints:

```bash
# Code pairs (stricter filtering)
python scripts/preprocess_synthetic.py \
  --input data/synthetic/code_v2.jsonl \
  --output data/validated/code_pairs.jsonl \
  --rejected data/validated/rejected_code_pairs.jsonl \
  --max-char-ratio 1.0 \
  --max-token-ratio 1.0

# NL pairs
python scripts/preprocess_synthetic.py \
  --input data/synthetic/nl_v2.jsonl \
  --output data/validated/nl_pairs.jsonl \
  --rejected data/validated/rejected_nl_pairs.jsonl \
  --max-char-ratio 0.95 \
  --max-token-ratio 0.95
```

## 3. Format into Train/Valid/Test Splits

```bash
python scripts/format_training_data.py \
  --input data/validated \
  --output data/training \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

## 4. Sanitize Each Split

Remove any remaining artifacts or formatting issues:

```bash
for split in train valid test; do
  python scripts/data_sanitization.py \
    --input "data/training/${split}.jsonl" \
    --sanitized "data/training/sanitized_${split}.jsonl" \
    --unsanitized "data/training/unsanitized_${split}.jsonl"
done

# Replace originals with sanitized versions
for split in train valid test; do
  mv "data/training/sanitized_${split}.jsonl" "data/training/${split}.jsonl"
done
```

## Current Dataset Statistics

| File | Rows |
| --- | ---: |
| `data/raw/code.jsonl` | 1,571 |
| `data/raw/nl_docs.jsonl` | 1,234 |
| `data/synthetic/code_v2.jsonl` | 17,315 |
| `data/synthetic/nl_v2.jsonl` | 17,056 |
| `data/validated/code_pairs.jsonl` | 17,125 |
| `data/validated/nl_pairs.jsonl` | 10,505 |
| `data/training/train.jsonl` | 19,845 |
| `data/training/valid.jsonl` | 2,473 |
| `data/training/test.jsonl` | 2,497 |

The final training dataset is also published on HuggingFace:
[`Sudhendra/semantic-compression-sft`](https://huggingface.co/datasets/Sudhendra/semantic-compression-sft)
