# LLM Compression Layer

Universal semantic compression layer for LLM inputs. Compresses memories, code,
and context before API calls while preserving reasoning equivalence across
Claude, GPT, and Gemini.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest tests/ -v
```

## Environment Variables

Create a `.env` file with:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
HF_TOKEN=hf_...
TINKER_API_KEY=tk_...
```

## Ground-Truth Run Snapshot (from local artifacts)

These metrics are taken directly from run/eval artifacts in this repo (not from
memory):

- `models/runs/mlx/latest/run.json`
- `models/runs/mlx/latest/train.log`
- `models/adapters/tinker/tinker_run.json`
- `models/eval/tinker_eval_tightened_full.jsonl`
- `models/eval/tinker_equiv_tightened_full_llm_judge.jsonl`

### Local 4B run (MLX)

From `models/runs/mlx/latest/run.json` and `models/runs/mlx/latest/train.log`:

| Metric | Value |
| --- | --- |
| Model | `mlx-community/Qwen3-4B-Instruct-2507-8bit` |
| LoRA | rank `8`, alpha `16`, layers `16` |
| Train config | `500` iters, batch size `4`, lr `1e-4` |
| Final val loss (iter 500) | `0.510` |

### Cloud 8B run (Tinker)

From `models/adapters/tinker/tinker_run.json`:

| Metric | Value |
| --- | --- |
| Run ID | `161b1f39-3e50-53c0-9d75-f5ce804db7eb:train:0` |
| Model | `Qwen/Qwen3-8B` |
| Train examples | `19,845` |
| LoRA | rank `16`, alpha `32`, dropout `0.05` |
| Train config | epochs `2`, batch size `4`, lr `2e-4` |
| Status | completed, early stopped at step `4962` (planned total `9924`) |
| Best val loss | `0.2657` |
| Final loss | `0.2845` |
| Latest checkpoint | `tinker://.../weights/final` |

### 8B evaluation results

Source files:

- Compression: `models/eval/tinker_eval_tightened_full.jsonl`
- Equivalence: `models/eval/tinker_equiv_tightened_full_llm_judge.jsonl`

### Compression Results (8B, full test set)

| Run | Split Size | Evaluated | Avg Compression Ratio | Avg Token Reduction | Median Compression Ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tinker Qwen3-8B tightened eval | 2,497 | 2,497 | 0.5207 | 47.93% | 0.5847 |

### Equivalence Results (8B, current progress)

| Run | Threshold | Judge | Models | Evaluated | Coverage | Pass Rate | Avg Min-Equiv | Median Min-Equiv | Min Min-Equiv |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tinker Qwen3-8B tightened equiv | 0.80 | On (`--use-llm-judge`) | Claude Sonnet + GPT-4o-mini + Gemini 2.0 Flash | 1,753 | 70.20% (1,753/2,497) | 21.11% | 0.6791 | 0.7102 | 0.1538 |

### Equivalence Model Breakdown

| Model | Combined Avg | Combined Min | Judge-Only Avg | Judge-Only Min |
| --- | ---: | ---: | ---: | ---: |
| `claude-sonnet-4-20250514` | 0.7868 | 0.1842 | 0.7784 | 0.2500 |
| `gpt-4o-mini` | 0.7911 | 0.1538 | 0.7777 | 0.2500 |
| `gemini-2.0-flash` | 0.7092 | 0.1682 | 0.6968 | 0.2562 |

## Local Tinker-Style 4B Training (MLX backend)

This uses the same `train_tinker.py` workflow but runs locally on MLX.

```bash
source .venv/bin/activate

python scripts/train_tinker.py \
  --backend local \
  --model Qwen/Qwen3-4B-Instruct-2507 \
  --local-model mlx-community/Qwen3-4B-Instruct-2507-8bit \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/adapters/tinker_local_4b
```

Evaluate that local run with the same eval workflow:

```bash
python scripts/evaluate_tinker.py \
  --backend local \
  --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
  --adapter-path models/adapters/tinker_local_4b \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/eval/tinker4b_local_eval.jsonl
```

## Data Bootstrapping: Generate -> Clean -> Format -> Sanitize

### 1) Generate synthetic pairs with local adapter

```bash
python scripts/generate_synthetic.py \
  --input data/raw/code.jsonl \
  --domain code \
  --adapter models/runs/mlx/latest/adapter \
  --output data/synthetic/code_v2.jsonl \
  --resume

python scripts/generate_synthetic.py \
  --input data/raw/nl_docs.jsonl \
  --domain nl \
  --adapter models/runs/mlx/latest/adapter \
  --output data/synthetic/nl_v2.jsonl \
  --resume
```

### 2) Clean and ratio-filter synthetic outputs

```bash
python scripts/preprocess_synthetic.py \
  --input data/synthetic/code_v2.jsonl \
  --output data/validated/code_pairs.jsonl \
  --rejected data/validated/rejected_code_pairs.jsonl \
  --max-char-ratio 1.0 \
  --max-token-ratio 1.0

python scripts/preprocess_synthetic.py \
  --input data/synthetic/nl_v2.jsonl \
  --output data/validated/nl_pairs.jsonl \
  --rejected data/validated/rejected_nl_pairs.jsonl \
  --max-char-ratio 0.95 \
  --max-token-ratio 0.95
```

### 3) Format validated pairs into train/valid/test

```bash
python scripts/format_training_data.py \
  --input data/validated \
  --output data/training \
  --train-ratio 0.8 \
  --valid-ratio 0.1 \
  --test-ratio 0.1 \
  --seed 42
```

### 4) Sanitize each split

```bash
for split in train valid test; do
  python scripts/data_sanitization.py \
    --input "data/training/${split}.jsonl" \
    --sanitized "data/training/sanitized_${split}.jsonl" \
    --unsanitized "data/training/unsanitized_${split}.jsonl"
done

for split in train valid test; do
  mv "data/training/sanitized_${split}.jsonl" "data/training/${split}.jsonl"
done
```

### Current data counts in workspace

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

## Reproduce the 8B Evaluation Pipeline

### 1) Run generation eval against Tinker checkpoint

`evaluate_tinker.py` will auto-read checkpoint from
`models/adapters/tinker/tinker_run.json` if `--checkpoint-path` is omitted.

```bash
python scripts/evaluate_tinker.py \
  --backend tinker \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/eval/tinker_eval_tightened_full.jsonl \
  --show-examples 0 \
  --resume
```

### 2) Convert generation output to validation pairs

```bash
python - <<'PY'
import json
from pathlib import Path
from src.inference.domain_classifier import DomainClassifier

src = Path("models/eval/tinker_eval_tightened_full.jsonl")
dst = Path("models/eval/tinker_eval_tightened_full_pairs.jsonl")
clf = DomainClassifier()

with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        if not line.strip():
            continue
        row = json.loads(line)
        verbose = row["input_text"]
        compressed = row["generated_output"]
        domain = clf.classify(verbose).value
        fout.write(json.dumps({"verbose": verbose, "compressed": compressed, "domain": domain}) + "\n")

print(dst)
PY
```

### 3) Run equivalence eval (with LLM judge)

```bash
python scripts/validate_batch.py \
  --input models/eval/tinker_eval_tightened_full_pairs.jsonl \
  --output models/eval/tinker_equiv_tightened_full_llm_judge.jsonl \
  --models claude gpt gemini \
  --threshold 0.80 \
  --use-llm-judge \
  --save-all \
  --concurrency 2 \
  --resume
```

## Project Structure

```
compression-layer/
├── src/
│   ├── validation/     # Cross-model equivalence testing
│   ├── generation/     # Compression pair generation
│   ├── training/       # Tinker + MLX training
│   ├── inference/      # Compression inference paths
│   └── utils/          # Tokenizers, caching, cost tracking
├── data/               # Corpora and generated datasets (gitignored)
├── models/             # Checkpoints and eval artifacts (gitignored)
├── configs/            # YAML configs
├── scripts/            # Entry points
└── tests/
```

## License

MIT
