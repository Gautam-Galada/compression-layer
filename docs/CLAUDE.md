# CLAUDE.md

## Project Summary

Universal semantic compression layer for LLM inputs. Compresses memories, code, context before API calls while preserving reasoning equivalence across Claude/GPT/Gemini.

**Model**: Qwen3-4B-Instruct-2507-8bit (local MLX) / Qwen3-8B (Tinker cloud)  
**Training**: MLX LoRA (local) + Tinker (cloud production)  
**License**: Apache 2.0  
**Repo**: https://github.com/Sudhendra/compression-layer

## Build & Run

```bash
# Clone and setup
git clone https://github.com/Sudhendra/compression-layer.git
cd compression-layer
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Copy env and add API keys
cp .env.example .env

# Local (MLX on M4 Pro)
pip install -U mlx-lm

# Cloud (Tinker)
pip install tinker

# Run all checks before committing
ruff check src/ tests/ && mypy src/ && pytest tests/ -v
```

## Git Workflow

### Branch Strategy
```
main (protected)
  └── phase-1-foundation    ✅ Complete
  └── phase-2-generation    
  └── phase-3-training      
  └── phase-4-inference     
  └── phase-5-evaluation    
```

### Development Cycle
```bash
# 1. Create phase branch from main
git checkout main && git pull
git checkout -b phase-2-generation

# 2. Implement, commit often
git add . && git commit -m "feat: add seed generator"

# 3. Verify CI passes locally before pushing
ruff check src/ tests/
mypy src/ --ignore-missing-imports
pytest tests/ -v

# 4. Push and create PR
git push -u origin phase-2-generation
gh pr create --title "Phase 2: Generation" --body "..."

# 5. Wait for CI, then merge
gh pr merge --squash
```

### CI Requirements (must pass before merge)
- **Lint**: `ruff check src/ tests/`
- **Format**: `ruff format --check src/ tests/`
- **Types**: `mypy src/ --ignore-missing-imports`
- **Tests**: `pytest tests/ -v`

## Directory Purpose

- `src/validation/` — Equivalence testing across models
- `src/generation/` — Compression pair synthesis
- `src/training/` — MLX local + Tinker cloud fine-tuning
- `src/inference/` — Production compressor
- `data/synthetic/` — Raw synthetic pairs from v1 adapter (gitignored)
- `data/validated/` — Preprocessed and filtered pairs (gitignored)
- `data/training/` — Final train/valid/test splits in chat format (gitignored)
- `models/` — LoRA checkpoints, run logs, GGUF exports (gitignored)

## Code Conventions

- Async by default for I/O
- Type hints required
- Pydantic models for structured data
- `rich` for CLI output

## Critical Implementation Notes

### Training
- **Local (MLX)**: Quick iteration with Qwen3-4B on M4 Pro
- **Cloud (Tinker)**: Production runs with Qwen3-8B (~$10-50 total)
- Export adapters, run inference locally

### Inference (`src/inference/compressor.py`)
- Call `FastLanguageModel.for_inference(model)` — 2x faster
- Domain classifier is regex-based (no ML overhead)
- Use GGUF export + llama.cpp for lowest latency

### Validation (`src/validation/harness.py`)
- Compression passes only if min(equivalence) >= 0.72 across ALL models
- NL: Pure semantic similarity (cosine embeddings) - lexical removed due to symbol penalty
- Code: 0.5 × AST similarity + 0.5 × semantic
- Temperature=0.0 for deterministic validation
- Optional LLM-as-judge for higher accuracy on borderline cases

## API Keys Required

```bash
ANTHROPIC_API_KEY=
OPENAI_API_KEY=
GOOGLE_API_KEY=
HF_TOKEN=           # For model downloads
TINKER_API_KEY=     # For cloud training
DAGSHUB_OWNER=      # For MLflow logging (default: Sudhendra)
DAGSHUB_REPO=       # For MLflow logging (default: compression-layer)
```

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/train_local.py` | MLX LoRA training with run storage |
| `scripts/preprocess_synthetic.py` | Heuristic filtering of synthetic pairs |
| `scripts/format_training_data.py` | Format validated pairs into train/valid/test splits |
| `scripts/data_sanitization.py` | Structure/role/encoding checks on training data |
| `scripts/mlflow_logger.py` | Post-training MLflow/DagsHub logging |
| `scripts/evaluate_adapter.py` | Cross-model equivalence evaluation |

## Common Tasks

**Change base model**: Update `configs/training.yaml` → `cloud.model`

**Add compression domain**: 
1. Add prompt in `src/generation/prompts/`
2. Update `DomainClassifier` patterns
3. Add domain-specific equivalence in `metrics.py`

**Export for production**:
```python
model.save_pretrained_gguf("models/gguf", tokenizer, quantization_method="q4_k_m")
```

## Gotchas

- MLX requires macOS 15+ for best performance
- Use 4-bit models from `mlx-community/` for local work
- AST parsing fails on incomplete code — wrap in try/except
- Tinker dataset format: JSONL with `text` or `messages` field
- **Always run CI checks locally before pushing**: `ruff check && mypy src/ && pytest`
- **Never commit `.env`** — it's gitignored for security
