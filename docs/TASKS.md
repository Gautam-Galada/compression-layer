# TASKS.md

Prioritized implementation tasks.

## Phase 1: Foundation ✅ COMPLETE

### P0 — Core Infrastructure
- [x] Create directory structure per AGENTS.md
- [x] Set up pyproject.toml, install deps
- [x] Create `src/utils/config.py` — Pydantic settings
- [x] Create `src/utils/caching.py` — Disk-backed cache
- [x] Create `src/utils/tokenizers.py` — Multi-tokenizer utils
- [x] Create `src/utils/costs.py` — API cost tracking

### P0 — API Clients
- [x] `src/validation/models.py` — ModelType enum, ModelClient
- [x] Implement Anthropic/OpenAI/Gemini async clients
- [x] Add retry logic with exponential backoff

### P0 — Validation Harness
- [x] `src/validation/metrics.py` — Equivalence scoring
- [x] `src/validation/harness.py` — ValidationHarness class

### P1 — Testing
- [x] `tests/test_metrics.py`
- [x] `tests/test_harness.py` (mock API calls)
- [x] `tests/conftest.py` — Pytest fixtures

---

## Phase 2: Generation ✅ COMPLETE

### P0 — Prompts
- [x] `src/generation/prompts/compress_nl.txt` (already exists)
- [x] `src/generation/prompts/compress_code.txt` (already exists)

### P0 — Seed Generator
- [x] `src/generation/seed_generator.py`
- [x] Caching integration

### P1 — CLI Scripts
- [x] `scripts/generate_seed.py`
- [x] `scripts/validate_batch.py`

### P1 — Testing
- [x] `tests/test_seed_generator.py`

---

## Phase 2.5: Corpus Loaders ✅ COMPLETE

### P0 — Code Corpus Loader
- [x] `src/generation/corpus_loader.py` — AST-based Python code extraction
  - [x] Extract functions, classes, methods
  - [x] Quality filters (skip trivial, tests, dunders)
  - [x] Configurable min/max lines and chars
- [x] `scripts/prepare_corpus.py` — CLI for corpus preparation

### P0 — NL Corpus Formatter
- [x] `src/generation/md_formatter.py` — Markdown → JSONL converter
  - [x] Paragraph-based chunking
  - [x] Strip code blocks, frontmatter
  - [x] Configurable min/max chars
- [x] `scripts/format_markdown.py` — CLI for MD formatting

### P1 — Testing
- [x] `tests/test_corpus_loader.py`
- [x] `tests/test_md_formatter.py`

### Sample Size Targets
- **Initial seed**: 200-300 NL + 200-300 code = 400-600 total
- **V1 training**: 1,500-2,000 each domain = 3,000-4,000 total
- **V2 scaling**: 5,000+ each domain = 10,000+ total

---

## Phase 3: Training ✅ COMPLETE

### P0 — Local Setup (MLX)
- [x] Install MLX: `pip install -U mlx-lm`
- [x] Download model: `mlx-community/Qwen3-4B-Instruct-2507-8bit`
- [x] Test inference locally

### P0 — Cloud Setup (Tinker)
- [x] Get Tinker API key from https://tinker.thinkingmachines.ai
- [x] Install: `pip install tinker`
- [x] Verify API key is set in environment or `.env`

### P0 — Training Scripts
- [x] `src/training/train_mlx.py` — Local MLX training wrapper
- [x] `src/training/train_tinker.py` — Tinker cloud training
- [x] Dataset formatting for both platforms

### P1 — MLX Run Storage
- [x] MLX run storage + checkpoint logging (`models/runs/mlx/`)

### P1 — Dataset Builder
- [x] `src/training/dataset.py` — Format for instruction tuning
- [x] JSONL export compatible with MLX and Tinker
- [x] `scripts/format_training_data.py` — CLI for data formatting

### V1 Training Results (MLX)
- **Model**: Qwen3-4B-Instruct-2507-8bit
- **Adapter**: `models/runs/mlx/2026-01-30_17-14-36/adapter` (iter-500)
- **Training Data**: 2,199 examples (1,759 train / 219 valid / 221 test)
- **Pass Rate**: 89% (Claude + GPT evaluators, 0.80 threshold)
- **Avg Compression Ratio**: 62% (38% token savings)
- **Avg Equivalence Score**: 0.859

---

## Phase 4: Inference ✅ COMPLETE

### P0 — Compressor Service
- [x] `src/inference/compressor.py`
  - [x] Load adapter via MLX
  - [x] Single/batch compression
  - [x] Support both local and downloaded Tinker adapters

### P0 — Domain Classifier
- [x] `src/inference/domain_classifier.py` — Regex-based

### P1 — Performance
- [x] Benchmark inference speed on M4 Pro (~1-8s per compression)
- [ ] Optimize for <100ms latency (stretch goal)

---

## Phase 5: Evaluation & Integration ✅ COMPLETE

### P0 — Benchmark Suite
- [x] `src/evaluation/evaluate_adapter.py` — Task-equivalence evaluation
- [x] `scripts/evaluate_adapter.py` — CLI wrapper
- [x] Cross-model validation (Claude + GPT)
- [x] Metrics reporting (compression ratio, equivalence, pass rate)

### Key Findings
- Gemini evaluator produces inconsistent results → removed from evaluation
- 0.80 equivalence threshold is appropriate (21/35 failures were 0.75-0.80)
- Aggressive compression (<50% ratio) correlates with failures

### P1 — Integration
- [ ] Example: Memory store integration
- [ ] A/B test framework

---

## Phase 6: V2 Production Training 🚧 IN PROGRESS

See: `docs/plans/2026-01-31-v2-production-training.md`

### Goals
- Scale from 2K to 10K+ validated training pairs
- Train production model on Tinker (Qwen3-8B)
- Release model on HuggingFace

### Data Generation ✅ COMPLETE
- [x] Generate synthetic code pairs using v1 adapter → **17,315 pairs** (`data/synthetic/code_v2.jsonl`)
- [x] Generate synthetic NL pairs using v1 adapter → **17,056 pairs** (`data/synthetic/nl_v2.jsonl`)
- [x] Total raw synthetic: **34,371 pairs**

### Data Preprocessing & Sanitization ✅ COMPLETE
- [x] Build heuristic preprocessing script (`scripts/preprocess_synthetic.py`)
  - Strips `<think>`/`<tool_call>` generation artifacts
  - Filters by character ratio and token ratio thresholds
  - Writes clean + rejected outputs with rejection reasons
- [x] Preprocess code pairs (max-char-ratio 1.0, max-token-ratio 1.0) → **17,125 passed**, 190 rejected
- [x] Preprocess NL pairs (max-char-ratio 0.95, max-token-ratio 0.95) → **10,505 passed**, 6,551 rejected
- [x] Format into train/valid/test splits (80/10/10, seed 42) via `scripts/format_training_data.py`
- [x] Sanitize all splits via `scripts/data_sanitization.py` (structure, role, encoding checks)
- [x] Validate final data: 0 bad JSON, 0 bad messages, 0 bad roles across all splits

### V2 Data Summary
| Split | Count | Status |
|-------|-------|--------|
| `data/training/train.jsonl` | 19,845 | Sanitized, chat format ✅ |
| `data/training/valid.jsonl` | 2,473 | Sanitized, chat format ✅ |
| `data/training/test.jsonl` | 2,497 | Sanitized, chat format ✅ |
| Unsanitized (rejected by sanitizer) | 2,815 | Archived |

All training files have correct `(system, user, assistant)` message structure.
Prompt masking verified: loss computed only on assistant tokens (`--mask-prompt`).

### MLflow/DagsHub Logging ✅ COMPLETE
- [x] Post-training MLflow logger (`scripts/mlflow_logger.py`)
  - Reads `run.json` + `train.log` from MLX run directories
  - Logs params, step metrics, artifacts, and loss curve plots to DagsHub/MLflow
  - CLI: `--experiment-name`, `--dagshub-owner`, `--dagshub-repo`
- [x] Dependencies installed: `dagshub` (v0.6.5), `mlflow` (v3.9.0)

### Remaining Tasks
- [ ] Train v2 model locally (MLX) on 19,845 examples
- [ ] Train v2 model on Tinker (Qwen3-8B) for production
- [ ] Evaluate v2 and compare with v1
- [ ] Log v2 training run to DagsHub/MLflow
- [ ] Package and release model on HuggingFace

### Cost Notes
- Synthetic generation: done via v1 adapter (free, local)
- Preprocessing/sanitization: heuristic pipeline (free, no API calls)
- Multi-model validation skipped in favor of heuristic filtering (saved ~$40-80)
- Tinker training: ~$10-20 estimated

---

## Stretch Goals

- [ ] Multi-language code support
- [ ] Streaming compression
- [ ] Adaptive compression by task type
- [ ] Sub-100ms inference latency
