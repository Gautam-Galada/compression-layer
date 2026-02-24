# Local Tinker Parity + HF Loader Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Run the existing Tinker training + evaluation workflow locally on Mac with parity behavior, while sourcing data from Hugging Face dataset `Sudhendra/semantic-compression-sft` instead of relying on pre-existing local JSONL files.

**Architecture:** Keep the current Tinker entry points (`scripts/train_tinker.py`, `scripts/evaluate_tinker.py`) and add a backend switch that preserves current Tinker behavior while enabling a local MLX backend. Introduce one shared HF loader that materializes `train/valid/test` chat JSONL files in the same schema currently consumed by training/eval.

**Tech Stack:** Python, `datasets` (HF), MLX (`mlx_lm`), existing Tinker training/eval modules, pytest.

---

### Task 1: Add HF dataset loader (single source for train/valid/test materialization)

**Files:**
- Create: `src/training/hf_dataset_loader.py`
- Modify: `src/training/__init__.py`
- Test: `tests/test_hf_dataset_loader.py`

**Step 1: Write the failing tests**

Create `tests/test_hf_dataset_loader.py` with tests for:

```python
def test_materialize_hf_dataset_writes_train_valid_test(tmp_path):
    ...

def test_materialize_hf_dataset_maps_validation_to_valid(tmp_path):
    ...

def test_materialize_hf_dataset_rejects_missing_messages_field(tmp_path):
    ...
```

Use `monkeypatch` to stub `datasets.load_dataset` (no network in tests).

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_hf_dataset_loader.py -v`
Expected: FAIL (`ModuleNotFoundError` or missing function).

**Step 3: Write minimal implementation**

Create `src/training/hf_dataset_loader.py` with:

```python
def materialize_hf_chat_dataset(
    dataset_name: str,
    output_dir: Path,
    *,
    split_map: dict[str, str] | None = None,
    force: bool = True,
) -> dict[str, int]:
    ...
```

Rules:
- Load HF dataset once (`load_dataset(dataset_name)`).
- Expect source splits: `train`, `validation`, `test`.
- Write destination files: `train.jsonl`, `valid.jsonl`, `test.jsonl`.
- Preserve current schema exactly: one JSON object per line with top-level `messages`.
- Validate `messages` is present and list-typed; fail fast on malformed rows.
- Return counts per split for CLI display.

**Step 4: Export loader through training package**

In `src/training/__init__.py`, add:

```python
from .hf_dataset_loader import materialize_hf_chat_dataset
```

and include it in `__all__`.

**Step 5: Run test to verify pass**

Run: `pytest tests/test_hf_dataset_loader.py -v`
Expected: PASS.

**Step 6: Commit**

```bash
git add src/training/hf_dataset_loader.py src/training/__init__.py tests/test_hf_dataset_loader.py
git commit -m "feat: add HF dataset loader for chat split materialization"
```

---

### Task 2: Wire HF loader into Tinker training CLI (replace direct dependency on existing local JSONL)

**Files:**
- Modify: `scripts/train_tinker.py`
- Test: `tests/test_train_tinker_cli.py`

**Step 1: Write the failing CLI tests**

Create `tests/test_train_tinker_cli.py` with tests for:

```python
def test_train_cli_materializes_hf_dataset_before_training(monkeypatch, tmp_path):
    ...

def test_train_cli_uses_hf_default_dataset_name(monkeypatch, tmp_path):
    ...
```

Assert loader is called with `Sudhendra/semantic-compression-sft`, and the resulting directory is used as `config.dataset_path`.

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_tinker_cli.py -v`
Expected: FAIL (loader not invoked yet).

**Step 3: Implement minimal CLI wiring**

In `scripts/train_tinker.py`:
- Add argument:

```python
parser.add_argument(
    "--hf-dataset",
    type=str,
    default="Sudhendra/semantic-compression-sft",
    help="Hugging Face dataset to materialize for training/eval splits",
)
```

- Before counting examples/training start, call:

```python
counts = materialize_hf_chat_dataset(args.hf_dataset, config.dataset_path)
```

- Keep existing behavior otherwise (same training flow, same output structure, same cost estimate logic).

**Step 4: Run tests to verify pass**

Run: `pytest tests/test_train_tinker_cli.py -v`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/train_tinker.py tests/test_train_tinker_cli.py
git commit -m "feat: source train splits from HF dataset in train_tinker CLI"
```

---

### Task 3: Add local backend to Tinker training module with artifact parity

**Files:**
- Modify: `src/training/train_tinker.py`
- Modify: `scripts/train_tinker.py`
- Test: `tests/test_train_tinker.py`

**Step 1: Write failing tests for local backend dispatch + artifacts**

Extend `tests/test_train_tinker.py` with:

```python
def test_train_on_tinker_local_backend_dispatches_to_mlx(tmp_path, monkeypatch):
    ...

def test_local_backend_writes_tinker_run_and_metrics_files(tmp_path, monkeypatch):
    ...

def test_local_backend_uses_resume_adapter_file_when_enabled(tmp_path, monkeypatch):
    ...
```

Mock `subprocess.run` so tests do not execute real MLX training.

**Step 2: Run targeted tests to verify fail**

Run: `pytest tests/test_train_tinker.py -k "local_backend" -v`
Expected: FAIL.

**Step 3: Add backend selection to config**

In `TinkerTrainingConfig`, add:

```python
backend: str = "tinker"  # "tinker" | "local"
local_model: str | None = None
```

In `scripts/train_tinker.py`, add:

```python
parser.add_argument("--backend", choices=["tinker", "local"], default="tinker")
parser.add_argument("--local-model", type=str, default=None)
```

Assign into config.

**Step 4: Implement local training backend (minimal parity, no new behavior)**

In `src/training/train_tinker.py`, add private function:

```python
def _train_with_local_mlx_backend(config: TinkerTrainingConfig) -> TinkerTrainingResult:
    ...
```

Implementation requirements:
- Use existing `config.dataset_path` (`train.jsonl`, `valid.jsonl`, `test.jsonl`).
- Run local MLX LoRA subprocess (`python -m mlx_lm.lora ...`) with mapped arguments from current Tinker config (epochs/batch/lr/log interval/checkpoint interval).
- Write Tinker-compatible artifacts to `config.output_dir`:
  - `tinker_run.json`
  - `run.json`
  - `train.log`
  - `metrics.jsonl`
- Preserve resume semantics by reading existing `tinker_run.json` and using MLX resume option when available.
- Return `TinkerTrainingResult` in the same shape as Tinker backend.

**Step 5: Dispatch by backend without changing existing cloud flow**

In `train_on_tinker(...)`:

```python
if config.backend == "local":
    return _train_with_local_mlx_backend(config)
```

Keep all existing Tinker client behavior unchanged when backend is `tinker`.

**Step 6: Run targeted + existing training tests**

Run: `pytest tests/test_train_tinker.py -v`
Expected: PASS (including prior service-client tests).

**Step 7: Commit**

```bash
git add src/training/train_tinker.py scripts/train_tinker.py tests/test_train_tinker.py
git commit -m "feat: enable local MLX backend for tinker training flow"
```

---

### Task 4: Add local backend to Tinker evaluation script (keep current decoding/resume logic)

**Files:**
- Modify: `scripts/evaluate_tinker.py`
- Test: `tests/test_evaluate_tinker.py`

**Step 1: Write failing tests for local generator + backend selection**

Add tests:

```python
def test_local_backend_uses_mlx_generator(monkeypatch):
    ...

def test_eval_uses_hf_loader_for_test_split(monkeypatch, tmp_path):
    ...
```

Mock `mlx_lm.load`, `mlx_lm.generate`, and HF loader.

**Step 2: Run targeted tests to verify fail**

Run: `pytest tests/test_evaluate_tinker.py -k "local_backend or hf_loader" -v`
Expected: FAIL.

**Step 3: Implement local generator path**

In `scripts/evaluate_tinker.py`:
- Add args:

```python
parser.add_argument("--backend", choices=["tinker", "local"], default="tinker")
parser.add_argument("--model", type=str, default="mlx-community/Qwen3-4B-Instruct-2507-8bit")
parser.add_argument("--hf-dataset", type=str, default="Sudhendra/semantic-compression-sft")
```

- Add `create_local_generator(...)` using MLX tokenizer chat template + `mlx_lm.generate`.
- Reuse existing decoding hardening functions exactly (`_build_user_message`, `_compute_generation_budget`, `_strip_generation_artifacts`, `_truncate_repetition`, `_cap_output_length`).

**Step 4: Keep resume/output behavior unchanged**

Do not change:
- `_load_existing_results`
- `--resume` behavior
- incremental JSONL append
- summary/sample output flow

Only switch the generator implementation based on `--backend`.

**Step 5: Source eval split from HF loader**

Before loading examples, materialize HF dataset into a local working dir and point evaluation to generated `test.jsonl`.

**Step 6: Run all eval tests**

Run: `pytest tests/test_evaluate_tinker.py -v`
Expected: PASS.

**Step 7: Commit**

```bash
git add scripts/evaluate_tinker.py tests/test_evaluate_tinker.py
git commit -m "feat: enable local backend for tinker evaluation with HF dataset input"
```

---

### Task 5: Config/docs parity + smoke verification commands

**Files:**
- Modify: `configs/training.yaml`
- Modify: `docs/SETUP.md`
- Modify: `docs/MLX_TRAINING.md`

**Step 1: Add explicit defaults for parity execution**

In `configs/training.yaml`, add minimal fields under `cloud` (used by `train_tinker.py` only):

```yaml
cloud:
  backend: "tinker"  # switch to "local" to run same flow on Mac
  hf_dataset: "Sudhendra/semantic-compression-sft"
```

Keep all existing hyperparameters untouched.

**Step 2: Document exact local-parity commands**

In docs, add:

```bash
# Train using the tinker workflow locally on Mac
python scripts/train_tinker.py \
  --backend local \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/adapters/tinker

# Evaluate using the tinker evaluation workflow locally on Mac
python scripts/evaluate_tinker.py \
  --backend local \
  --hf-dataset Sudhendra/semantic-compression-sft \
  --output models/eval/tinker_eval_local.jsonl
```

**Step 3: Run focused verification**

Run:

```bash
pytest tests/test_hf_dataset_loader.py tests/test_train_tinker.py tests/test_evaluate_tinker.py -v
```

Expected: PASS.

**Step 4: Run smoke commands (short)**

Run:

```bash
python scripts/train_tinker.py --backend local --hf-dataset Sudhendra/semantic-compression-sft --epochs 1 --batch-size 1 --output models/adapters/tinker_local_smoke
python scripts/evaluate_tinker.py --backend local --hf-dataset Sudhendra/semantic-compression-sft --limit 10 --output models/eval/tinker_eval_local_smoke.jsonl
```

Expected:
- training writes `tinker_run.json`, `run.json`, `train.log`, `metrics.jsonl` in output dir
- eval writes 10 JSONL rows and prints summary

**Step 5: Commit**

```bash
git add configs/training.yaml docs/SETUP.md docs/MLX_TRAINING.md
git commit -m "docs: add local parity workflow for tinker training and eval"
```

---

### Task 6: Final regression pass (no behavior drift)

**Files:**
- Verify only (no required file edits)

**Step 1: Run key regression suites**

Run:

```bash
pytest tests/test_train_tinker.py tests/test_evaluate_tinker.py tests/test_train_mlx_runs.py tests/test_train_mlx_eval.py -v
```

Expected: PASS.

**Step 2: Sanity-check cloud path still intact**

Run:

```bash
python scripts/train_tinker.py --help
python scripts/evaluate_tinker.py --help
```

Expected: existing cloud options still present; new options limited to `--backend`, `--local-model` (train), and `--hf-dataset`.

**Step 3: Commit test-only fixes if needed**

```bash
git add tests/
git commit -m "test: cover local tinker parity and HF data loader paths"
```
