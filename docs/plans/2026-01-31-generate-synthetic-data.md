# Generate Synthetic Data Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Generate new synthetic compression pairs from updated raw corpora and validate a cost-aware subset.

**Architecture:** Use `scripts/generate_synthetic.py` to produce JSONL pairs from raw corpora, then use `scripts/validate_synthetic.py` with Claude+GPT models to filter a subset by equivalence threshold. Outputs go to `data/synthetic` and `data/validated`.

**Tech Stack:** Python, rich, MLX adapter, JSONL.

---

### Task 1: Preflight inputs and adapter path

**Files:**
- Create: none
- Modify: none
- Test: none
- Verify: `data/raw/code_merged.jsonl`, `data/raw/nl_merged.jsonl`, `models/runs/mlx/latest/adapter`

**Step 1: Write the failing test**

```python
from pathlib import Path

assert Path("data/raw/does_not_exist.jsonl").exists()
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'
from pathlib import Path

assert Path("data/raw/does_not_exist.jsonl").exists()
PY`

Expected: FAIL with `AssertionError`

**Step 3: Write minimal implementation**

```python
from pathlib import Path

assert Path("data/raw/code_merged.jsonl").exists()
assert Path("data/raw/nl_merged.jsonl").exists()
assert Path("models/runs/mlx/latest/adapter").exists()
```

**Step 4: Run test to verify it passes**

Run: `python - <<'PY'
from pathlib import Path

assert Path("data/raw/code_merged.jsonl").exists()
assert Path("data/raw/nl_merged.jsonl").exists()
assert Path("models/runs/mlx/latest/adapter").exists()
PY`

Expected: PASS (exit 0)

**Step 5: Commit**

Skip commit; no repo changes expected.

### Task 2: Generate code synthetic pairs (v2)

**Files:**
- Create: `data/synthetic/code_v2.jsonl`
- Modify: none
- Test: none

**Step 1: Write the failing test**

```python
from pathlib import Path

path = Path("data/synthetic/code_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/synthetic/code_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: FAIL with `AssertionError`

**Step 3: Write minimal implementation**

Run: `python scripts/generate_synthetic.py --input data/raw/code_merged.jsonl --domain code --output data/synthetic/code_v2.jsonl --limit 5000`

**Step 4: Run test to verify it passes**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/synthetic/code_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: PASS (exit 0)

**Step 5: Commit**

Skip commit; generated data files are not tracked.

### Task 3: Generate NL synthetic pairs (v2)

**Files:**
- Create: `data/synthetic/nl_v2.jsonl`
- Modify: none
- Test: none

**Step 1: Write the failing test**

```python
from pathlib import Path

path = Path("data/synthetic/nl_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/synthetic/nl_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: FAIL with `AssertionError`

**Step 3: Write minimal implementation**

Run: `python scripts/generate_synthetic.py --input data/raw/nl_merged.jsonl --domain nl --output data/synthetic/nl_v2.jsonl --limit 5000`

**Step 4: Run test to verify it passes**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/synthetic/nl_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: PASS (exit 0)

**Step 5: Commit**

Skip commit; generated data files are not tracked.

### Task 4: Validate code subset (cost-aware)

**Files:**
- Create: `data/validated/code_v2.jsonl`
- Modify: none
- Test: none

**Step 1: Write the failing test**

```python
from pathlib import Path

path = Path("data/validated/code_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/validated/code_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: FAIL with `AssertionError`

**Step 3: Write minimal implementation**

Run: `python scripts/validate_synthetic.py --input data/synthetic/code_v2.jsonl --output data/validated/code_v2.jsonl --threshold 0.80 --concurrency 4 --models claude gpt --limit 1000`

**Step 4: Run test to verify it passes**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/validated/code_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: PASS (exit 0)

**Step 5: Commit**

Skip commit; generated data files are not tracked.

### Task 5: Validate NL subset (cost-aware)

**Files:**
- Create: `data/validated/nl_v2.jsonl`
- Modify: none
- Test: none

**Step 1: Write the failing test**

```python
from pathlib import Path

path = Path("data/validated/nl_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
```

**Step 2: Run test to verify it fails**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/validated/nl_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: FAIL with `AssertionError`

**Step 3: Write minimal implementation**

Run: `python scripts/validate_synthetic.py --input data/synthetic/nl_v2.jsonl --output data/validated/nl_v2.jsonl --threshold 0.80 --concurrency 4 --models claude gpt --limit 1000`

**Step 4: Run test to verify it passes**

Run: `python - <<'PY'
from pathlib import Path

path = Path("data/validated/nl_v2.jsonl")
assert path.exists()
assert path.stat().st_size > 0
PY`

Expected: PASS (exit 0)

**Step 5: Commit**

Skip commit; generated data files are not tracked.
