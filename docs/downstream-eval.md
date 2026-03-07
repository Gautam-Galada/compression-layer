# Cheap Downstream Evaluation

Use this workflow to sanity-check downstream behavior before spending money on the
full cross-model equivalence stack.

Current state in this branch:

- `scripts/prepare_downstream_eval.py` is implemented and can build JSONL datasets
  for `hotpotqa`, `qasper`, and `ds1000`.
- `scripts/evaluate_downstream.py` runs paired full-context vs compressed-context
  evaluations and writes per-example JSONL rows plus an aggregate summary.
- Supported compressors are the deterministic baselines `identity`, `truncate`,
  and `extractive`, plus explicit learned backends `adapter_local` and
  `adapter_tinker`.

## 1. Build small dev datasets

These commands are current supported dataset-build commands.

```bash
python scripts/prepare_downstream_eval.py --benchmark hotpotqa --split validation --limit 100 --output data/eval/downstream/hotpotqa_validation_100.jsonl
python scripts/prepare_downstream_eval.py --benchmark qasper --split validation --limit 50 --max-chunks 6 --output data/eval/downstream/qasper_validation_50.jsonl
python scripts/prepare_downstream_eval.py --benchmark ds1000 --split test --limit 25 --output data/eval/downstream/ds1000_test_25.jsonl
```

Notes:

- `hotpotqa` uses the distractor subset.
- `qasper` keeps evidence paragraphs and then adds up to `--max-chunks` lexical
  distractors.
- `ds1000` writes code-generation examples with the prompt as the query and the
  provided code context as retrieved context.

## 2. Cheap baseline runs

These are current supported baseline commands.

```bash
python scripts/evaluate_downstream.py --dataset data/eval/downstream/hotpotqa_validation_100.jsonl --compressor identity --task-model gpt-4o-mini --output models/eval/downstream_hotpotqa_identity_gpt4omini.jsonl --resume
python scripts/evaluate_downstream.py --dataset data/eval/downstream/hotpotqa_validation_100.jsonl --compressor truncate --task-model gpt-4o-mini --output models/eval/downstream_hotpotqa_truncate_gpt4omini.jsonl --resume
python scripts/evaluate_downstream.py --dataset data/eval/downstream/hotpotqa_validation_100.jsonl --compressor extractive --task-model gpt-4o-mini --output models/eval/downstream_hotpotqa_extractive_gpt4omini.jsonl --resume
```

Development rule: do not claim a compression win unless you also compare against
`identity`, `truncate`, and `extractive`.

## 3. Learned compressor runs

Use explicit backend flags for learned compressors.

```bash
python scripts/evaluate_downstream.py --dataset data/eval/downstream/hotpotqa_validation_100.jsonl --compressor adapter_local --adapter-model mlx-community/Nanbeige4.1-3B-8bit --adapter-path models/runs/mlx/.../adapter --task-model gpt-4o-mini --output models/eval/downstream_hotpotqa_adapterlocal_gpt4omini.jsonl --resume
python scripts/evaluate_downstream.py --dataset data/eval/downstream/hotpotqa_validation_100.jsonl --compressor adapter_tinker --checkpoint-path "tinker://.../weights/step-004500" --task-model gpt-4o-mini --output models/eval/downstream_hotpotqa_adaptertinker_gpt4omini.jsonl --resume
```

Notes:

- `adapter_local` requires both `--adapter-model` and `--adapter-path`.
- `adapter_tinker` requires `--checkpoint-path` and `TINKER_API_KEY`.

## 4. Required reporting metrics

Current output rows include these fields:

- `full_exact_match`
- `compressed_exact_match`
- `delta_exact_match`
- `full_f1`
- `compressed_f1`
- `delta_f1`
- `context_tokens_full`
- `context_tokens_compressed`
- `compression_ratio`
- `latency_ms_full`
- `latency_ms_compressed`
- `cost_usd_full`
- `cost_usd_compressed`

Implementation notes:

- The CLI writes one JSONL result row per evaluated example and a separate summary
  JSON file with aggregated metrics.
- `qasper` dataset prep still depends on the local dataset loader working. If the
  installed `datasets` package cannot load `allenai/qasper`, dataset preparation
  remains blocked until that loader issue is fixed or the dataset is cached in a
  compatible form.

## 5. Budget rules

- Do not use LLM judges in the primary loop.
- Do not run all three frontier models during development.
- Do not evaluate more than 175 dev examples before the baseline curves look sensible.
- Always compare against `identity`, `truncate`, and `extractive` before claiming progress.

Practical default:

- Start with one small dataset build, one cheap task model, and deterministic
  baselines first.
- Only scale to larger sample counts or more expensive models after the baseline
  ordering looks reasonable.
