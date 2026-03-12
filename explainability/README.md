# Project Investigation Roadmap: Post-Hoc Analysis & Strategy

> **Current status:** Investigation 1 complete. Investigation 2 active.

---

## Investigation 1: Dataset Cartography — COMPLETE

Dataset cartography (Swayamdipta et al., EMNLP 2020) tracks per-sample confidence and variability across training checkpoints, mapping the dataset into three diagnostic regions. We ran this on the NL subset of our training data: 300 samples × 10 checkpoints (iters 50–500), Qwen3-4B LoRA run.

### Regions

| Region | Definition | Count | Avg Confidence | Avg Correctness |
|--------|-----------|-------|---------------|-----------------|
| Easy | High confidence, low variability | 119 | 0.739 | 0.924 |
| Hard | Low confidence, low variability | 31 | 0.569 | 0.177 |
| Ambiguous | High variability | 150 | 0.551 | 0.301 |

### Findings

**1. NL and code are different learning problems.**
Code mean loss reached 0.238 by step 500; NL is at 0.503. Running them in a single training pass means iteration budget, evaluation thresholds, and learning dynamics are calibrated to code — making NL appear to be failing when it is simply undertrained on a harder task.

**2. Training stopped too early for NL.**
120 samples (Ambiguous-improving + Hard-improving) show declining loss at step 500 with no plateau. The Ambiguous group alone dropped from mean loss 0.932 → 0.508 across the run. NL requires 1000–1500 iterations. The Easy group (111 samples, correctness 0.924, smooth monotonic decline) serves as an internal control confirming model capacity is not the bottleneck.

**3. 67 oscillating samples have corrupted synthetic targets.**
Oscillation is defined as max_rise > 0.10 across consecutive checkpoints. These samples caused the model to learn a compression at one checkpoint, then be penalised for it at the next — only possible if the bootstrapper produced contradictory targets for the same input. Two distinct instability events were identified:

- **Iter 300:** 46 samples spiked (mean delta across population: −0.002)
- **Iter 450:** 64 samples spiked (mean delta: **+0.012** — the only step where population mean loss increased)
- Only 22 samples appear in both events

The non-overlap between events rules out a learning rate schedule explanation. The pattern is consistent with label noise in specific input types.

**4. Correctness is bimodal — not a spectrum.**
74 samples score correctness = 0.0 (above median loss at all 10 checkpoints). 110 score ≥ 0.9. Only 24 fall in the 0.1–0.4 range. These are two structurally separate populations from step 50 onward, not a continuum.

### Action items from Investigation 1

| Action | Scope | Script |
|--------|-------|--------|
| Re-generate targets | 67 oscillating samples (List A) | `regen_oscillating.py` |
| Retrain NL adapter | 1500 iters, clean data | `scripts/train_local.py` |
| Re-run cartography | Verify instability events resolved | `compute_cartography.py` |
| Re-run HotPotQA eval | Measure ΔF1 improvement | `docs/downstream-eval.md` |

> The current 1–2% intrinsic equivalence pass rate is a training duration and data quality artifact, not a model capacity ceiling.

---

## Investigation 2: Bootstrapper Prompt Analysis — ACTIVE

### Motivation

The 67 oscillating samples prove the bootstrapper was inconsistent for certain input types. Re-generating those samples without understanding *why* they failed will reproduce the same errors. This investigation audits the prompts that generated the synthetic compression pairs, with the oscillating samples as the primary evidence.

### Scope

- Load the 67 oscillating inputs (List A, `oscillating_indices.json`) from `train.jsonl`
- Identify structural patterns that caused bootstrapper inconsistency:
  - Multi-question inputs (ambiguous compression scope)
  - Inputs requiring world knowledge the bootstrapper may not have had
  - Inputs with no clear compression target (e.g. single-sentence opinions)
  - Inputs where the compression boundary is genuinely ambiguous
- Audit the bootstrapper system prompt for under-specification on these NL subtypes
- Output: a taxonomy of bad input patterns + a revised bootstrapper prompt

### Why this precedes re-generation

Running `regen_oscillating.py` without fixing the bootstrapper prompt will produce new contradictory targets for the same ambiguous inputs. The prompt fix is the prerequisite.

---

## Investigation 3: Attention Analysis — QUEUED

Determine whether the compressor attends to semantically heavy tokens during inference, or performs surface deletion (style transfer, not semantic compression). Uses activation captures on held-out samples. Distinguishes genuine compression capability from prompt-following with a tuned prior.

---

## Investigation 4: Gradient Attribution — QUEUED

Integrated Gradients across four directives:

- **Fact overlap gates** — did specific facts survive compression?
- **Trace backwards** — which input tokens did the model ignore when it dropped a fact?
- **Numerical / proper noun survival** — high-value tokens that must never be dropped
- **Loss function mismatch** — cross-entropy treats every token equally; semantic tokens are not equal

Prospect: reweighted training loss that penalises dropping high-value tokens more than filler.

---

## Investigation 5: Probing Classifiers — QUEUED

Test whether compression-relevant concepts (entailment, topic sentence identification, coreference) are linearly separable in the model's internal representations. If not, LoRA (a low-rank perturbation) cannot install that capability regardless of training duration. Compare 3B vs 8B to determine whether model scale is a hard constraint.

---

## Investigation 6: Prompt Sensitivity — QUEUED

Systematic prompt variation on identical inputs, measuring output variance. Determines whether the LoRA has internalised compression as a weight-level capability or is following instructions with a tuned prior. High variance under minor rewording = fragile system, not a trained compressor.

