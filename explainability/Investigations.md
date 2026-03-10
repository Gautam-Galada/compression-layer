# Project Investigation Roadmap: Post-Hoc Analysis & Strategy

[Currently : dataset catastrophing]

---

## Investigation 1: Bootstrapping Data as a Strategy

This investigation focuses on the health and diversity of the synthetic dataset used for training, utilizing Dataset Cartography to identify learning patterns.

### Dataset Cartography
Dataset cartography tracks the confidence and variability of each training example across epochs to map the dataset into three distinct regions:
* **Easy-to-Learn:** High-confidence, low-variability. The model has already overfit these.
* **Hard-to-Learn:** Low-confidence, low-variability. These may be noise or "impossible" tasks.
* **Ambiguous:** High-variability. The model oscillates here, suggesting inconsistent or contradictory compression targets.



### Deduplication and Diversity Analysis
Bootstrapped data often suffers from low effective diversity. Even a dataset of 20K examples may consist of a few repeated templates.
* **Manifold Analysis:** By embedding the 20K examples, we can identify tight clusters.
* **The Risk:** If the model is trained on repetitive templates, it will fail catastrophically when encountering out-of-distribution (OOD) domains.

---

## Investigation 2: Posthoc Analysis - Rethinking Training Paradigm

This investigation examines whether the model has truly internalized the task or is simply performing surface-level manipulation across the Model, Data, and Evaluation levels.

### 1. Attention Analysis
Determining whether the compressor model during inference is attending to semantically heavy tokens (Semantic Compression) or merely skipping adjectives and filler phrases (Style Transfer).



### 2. Gradient Attribution
Using techniques like Integrated Gradients to analyze four specific directives:
* **Fact Overlap Gates:** A test to see if specific facts survived the compression.
* **Trace Backwards:** Tracing an error back to see which input words the model ignored.
* **Numerical/Proper Nouns:** High-value data points that must never be dropped.
* **Loss Function Mismatch:** Addressing the math that treats every word as equally important.

**Prospects:**
* **Reweighted Training Paradigm:** Adjusting loss to provide specificity over concerned tokens.
* **Compression Intent:** Determining if the model understands semantic reduction or blindly deletes.

---

## Investigation 3: Probing Classifiers - Model Choice

Testing whether the internal representations of a model encode the kinds of distinctions compression requires.

* **Linear Separability:** Testing if concepts like "Is this sentence entailed by the document?" or "Is this a topic sentence?" are linearly separable. If they are not, LoRA adaptation (a low-rank perturbation) likely cannot install that capability.
* **Reasoning Depth:** Comparing capability across 3B vs 8B parameter models.

---

## Investigation 4: Nature of Prompts

Determining how much of the observed behavior is weight-driven versus prompt-driven.

### Prompt Sensitivity Analysis
Running the same inputs with systematically varied prompts and measuring output variance.
* **Weight-Driven:** The LoRA has internalized compression as a core capability.
* **Prompt-Driven:** The model is merely following instructions with a slightly tuned prior.

If behavior changes dramatically with minor rewording, the LoRA hasn't internalized the capability, resulting in a fragile system.
