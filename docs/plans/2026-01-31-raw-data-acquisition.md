# Raw Data Acquisition Plan for 10K Training Samples

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Acquire and process diverse raw data sources to generate ~10K validated compression training pairs (5K code + 5K NL).

**Architecture:** Download/export raw data → convert to seed format (JSONL with `text` field) → generate compressions with V1 adapter → validate with Claude+GPT.

**Tech Stack:** HuggingFace datasets, Python scripts, existing corpus loaders.

---

## Current State

### Existing Raw Data
```
data/raw/code.jsonl     1,571 samples  (1.5MB)  - Python functions/classes from pydantic
data/raw/nl_docs.jsonl  1,234 samples  (1.3MB)  - Technical docs (AI, aerospace, etc.)
────────────────────────────────────────────────
Total:                  2,805 samples  (2.8MB)
```

### Target
- **Code domain:** 5,000+ raw samples → ~4,500 validated pairs (90% pass rate)
- **NL domain:** 5,000+ raw samples → ~4,500 validated pairs (90% pass rate)
- **Total validated:** ~9,000-10,000 pairs

### Gap Analysis
| Domain | Current | Target | Gap |
|--------|---------|--------|-----|
| Code | 1,571 | 5,500+ | ~4,000 |
| NL | 1,234 | 5,500+ | ~4,300 |

---

## Data Source Strategy

### Code Domain Sources

| Source | Est. Samples | Quality | Effort |
|--------|--------------|---------|--------|
| **Clone more repos** (local) | 2,000+ | High | Low |
| **HuggingFace: bigcode/the-stack** | 2,000+ | High | Medium |
| **HuggingFace: codeparrot/github-code** | 1,000+ | Medium | Medium |
| **Total** | **5,000+** | | |

### NL Domain Sources

| Source | Est. Samples | Quality | Effort |
|--------|--------------|---------|--------|
| **Claude conversation exports** | 500-1,000 | Very High | Low |
| **HuggingFace: databricks/dolly-15k** | 2,000+ | High | Low |
| **HuggingFace: OpenAssistant/oasst1** | 2,000+ | High | Medium |
| **More Wikipedia/docs** | 500+ | Medium | Low |
| **Total** | **5,000+** | | |

---

## Phase 1: Code Data Acquisition

### Task 1: Clone additional high-quality Python repos

**Step 1: Select repos with good code quality**

Target repos (popular, well-documented, diverse domains):
```bash
# Web frameworks
git clone --depth 1 https://github.com/encode/starlette.git data/raw/code/starlette
git clone --depth 1 https://github.com/encode/httpx.git data/raw/code/httpx

# Data/ML
git clone --depth 1 https://github.com/pola-rs/polars.git data/raw/code/polars  # Python bindings
git clone --depth 1 https://github.com/huggingface/transformers.git data/raw/code/transformers

# CLI/Tools
git clone --depth 1 https://github.com/Textualize/rich.git data/raw/code/rich
git clone --depth 1 https://github.com/pallets/click.git data/raw/code/click

# Async
git clone --depth 1 https://github.com/python-trio/trio.git data/raw/code/trio
```

**Step 2: Extract code samples**

```bash
# Extract from each repo
python scripts/prepare_corpus.py \
  --input data/raw/code/starlette \
  --output data/raw/code_starlette.jsonl \
  --min-lines 5 --max-lines 60 \
  --min-chars 150 --max-chars 2000

# Repeat for each repo...

# Merge all code files
cat data/raw/code.jsonl data/raw/code_*.jsonl > data/raw/code_all.jsonl
```

**Step 3: Verify and deduplicate**

```bash
# Count and verify
wc -l data/raw/code_all.jsonl

# Simple dedup by text hash (create script if needed)
python scripts/dedupe_corpus.py --input data/raw/code_all.jsonl --output data/raw/code_deduped.jsonl
```

---

### Task 2: Download HuggingFace code datasets

**Files:**
- Create: `scripts/download_hf_code.py`

**Step 1: Write download script**

```python
#!/usr/bin/env python3
"""Download code samples from HuggingFace datasets.

Usage:
    python scripts/download_hf_code.py --dataset bigcode/the-stack --lang python --limit 2000
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


def extract_code_samples(
    dataset_name: str,
    language: str = "python",
    limit: int = 2000,
    min_chars: int = 150,
    max_chars: int = 2000,
    split: str = "train",
) -> list[dict]:
    """Extract code samples from HuggingFace dataset."""
    
    console.print(f"[cyan]Loading {dataset_name}...[/cyan]")
    
    # Handle different dataset structures
    if dataset_name == "bigcode/the-stack":
        ds = load_dataset(dataset_name, data_dir=f"data/{language}", split=split, streaming=True)
        content_key = "content"
    elif dataset_name == "codeparrot/github-code":
        ds = load_dataset(dataset_name, languages=[language], split=split, streaming=True)
        content_key = "code"
    else:
        ds = load_dataset(dataset_name, split=split, streaming=True)
        content_key = "content"  # Adjust as needed
    
    samples = []
    seen_hashes = set()
    
    for item in track(ds, description="Processing...", total=limit * 3):
        if len(samples) >= limit:
            break
            
        code = item.get(content_key, "")
        if not code:
            continue
            
        # Filter by length
        if len(code) < min_chars or len(code) > max_chars:
            continue
        
        # Simple dedup
        code_hash = hash(code[:500])
        if code_hash in seen_hashes:
            continue
        seen_hashes.add(code_hash)
        
        # Filter out non-function/class code (simple heuristic)
        if not any(kw in code for kw in ["def ", "class ", "async def "]):
            continue
        
        samples.append({
            "text": code,
            "language": language,
            "source": dataset_name,
        })
    
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Download code from HuggingFace")
    parser.add_argument("--dataset", default="bigcode/the-stack", 
                       choices=["bigcode/the-stack", "codeparrot/github-code"])
    parser.add_argument("--lang", default="python")
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=Path("data/raw/hf_code.jsonl"))
    parser.add_argument("--min-chars", type=int, default=150)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()
    
    samples = extract_code_samples(
        args.dataset,
        language=args.lang,
        limit=args.limit,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    console.print(f"[green]Saved {len(samples)} samples to {args.output}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Download from the-stack**

```bash
# Requires: pip install datasets
python scripts/download_hf_code.py \
  --dataset bigcode/the-stack \
  --lang python \
  --limit 2000 \
  --output data/raw/hf_thestack.jsonl
```

**Step 3: Download from github-code**

```bash
python scripts/download_hf_code.py \
  --dataset codeparrot/github-code \
  --lang python \
  --limit 1500 \
  --output data/raw/hf_githubcode.jsonl
```

---

## Phase 2: NL Data Acquisition

### Task 3: Export Claude conversations

**Step 1: Export from Claude.ai**

1. Go to https://claude.ai/settings
2. Click "Export Data" 
3. Download the JSON export
4. Save to `data/raw/claude_export/conversations.json`

**Step 2: Create conversion script**

**Files:**
- Create: `scripts/convert_claude_export.py`

```python
#!/usr/bin/env python3
"""Convert Claude conversation export to seed format.

Usage:
    python scripts/convert_claude_export.py \
      --input data/raw/claude_export/conversations.json \
      --output data/raw/claude_conversations.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console

console = Console()


def extract_user_messages(
    conversations_path: Path,
    min_chars: int = 100,
    max_chars: int = 2000,
) -> list[dict]:
    """Extract user messages from Claude export."""
    
    with open(conversations_path) as f:
        data = json.load(f)
    
    samples = []
    
    # Claude export format: list of conversations
    conversations = data if isinstance(data, list) else data.get("conversations", [])
    
    for conv in conversations:
        messages = conv.get("chat_messages", conv.get("messages", []))
        
        for msg in messages:
            # Only extract user messages (these are what we want to compress)
            if msg.get("sender") != "human" and msg.get("role") != "user":
                continue
            
            text = msg.get("text", msg.get("content", ""))
            if not text:
                continue
            
            # Filter by length
            if len(text) < min_chars or len(text) > max_chars:
                continue
            
            # Skip very short or code-heavy messages
            code_ratio = text.count("```") / max(len(text), 1)
            if code_ratio > 0.3:  # Skip if >30% is code blocks
                continue
            
            samples.append({
                "text": text,
                "source": "claude_export",
                "conversation_id": conv.get("uuid", conv.get("id", "")),
            })
    
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Claude export to seed format")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/raw/claude_conversations.jsonl"))
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()
    
    if not args.input.exists():
        console.print(f"[red]Input file not found: {args.input}[/red]")
        return 1
    
    samples = extract_user_messages(args.input, args.min_chars, args.max_chars)
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")
    
    console.print(f"[green]Extracted {len(samples)} user messages to {args.output}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 3: Run conversion**

```bash
python scripts/convert_claude_export.py \
  --input data/raw/claude_export/conversations.json \
  --output data/raw/claude_conversations.jsonl
```

---

### Task 4: Download HuggingFace conversation datasets

**Files:**
- Create: `scripts/download_hf_conversations.py`

```python
#!/usr/bin/env python3
"""Download conversation samples from HuggingFace datasets.

Usage:
    python scripts/download_hf_conversations.py --dataset databricks/dolly-15k --limit 2000
    python scripts/download_hf_conversations.py --dataset OpenAssistant/oasst1 --limit 2000
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.progress import track

console = Console()


def extract_dolly_samples(limit: int, min_chars: int, max_chars: int) -> list[dict]:
    """Extract from databricks/dolly-15k."""
    ds = load_dataset("databricks/dolly-15k", split="train")
    
    samples = []
    for item in track(ds, description="Processing dolly-15k..."):
        if len(samples) >= limit:
            break
        
        # Dolly has: instruction, context, response, category
        # We want the context + instruction as compressible text
        text_parts = []
        if item.get("context"):
            text_parts.append(item["context"])
        if item.get("instruction"):
            text_parts.append(item["instruction"])
        
        text = "\n\n".join(text_parts)
        
        if len(text) < min_chars or len(text) > max_chars:
            continue
        
        # Filter out pure code/math
        if text.count("```") > 2:
            continue
        
        samples.append({
            "text": text,
            "source": "databricks/dolly-15k",
            "category": item.get("category", ""),
        })
    
    return samples


def extract_oasst_samples(limit: int, min_chars: int, max_chars: int) -> list[dict]:
    """Extract from OpenAssistant/oasst1."""
    ds = load_dataset("OpenAssistant/oasst1", split="train")
    
    samples = []
    seen = set()
    
    for item in track(ds, description="Processing oasst1..."):
        if len(samples) >= limit:
            break
        
        # OASST has: text, role, lang, message_tree_id, etc.
        # We want prompter (user) messages in English
        if item.get("role") != "prompter":
            continue
        if item.get("lang") != "en":
            continue
        
        text = item.get("text", "")
        
        if len(text) < min_chars or len(text) > max_chars:
            continue
        
        # Dedup
        text_hash = hash(text[:200])
        if text_hash in seen:
            continue
        seen.add(text_hash)
        
        samples.append({
            "text": text,
            "source": "OpenAssistant/oasst1",
            "message_id": item.get("message_id", ""),
        })
    
    return samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Download conversations from HuggingFace")
    parser.add_argument("--dataset", required=True,
                       choices=["databricks/dolly-15k", "OpenAssistant/oasst1", "all"])
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=2000)
    args = parser.parse_args()
    
    all_samples = []
    
    if args.dataset in ["databricks/dolly-15k", "all"]:
        samples = extract_dolly_samples(args.limit, args.min_chars, args.max_chars)
        all_samples.extend(samples)
        console.print(f"[green]Extracted {len(samples)} from dolly-15k[/green]")
    
    if args.dataset in ["OpenAssistant/oasst1", "all"]:
        samples = extract_oasst_samples(args.limit, args.min_chars, args.max_chars)
        all_samples.extend(samples)
        console.print(f"[green]Extracted {len(samples)} from oasst1[/green]")
    
    # Set output path
    if args.output is None:
        if args.dataset == "all":
            args.output = Path("data/raw/hf_conversations.jsonl")
        else:
            name = args.dataset.replace("/", "_")
            args.output = Path(f"data/raw/hf_{name}.jsonl")
    
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    console.print(f"[green]Saved {len(all_samples)} total samples to {args.output}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

**Step 2: Download datasets**

```bash
# Install datasets library if needed
pip install datasets

# Download Dolly
python scripts/download_hf_conversations.py \
  --dataset databricks/dolly-15k \
  --limit 2500 \
  --output data/raw/hf_dolly.jsonl

# Download OASST
python scripts/download_hf_conversations.py \
  --dataset OpenAssistant/oasst1 \
  --limit 2500 \
  --output data/raw/hf_oasst.jsonl
```

---

## Phase 3: Data Consolidation

### Task 5: Create unified corpus merger

**Files:**
- Create: `scripts/merge_corpus.py`

```python
#!/usr/bin/env python3
"""Merge multiple JSONL corpus files into unified format.

Usage:
    python scripts/merge_corpus.py \
      --inputs data/raw/code.jsonl data/raw/hf_thestack.jsonl \
      --output data/raw/code_merged.jsonl \
      --domain code \
      --dedupe
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def compute_hash(text: str) -> str:
    """Compute hash for deduplication."""
    return hashlib.md5(text.encode()).hexdigest()[:16]


def merge_corpus_files(
    input_files: list[Path],
    output_file: Path,
    domain: str,
    dedupe: bool = True,
    min_chars: int = 100,
    max_chars: int = 2500,
) -> dict:
    """Merge multiple JSONL files into one."""
    
    all_samples = []
    seen_hashes = set()
    stats = {"total_read": 0, "duplicates": 0, "filtered": 0, "kept": 0}
    source_counts = {}
    
    for input_file in input_files:
        if not input_file.exists():
            console.print(f"[yellow]Warning: {input_file} not found, skipping[/yellow]")
            continue
        
        file_count = 0
        with open(input_file) as f:
            for line in f:
                if not line.strip():
                    continue
                
                stats["total_read"] += 1
                
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                
                # Normalize text field
                text = item.get("text") or item.get("content") or item.get("code", "")
                if not text:
                    stats["filtered"] += 1
                    continue
                
                # Length filter
                if len(text) < min_chars or len(text) > max_chars:
                    stats["filtered"] += 1
                    continue
                
                # Dedupe
                if dedupe:
                    text_hash = compute_hash(text)
                    if text_hash in seen_hashes:
                        stats["duplicates"] += 1
                        continue
                    seen_hashes.add(text_hash)
                
                # Normalize to seed format
                normalized = {
                    "text": text,
                    "domain": domain,
                    "source": item.get("source", str(input_file.stem)),
                }
                
                # Preserve useful metadata
                if "language" in item:
                    normalized["language"] = item["language"]
                if "category" in item:
                    normalized["category"] = item["category"]
                
                all_samples.append(normalized)
                file_count += 1
                stats["kept"] += 1
        
        source_counts[input_file.name] = file_count
        console.print(f"  {input_file.name}: {file_count} samples")
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + "\n")
    
    stats["source_counts"] = source_counts
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge corpus files")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", choices=["code", "nl"], required=True)
    parser.add_argument("--dedupe", action="store_true", default=True)
    parser.add_argument("--no-dedupe", dest="dedupe", action="store_false")
    parser.add_argument("--min-chars", type=int, default=100)
    parser.add_argument("--max-chars", type=int, default=2500)
    args = parser.parse_args()
    
    console.print(f"[cyan]Merging {len(args.inputs)} files...[/cyan]")
    
    stats = merge_corpus_files(
        args.inputs,
        args.output,
        args.domain,
        dedupe=args.dedupe,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )
    
    # Print summary
    table = Table(title="Merge Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Total read", str(stats["total_read"]))
    table.add_row("Duplicates removed", str(stats["duplicates"]))
    table.add_row("Filtered (length)", str(stats["filtered"]))
    table.add_row("Final count", str(stats["kept"]))
    table.add_row("Output", str(args.output))
    console.print(table)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

---

### Task 6: Merge all data sources

**Step 1: Merge code sources**

```bash
python scripts/merge_corpus.py \
  --inputs \
    data/raw/code.jsonl \
    data/raw/code_starlette.jsonl \
    data/raw/code_httpx.jsonl \
    data/raw/code_rich.jsonl \
    data/raw/hf_thestack.jsonl \
    data/raw/hf_githubcode.jsonl \
  --output data/raw/code_merged.jsonl \
  --domain code \
  --dedupe
```

**Step 2: Merge NL sources**

```bash
python scripts/merge_corpus.py \
  --inputs \
    data/raw/nl_docs.jsonl \
    data/raw/claude_conversations.jsonl \
    data/raw/hf_dolly.jsonl \
    data/raw/hf_oasst.jsonl \
  --output data/raw/nl_merged.jsonl \
  --domain nl \
  --dedupe
```

---

## Phase 4: Quality Verification

### Task 7: Verify corpus quality

**Step 1: Sample inspection**

```bash
# Check code samples
head -5 data/raw/code_merged.jsonl | python -m json.tool

# Check NL samples  
head -5 data/raw/nl_merged.jsonl | python -m json.tool

# Verify counts
wc -l data/raw/code_merged.jsonl data/raw/nl_merged.jsonl
```

**Step 2: Length distribution check**

```bash
python -c "
import json
from pathlib import Path
from collections import Counter

for domain in ['code', 'nl']:
    path = Path(f'data/raw/{domain}_merged.jsonl')
    if not path.exists():
        continue
    
    lengths = []
    with open(path) as f:
        for line in f:
            text = json.loads(line).get('text', '')
            lengths.append(len(text))
    
    print(f'\n{domain.upper()} corpus ({len(lengths)} samples):')
    print(f'  Min: {min(lengths)}, Max: {max(lengths)}, Avg: {sum(lengths)//len(lengths)}')
    
    # Bucket distribution
    buckets = Counter()
    for l in lengths:
        if l < 200: buckets['<200'] += 1
        elif l < 500: buckets['200-500'] += 1
        elif l < 1000: buckets['500-1000'] += 1
        elif l < 2000: buckets['1000-2000'] += 1
        else: buckets['>2000'] += 1
    
    for bucket, count in sorted(buckets.items()):
        print(f'  {bucket}: {count} ({count*100//len(lengths)}%)')
"
```

---

## Execution Summary

### Quick Reference Commands

```bash
# === PHASE 1: CODE ===

# 1a. Clone repos
git clone --depth 1 https://github.com/encode/starlette.git data/raw/code/starlette
git clone --depth 1 https://github.com/encode/httpx.git data/raw/code/httpx
git clone --depth 1 https://github.com/Textualize/rich.git data/raw/code/rich
git clone --depth 1 https://github.com/pallets/click.git data/raw/code/click

# 1b. Extract code
for repo in starlette httpx rich click; do
  python scripts/prepare_corpus.py \
    --input data/raw/code/$repo \
    --output data/raw/code_$repo.jsonl \
    --min-lines 5 --max-lines 60
done

# 1c. Download HuggingFace code
python scripts/download_hf_code.py --dataset bigcode/the-stack --limit 2000
python scripts/download_hf_code.py --dataset codeparrot/github-code --limit 1500


# === PHASE 2: NL ===

# 2a. Convert Claude export (after downloading from claude.ai)
python scripts/convert_claude_export.py \
  --input data/raw/claude_export/conversations.json \
  --output data/raw/claude_conversations.jsonl

# 2b. Download HuggingFace conversations
python scripts/download_hf_conversations.py --dataset databricks/dolly-15k --limit 2500
python scripts/download_hf_conversations.py --dataset OpenAssistant/oasst1 --limit 2500


# === PHASE 3: MERGE ===

# 3a. Merge code
python scripts/merge_corpus.py \
  --inputs data/raw/code*.jsonl data/raw/hf_thestack.jsonl data/raw/hf_githubcode.jsonl \
  --output data/raw/code_merged.jsonl \
  --domain code --dedupe

# 3b. Merge NL
python scripts/merge_corpus.py \
  --inputs data/raw/nl_docs.jsonl data/raw/claude_conversations.jsonl data/raw/hf_dolly.jsonl data/raw/hf_oasst.jsonl \
  --output data/raw/nl_merged.jsonl \
  --domain nl --dedupe


# === PHASE 4: VERIFY ===
wc -l data/raw/*_merged.jsonl
```

### Expected Output

| File | Expected Samples |
|------|------------------|
| `data/raw/code_merged.jsonl` | 5,000-6,000 |
| `data/raw/nl_merged.jsonl` | 5,000-6,000 |
| **Total raw samples** | **10,000-12,000** |

### Time Estimates

| Task | Time |
|------|------|
| Clone repos | 5 min |
| Extract code | 10 min |
| Download HF code | 15-30 min |
| Claude export | 5 min (manual) |
| Download HF NL | 15-30 min |
| Merge & verify | 5 min |
| **Total** | **~1-1.5 hours** |

### After This Plan

Once you have the merged raw data, continue with:
1. `python scripts/generate_synthetic.py` - Generate compressions with V1 adapter
2. `python scripts/validate_synthetic.py` - Validate with Claude+GPT
3. `python scripts/format_training_data.py` - Format for training
4. `python scripts/train_tinker.py` - Train V2 on Tinker

---

## Seed Format Reference

All raw data should be normalized to this format:

```json
{
  "text": "The actual content to be compressed...",
  "domain": "code" | "nl",
  "source": "dataset_name or file_origin",
  "language": "python",  // optional, for code
  "category": "qa"       // optional, for NL
}
```

The `text` field is the only required field for compression generation.
