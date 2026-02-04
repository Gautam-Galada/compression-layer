# sanitize_training_data_v2.py
"""
Data Sanitization for Compression Training - Version 2
Implements Option 2: Separate validation rules for code vs natural language
"""

import json
import re
from collections import defaultdict
from pathlib import Path

# ============================================================================
# SYMBOL DEFINITIONS
# ============================================================================

SYMBOLS = {"→", "|", "@", "∵", ":"}

# Strict keywords for natural language only
LOCATION_KEYWORDS_NL = [
    "located in",
    "located at",
    "based in",
    "situated in",
    "found in",
    "positioned at",
    "positioned in",
    "place is",
    "place was",
    "city of",
    "town of",
    "on the shores of",
    "near the",
    "by the",
]

CAUSATION_KEYWORDS_NL = [
    "because of",
    "due to",
    "caused by",
    "as a result of",
    "leads to",
    "results in",
    "led to",
    "resulted in",
    "owing to",
    "on account of",
    "thanks to",
    "consequently",
    "therefore",
    "thus",
]

NEGATION_KEYWORDS = [
    "not",
    "no",
    "never",
    "neither",
    "nor",
    "n't",
    "without",
    "none",
    "nothing",
    "nobody",
    "nowhere",
    "no longer",
    "no more",
]

# Code detection indicators
CODE_INDICATORS = [
    # Python
    "def ",
    "class ",
    "import ",
    "return ",
    "yield ",
    "async ",
    "await ",
    "self.",
    "__init__",
    "__",
    "lambda ",
    "isinstance(",
    "raise ",
    "@classmethod",
    "@staticmethod",
    "@property",
    # JavaScript/TypeScript
    "function ",
    "const ",
    "let ",
    "var ",
    "=>",
    "async function",
    # General programming
    "fn:",
    "->",
    "fn(",
    "void ",
    "int ",
    "string ",
    "bool ",
    # Code blocks
    "```",
    "```python",
    "```javascript",
    "```java",
    # Common syntax patterns
    "    def ",
    "    class ",  # Indented code
]

# ============================================================================
# CONTENT TYPE DETECTION
# ============================================================================


def is_code_sample(verbose: str) -> bool:
    """
    Detect if sample is code-related.

    Returns True if:
    - Contains code indicators
    - Has significant indentation patterns
    - Contains common programming syntax
    """
    verbose_lower = verbose.lower()

    # Check for explicit code indicators
    for indicator in CODE_INDICATORS:
        if indicator.lower() in verbose_lower:
            return True

    # Check for code-like structure
    lines = verbose.split("\n")

    # Multiple indented lines suggest code
    indented_lines = sum(1 for line in lines if line.startswith("    ") or line.startswith("\t"))
    if indented_lines >= 2:
        return True

    # Check for common code patterns
    code_patterns = [
        r"\bdef\s+\w+\(",  # Python functions
        r"\bclass\s+\w+[\(:]",  # Python classes
        r"function\s+\w+\(",  # JS functions
        r"\w+\s*=\s*function\(",  # JS function assignment
        r"\w+\s*:\s*\w+\s*[,\)]",  # Type annotations
        r"\{[\s\S]*\}",  # Code blocks with braces
    ]

    return any(re.search(pattern, verbose) for pattern in code_patterns)


# ============================================================================
# EXTRACTION HELPERS
# ============================================================================


def extract_verbose_compressed(sample: dict) -> tuple[str, str]:
    """Extract input (verbose) and output (compressed) from message structure."""
    messages = sample.get("messages", [])

    verbose = ""
    compressed = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "user" and "Compress:" in content:
            verbose = content.split("Compress:", 1)[1].strip()
        elif role == "assistant":
            compressed = content.strip()

    return verbose, compressed


def compute_compression_ratio(verbose: str, compressed: str) -> float:
    """Compute word-level compression ratio."""
    v_words = len(verbose.split())
    c_words = len(compressed.split())

    if c_words == 0:
        return 0.0

    return v_words / c_words


# ============================================================================
# VALIDATION RULES - UNIVERSAL (Apply to both code and NL)
# ============================================================================


def rule_a_ratio_check(verbose: str, compressed: str) -> tuple[bool, str]:
    """Rule A: Remove samples with compression ratio < 1.0 (longer than input)"""
    ratio = compute_compression_ratio(verbose, compressed)

    if ratio < 1.0:
        return False, f"Ratio {ratio:.2f} < 1.0 (compressed is longer than input)"

    return True, ""


def rule_b_orphaned_symbols(compressed: str) -> tuple[bool, str]:
    """Rule B: Remove samples with orphaned symbols (at start/end or consecutive)"""
    if not compressed:
        return False, "Empty compression"

    # Check start (allow ``` for code blocks)
    if compressed[0] in SYMBOLS:
        return False, f"Orphaned symbol at start: '{compressed[0]}'"

    # Check end
    if compressed[-1] in SYMBOLS and compressed[-1] != ":":
        return False, f"Orphaned symbol at end: '{compressed[-1]}'"

    # Check consecutive symbols (except :: which is valid)
    for i in range(len(compressed) - 1):
        if compressed[i] in SYMBOLS and compressed[i + 1] in SYMBOLS:
            # Allow :: (type annotation)
            if compressed[i] == ":" and compressed[i + 1] == ":":
                continue
            return False, f"Consecutive symbols: '{compressed[i]}{compressed[i + 1]}'"

    return True, ""


# ============================================================================
# VALIDATION RULES - NATURAL LANGUAGE ONLY
# ============================================================================


def rule_c_negation_preservation(verbose: str, compressed: str) -> tuple[bool, str]:
    """Rule C: Remove samples that lost negation (NL only)"""
    verbose_lower = verbose.lower()
    compressed_lower = compressed.lower()

    # Check if input has negation
    has_input_negation = any(
        re.search(r"\b" + re.escape(kw) + r"\b", verbose_lower) for kw in NEGATION_KEYWORDS
    )

    if not has_input_negation:
        return True, ""  # No negation to preserve

    # Check if output preserved negation
    has_output_negation = any(
        re.search(r"\b" + re.escape(kw) + r"\b", compressed_lower) for kw in NEGATION_KEYWORDS
    )

    # Check for negation symbols
    has_neg_symbol = "¬" in compressed or "~" in compressed or "!" in compressed

    if not (has_output_negation or has_neg_symbol):
        return False, "Negation present in input but lost in compression"

    return True, ""


def rule_d_semantic_symbol_usage_nl(verbose: str, compressed: str) -> tuple[bool, str]:
    """Rule D: Remove samples that should use @ or ∵ but don't (NL only, strict keywords)"""
    verbose_lower = verbose.lower()

    # Check location context (strict multi-word phrases)
    has_location_context = any(kw in verbose_lower for kw in LOCATION_KEYWORDS_NL)
    if has_location_context and "@" not in compressed:
        return False, "Location context present but '@' not used"

    # Check causation context (strict multi-word phrases)
    has_causation_context = any(kw in verbose_lower for kw in CAUSATION_KEYWORDS_NL)
    if has_causation_context and "∵" not in compressed:
        return False, "Causation context present but '∵' not used"

    return True, ""


# ============================================================================
# MAIN SANITIZATION
# ============================================================================


def sanitize_dataset(input_path: Path, output_path: Path) -> dict:
    """
    Apply filtering rules separately for code and NL samples.

    Returns statistics dictionary.
    """
    print(f"Loading data from {input_path}...")

    with open(input_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    print(f"✓ Loaded {len(data)} samples\n")

    # Track statistics
    stats = {
        "total_input": len(data),
        "code_samples": 0,
        "nl_samples": 0,
        "code_passed": 0,
        "nl_passed": 0,
        "rule_a_failed": 0,
        "rule_b_failed": 0,
        "rule_c_failed": 0,
        "rule_d_failed": 0,
        "passed_all": 0,
        "failed_samples": [],
        "passed_samples": [],
        "code_failed_samples": [],
        "nl_failed_samples": [],
    }

    sanitized_data = []

    print("Applying filtering rules...\n")
    print("=" * 80)

    for idx, sample in enumerate(data):
        verbose, compressed = extract_verbose_compressed(sample)

        if not verbose or not compressed:
            stats["rule_a_failed"] += 1
            stats["failed_samples"].append(
                {
                    "id": idx,
                    "type": "unknown",
                    "reason": "Missing input or output",
                    "verbose": verbose[:50] if verbose else "",
                    "compressed": compressed[:50] if compressed else "",
                }
            )
            continue

        # Detect content type
        is_code = is_code_sample(verbose)
        content_type = "code" if is_code else "nl"

        if is_code:
            stats["code_samples"] += 1
        else:
            stats["nl_samples"] += 1

        # Apply rules based on content type
        passed = True
        failure_reason = ""

        # Rule A: Compression ratio (UNIVERSAL)
        rule_a_pass, reason = rule_a_ratio_check(verbose, compressed)
        if not rule_a_pass:
            stats["rule_a_failed"] += 1
            passed = False
            failure_reason = f"Rule A: {reason}"

        # Rule B: Orphaned symbols (UNIVERSAL)
        if passed:
            rule_b_pass, reason = rule_b_orphaned_symbols(compressed)
            if not rule_b_pass:
                stats["rule_b_failed"] += 1
                passed = False
                failure_reason = f"Rule B: {reason}"

        # Rule C: Negation preservation (NL ONLY)
        if passed and not is_code:
            rule_c_pass, reason = rule_c_negation_preservation(verbose, compressed)
            if not rule_c_pass:
                stats["rule_c_failed"] += 1
                passed = False
                failure_reason = f"Rule C: {reason}"

        # Rule D: Semantic symbol usage (NL ONLY)
        if passed and not is_code:
            rule_d_pass, reason = rule_d_semantic_symbol_usage_nl(verbose, compressed)
            if not rule_d_pass:
                stats["rule_d_failed"] += 1
                passed = False
                failure_reason = f"Rule D: {reason}"

        # Record result
        if passed:
            stats["passed_all"] += 1
            if is_code:
                stats["code_passed"] += 1
            else:
                stats["nl_passed"] += 1

            sanitized_data.append(sample)
            stats["passed_samples"].append(
                {
                    "id": idx,
                    "type": content_type,
                    "ratio": compute_compression_ratio(verbose, compressed),
                    "compressed": compressed[:80],
                }
            )
        else:
            failed_entry = {
                "id": idx,
                "type": content_type,
                "reason": failure_reason,
                "verbose": verbose[:80],
                "compressed": compressed[:80],
            }
            stats["failed_samples"].append(failed_entry)

            if is_code:
                stats["code_failed_samples"].append(failed_entry)
            else:
                stats["nl_failed_samples"].append(failed_entry)

    # Save sanitized data
    print(f"\nSaving sanitized data to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in sanitized_data:
            f.write(json.dumps(sample) + "\n")

    print(f"✓ Saved {len(sanitized_data)} sanitized samples\n")

    return stats


def print_statistics(stats: dict):
    """Print detailed statistics about sanitization."""
    print("=" * 80)
    print("SANITIZATION STATISTICS - OPTION 2 (Code vs NL Split)")
    print("=" * 80)
    print()

    print(f"Total input samples:        {stats['total_input']:5d}")
    print(
        f"  Code samples:             {stats['code_samples']:5d} ({stats['code_samples'] / stats['total_input'] * 100:5.1f}%)"
    )
    print(
        f"  NL samples:               {stats['nl_samples']:5d} ({stats['nl_samples'] / stats['total_input'] * 100:5.1f}%)"
    )
    print()

    print(
        f"Passed all rules:           {stats['passed_all']:5d} ({stats['passed_all'] / stats['total_input'] * 100:5.1f}%)"
    )
    print(
        f"  Code passed:              {stats['code_passed']:5d} ({stats['code_passed'] / stats['code_samples'] * 100 if stats['code_samples'] > 0 else 0:5.1f}%)"
    )
    print(
        f"  NL passed:                {stats['nl_passed']:5d} ({stats['nl_passed'] / stats['nl_samples'] * 100 if stats['nl_samples'] > 0 else 0:5.1f}%)"
    )
    print()

    print("Failed by rule:")
    print(f"  Rule A (ratio < 1.0):     {stats['rule_a_failed']:5d} (universal)")
    print(f"  Rule B (orphaned symbols):{stats['rule_b_failed']:5d} (universal)")
    print(f"  Rule C (lost negation):   {stats['rule_c_failed']:5d} (NL only)")
    print(f"  Rule D (missing @ or ∵):  {stats['rule_d_failed']:5d} (NL only)")
    print()

    total_failed = len(stats["failed_samples"])
    print(
        f"Total failed:               {total_failed:5d} ({total_failed / stats['total_input'] * 100:5.1f}%)"
    )
    print()

    # Show examples of failed CODE samples
    print("=" * 80)
    print("FAILED CODE SAMPLES (First 5 examples)")
    print("=" * 80)
    print()

    code_failed = stats["code_failed_samples"][:5]
    if code_failed:
        for sample in code_failed:
            print(f"Sample {sample['id']} (CODE):")
            print(f"  Reason: {sample['reason']}")
            print(f"  Input:  {sample['verbose']}...")
            print(f"  Output: {sample['compressed']}...")
            print()
    else:
        print("No code samples failed.\n")

    # Show examples of failed NL samples
    print("=" * 80)
    print("FAILED NL SAMPLES (First 5 examples)")
    print("=" * 80)
    print()

    nl_failed = stats["nl_failed_samples"][:5]
    if nl_failed:
        for sample in nl_failed:
            print(f"Sample {sample['id']} (NL):")
            print(f"  Reason: {sample['reason']}")
            print(f"  Input:  {sample['verbose']}...")
            print(f"  Output: {sample['compressed']}...")
            print()
    else:
        print("No NL samples failed.\n")

    # Show examples of passed samples (mixed)
    print("=" * 80)
    print("PASSED SAMPLES (First 10 examples - highest compression)")
    print("=" * 80)
    print()

    sorted_passed = sorted(stats["passed_samples"], key=lambda x: x.get("ratio", 0), reverse=True)

    for sample in sorted_passed[:10]:
        print(
            f"Sample {sample['id']} ({sample['type'].upper()}): Ratio {sample.get('ratio', 0):.2f}x"
        )
        print(f"  Output: {sample['compressed']}...")
        print()

    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print()

    retention_rate = stats["passed_all"] / stats["total_input"] * 100
    code_retention = (
        stats["code_passed"] / stats["code_samples"] * 100 if stats["code_samples"] > 0 else 0
    )
    nl_retention = stats["nl_passed"] / stats["nl_samples"] * 100 if stats["nl_samples"] > 0 else 0

    print(f"Overall retention:          {retention_rate:.1f}%")
    print(f"Code retention:             {code_retention:.1f}%")
    print(f"NL retention:               {nl_retention:.1f}%")
    print()

    if retention_rate > 70:
        print(f"✓ Good retention ({retention_rate:.1f}%)")
        print("  Your data quality is decent. Proceed with training.")
    elif retention_rate > 50:
        print(f"⚠ Moderate retention ({retention_rate:.1f}%)")
        print("  You lost some data but have enough to train.")
    else:
        print(f"❌ Low retention ({retention_rate:.1f}%)")
        print("  Most of your data failed validation. Consider regenerating.")

    print()

    if code_retention < 80:
        print(f"⚠ Code retention is low ({code_retention:.1f}%)")
        print("  Many code samples are being filtered. Review code validation rules.")

    if nl_retention < 60:
        print(f"⚠ NL retention is low ({nl_retention:.1f}%)")
        print("  Many NL samples are being filtered. Review NL validation rules.")

    print()


def analyze_symbol_distribution(sanitized_path: Path):
    """Analyze symbol usage in sanitized data, split by code vs NL."""
    print("=" * 80)
    print("SYMBOL DISTRIBUTION IN SANITIZED DATA")
    print("=" * 80)
    print()

    with open(sanitized_path, encoding="utf-8") as f:
        data = [json.loads(line) for line in f if line.strip()]

    code_symbol_counts = defaultdict(int)
    nl_symbol_counts = defaultdict(int)
    code_total = 0
    nl_total = 0

    for sample in data:
        verbose, compressed = extract_verbose_compressed(sample)
        is_code = is_code_sample(verbose)

        if is_code:
            code_total += 1
            for symbol in SYMBOLS:
                if symbol in compressed:
                    code_symbol_counts[symbol] += 1
        else:
            nl_total += 1
            for symbol in SYMBOLS:
                if symbol in compressed:
                    nl_symbol_counts[symbol] += 1

    print(f"Total sanitized samples: {len(data)}")
    print(f"  Code: {code_total}")
    print(f"  NL:   {nl_total}")
    print()

    print("CODE SAMPLES:")
    for symbol in SYMBOLS:
        count = code_symbol_counts[symbol]
        pct = (count / code_total * 100) if code_total > 0 else 0
        print(f"  {symbol} : {count:4d} / {code_total} ({pct:5.1f}%)")
    print()

    print("NL SAMPLES:")
    for symbol in SYMBOLS:
        count = nl_symbol_counts[symbol]
        pct = (count / nl_total * 100) if nl_total > 0 else 0
        print(f"  {symbol} : {count:4d} / {nl_total} ({pct:5.1f}%)")
    print()

    print("Expected behavior after Option 2 sanitization:")
    print("  CODE: High @ usage (decorators), high : usage (types)")
    print("  NL:   Balanced symbol usage, high @ and ∵ (enforced by Rule D)")
    print()


# ============================================================================
# MAIN EXECUTION
# ============================================================================


def main():
    # File paths
    input_path = Path("data/training/train.jsonl")
    output_path = Path("data/training/sanitized_train.jsonl")

    print("\n" + "=" * 80)
    print("DATA SANITIZATION - OPTION 2 (Code vs NL Split)")
    print("=" * 80)
    print()
    print("Filtering rules:")
    print("  UNIVERSAL (Code + NL):")
    print("    Rule A: Remove samples with compression ratio < 1.0")
    print("    Rule B: Remove samples with orphaned symbols")
    print("  NL ONLY:")
    print("    Rule C: Remove samples that lost negation")
    print("    Rule D: Remove samples missing @ or ∵ (strict keywords)")
    print()

    # Run sanitization
    stats = sanitize_dataset(input_path, output_path)

    # Print statistics
    print_statistics(stats)

    # Analyze symbol distribution
    analyze_symbol_distribution(output_path)

    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Review the sanitized data:")
    print(f"   cat {output_path} | head -n 5")
    print()
    print("2. Train BASELINE model on original data:")
    print("   (Use all 1759 samples from train.jsonl)")
    print()
    print("3. Train SANITIZED model on cleaned data:")
    print(f"   (Use {stats['passed_all']} samples from sanitized_train.jsonl)")
    print()
    print("4. Compare models:")
    print("   - Compression ratio")
    print("   - Symbol usage (@, ∵)")
    print("   - QA equivalence (most important)")
    print()


if __name__ == "__main__":
    main()
