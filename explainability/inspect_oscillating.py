#!/usr/bin/env python3
"""Inspect oscillating NL samples from dataset cartography.

Pulls samples identified as oscillating (loss rises significantly mid-training,
indicating contradictory or ambiguous compression targets) and prints:
  - Original verbose input
  - Compressed assistant output
  - Loss trajectory + max rise
  - Consistency score (how stable the compression target looks)

This is the manual inspection step before deciding to filter or re-generate.

Usage:
    # Inspect all oscillating samples (prints to terminal, paginated)
    python explainability/inspect_oscillating.py \
        --cartography explainability/outputs/cartography_nl.jsonl \
        --data data/training/train.jsonl

    # Export to file for easier reading
    python explainability/inspect_oscillating.py \
        --cartography explainability/outputs/cartography_nl.jsonl \
        --data data/training/train.jsonl \
        --export explainability/outputs/oscillating_samples.txt

    # Only show the worst N (highest max rise)
    python explainability/inspect_oscillating.py \
        --cartography explainability/outputs/cartography_nl.jsonl \
        --data data/training/train.jsonl \
        --top 20

    # Also export indices for downstream filtering/re-generation
    python explainability/inspect_oscillating.py \
        --cartography explainability/outputs/cartography_nl.jsonl \
        --data data/training/train.jsonl \
        --export-indices explainability/outputs/oscillating_indices.json
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

console = Console(highlight=False)

# ── Oscillation detection ─────────────────────────────────────────────────────
# A sample is oscillating if:
#   1. Its loss rises by > MIN_RISE at any consecutive checkpoint pair, AND
#   2. It was generally improving (total drop > 0.05) — rules out flat samples
#      where any noise looks like a rise

MIN_RISE = 0.10   # minimum upward jump to count as oscillating


def max_rise(losses: list[float]) -> float:
    """Largest upward jump between consecutive checkpoints."""
    arr = np.array(losses)
    diffs = np.diff(arr)
    return float(diffs.max()) if len(diffs) > 0 else 0.0


def total_drop(losses: list[float]) -> float:
    return losses[0] - losses[-1]


def is_oscillating(losses: list[float]) -> bool:
    return max_rise(losses) > MIN_RISE and total_drop(losses) > 0.05


def trajectory_label(losses: list[float]) -> str:
    """Human-readable shape label."""
    mr = max_rise(losses)
    td = total_drop(losses)
    if not is_oscillating(losses):
        if td < 0.05:
            return "stuck"
        if losses[-1] < losses[0] * 0.7:
            return "improving"
        return "slow_decline"
    # Find where the spike is
    arr = np.array(losses)
    spike_pos = int(np.argmax(np.diff(arr))) + 1  # checkpoint index of spike
    return f"oscillating (spike at ckpt {spike_pos})"


# ── Text helpers ──────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 100, indent: str = "  ") -> str:
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(indent + line)
        else:
            wrapped.extend(
                textwrap.wrap(line, width=width, initial_indent=indent,
                              subsequent_indent=indent)
            )
    return "\n".join(wrapped)


def truncate(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, {len(text) - max_chars} chars remaining]"


def extract_messages(raw_line: str) -> tuple[str, str, str]:
    """
    Returns (system, user_input, assistant_compression) from a JSONL row.
    All three may be empty strings on parse failure.
    """
    try:
        row = json.loads(raw_line)
        messages = row.get("messages", [])
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user   = next((m["content"] for m in messages if m["role"] == "user"),   "")
        asst   = next((m["content"] for m in messages if m["role"] == "assistant"), "")
        # Strip "Compress:\n" prefix from user content if present
        if user.startswith("Compress:\n"):
            user = user[len("Compress:\n"):]
        return system, user, asst
    except Exception:
        return "", "", ""


def compression_ratio(user: str, asst: str) -> float:
    if not user:
        return 0.0
    return len(asst.split()) / max(len(user.split()), 1)


# ── Loss sparkline ────────────────────────────────────────────────────────────

_SPARK = " ▁▂▃▄▅▆▇█"

def sparkline(losses: list[float]) -> str:
    lo, hi = min(losses), max(losses)
    if hi == lo:
        return "─" * len(losses)
    chars = []
    for l in losses:
        bucket = int((l - lo) / (hi - lo) * (len(_SPARK) - 1))
        chars.append(_SPARK[bucket])
    return "".join(chars)


# ── Main inspection loop ──────────────────────────────────────────────────────

def load_cartography(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_training_sample(train_jsonl: Path, target_idx: int) -> str | None:
    """
    Load the raw JSONL line at position target_idx from training data.
    Uses the idx field in cartography to match — falls back to line number.
    """
    with open(train_jsonl, encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if line_num == target_idx:
                return line
    return None


def load_all_training_samples(train_jsonl: Path) -> list[str]:
    samples = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(line)
    return samples


def render_sample(
    cart_row: dict,
    raw_line: str | None,
    rank: int,
    total: int,
    export_lines: list[str] | None = None,
) -> None:
    """Render one sample to console (and optionally collect for export)."""

    idx           = cart_row["idx"]
    train_line_no = cart_row.get("train_line_no")
    losses        = cart_row["losses"]
    ckpts         = cart_row["checkpoints"]
    conf          = cart_row["confidence"]
    var           = cart_row["variability"]
    mr            = max_rise(losses)
    td            = total_drop(losses)
    shape         = trajectory_label(losses)
    line_ref = f"  train_line=[dim]{train_line_no}[/dim]" if train_line_no is not None else \
               "  [red](re-run compute_cartography — train_line_no missing)[/red]"

    # Loss table
    loss_pairs = "  ".join(
        f"[dim]{c}[/dim]:[bold {'red' if i > 0 and losses[i] > losses[i-1] else 'green'}]{l:.3f}[/bold {'red' if i > 0 and losses[i] > losses[i-1] else 'green'}]"
        for i, (c, l) in enumerate(zip(ckpts, losses))
    )

    spark = sparkline(losses)

    header = (
        f"[bold cyan]Sample {rank}/{total}[/bold cyan]  "
        f"idx=[yellow]{idx}[/yellow]{line_ref}  "
        f"conf=[cyan]{conf:.3f}[/cyan]  "
        f"var=[magenta]{var:.3f}[/magenta]  "
        f"max_rise=[red]{mr:.3f}[/red]  "
        f"total_drop=[green]{td:.3f}[/green]\n"
        f"shape: [italic]{shape}[/italic]\n"
        f"trajectory: {spark}\n"
        f"{loss_pairs}"
    )

    console.print(Rule(f"[cyan]#{rank}[/cyan]  idx={idx}  train_line={train_line_no}", style="dim"))
    console.print(header)

    if raw_line is None:
        console.print("[red]  ✗ Could not load training sample[/red]\n")
        return

    system, user, asst = extract_messages(raw_line)
    ratio = compression_ratio(user, asst)

    console.print(f"\n[bold]INPUT[/bold]  ({len(user.split())} words → {len(asst.split())} words, ratio={ratio:.2f}x)")
    console.print(Panel(
        truncate(user, 800),
        border_style="blue", padding=(0, 1),
    ))

    console.print(f"\n[bold]COMPRESSION[/bold]")
    console.print(Panel(
        truncate(asst, 600),
        border_style="green", padding=(0, 1),
    ))

    # Quick quality signals
    signals = []
    if ratio > 0.8:
        signals.append("[red]⚠ barely compressed (ratio > 0.8)[/red]")
    if ratio < 0.1:
        signals.append("[red]⚠ over-compressed (ratio < 0.1)[/red]")
    if len(asst.strip()) < 10:
        signals.append("[red]⚠ near-empty compression[/red]")
    if asst.strip() == user.strip():
        signals.append("[red]⚠ compression identical to input[/red]")
    if not signals:
        signals.append("[dim]no obvious quality flags[/dim]")
    console.print("Quality signals: " + "  ".join(signals))
    console.print()

    # Collect for export
    if export_lines is not None:
        export_lines.append("=" * 80)
        export_lines.append(f"Sample {rank}/{total}  idx={idx}  conf={conf:.3f}  var={var:.3f}  max_rise={mr:.3f}  total_drop={td:.3f}")
        export_lines.append(f"Shape: {shape}")
        export_lines.append(f"Trajectory: {spark}")
        export_lines.append(f"Losses: {[round(l,3) for l in losses]}")
        export_lines.append(f"Checkpoints: {ckpts}")
        export_lines.append("")
        export_lines.append(f"INPUT ({len(user.split())} words):")
        export_lines.append(truncate(user, 1000))
        export_lines.append("")
        export_lines.append(f"COMPRESSION ({len(asst.split())} words, ratio={ratio:.2f}x):")
        export_lines.append(truncate(asst, 800))
        export_lines.append("")
        if signals:
            export_lines.append("Quality flags: " + "  ".join(
                sig.replace("[red]", "").replace("[/red]", "")
                .replace("[dim]", "").replace("[/dim]", "")
                for sig in signals
            ))
        export_lines.append("")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Inspect oscillating NL samples from cartography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--cartography", type=Path,
                   default=Path("explainability/outputs/cartography_nl.jsonl"),
                   help="NL cartography JSONL")
    p.add_argument("--data", type=Path,
                   default=Path("data/training/train.jsonl"),
                   help="Training data JSONL")
    p.add_argument("--top", type=int, default=0,
                   help="Only show top N by max_rise (0 = all)")
    p.add_argument("--min-rise", type=float, default=MIN_RISE,
                   help=f"Min upward jump to count as oscillating (default {MIN_RISE})")
    p.add_argument("--export", type=Path, default=None,
                   help="Export human-readable report to this file")
    p.add_argument("--export-indices", type=Path, default=None,
                   help="Export oscillating indices as JSON list")
    p.add_argument("--no-paginate", action="store_true",
                   help="Don't pause between samples (useful with --export)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # ── Load cartography ──────────────────────────────────────────────────────
    if not args.cartography.exists():
        console.print(f"[red]Cartography file not found: {args.cartography}[/red]")
        return 1

    cart_rows = load_cartography(args.cartography)
    console.print(f"[dim]Loaded {len(cart_rows)} cartography rows[/dim]")

    # ── Filter oscillating ────────────────────────────────────────────────────
    global MIN_RISE
    MIN_RISE = args.min_rise

    oscillating = [
        r for r in cart_rows
        if is_oscillating(r["losses"])
    ]
    oscillating.sort(key=lambda r: max_rise(r["losses"]), reverse=True)

    if args.top > 0:
        oscillating = oscillating[:args.top]

    console.print(Panel.fit(
        f"[bold]Oscillating NL Sample Inspection[/bold]\n\n"
        f"Cartography : {args.cartography}\n"
        f"Training data: {args.data}\n"
        f"Min rise threshold: {MIN_RISE}\n"
        f"Oscillating samples found: [red]{len(oscillating)}[/red]\n"
        f"Showing: {len(oscillating)}",
        border_style="cyan",
    ))

    if not oscillating:
        console.print("[green]No oscillating samples found at this threshold.[/green]")
        return 0

    # ── Export indices ────────────────────────────────────────────────────────
    if args.export_indices:
        args.export_indices.parent.mkdir(parents=True, exist_ok=True)
        indices = sorted(r["idx"] for r in oscillating)
        args.export_indices.write_text(json.dumps({
            "oscillating_indices": indices,
            "count": len(indices),
            "min_rise_threshold": MIN_RISE,
            "cartography_source": str(args.cartography),
        }, indent=2))
        console.print(f"[green]Indices saved → {args.export_indices}[/green]")

    # ── Load training data ────────────────────────────────────────────────────
    if not args.data.exists():
        console.print(f"[red]Training data not found: {args.data}[/red]")
        console.print("[yellow]Will show cartography stats only (no text)[/yellow]")
        samples = []
    else:
        console.print(f"[dim]Loading training samples...[/dim]")
        samples = load_all_training_samples(args.data)
        console.print(f"[dim]Loaded {len(samples)} training samples[/dim]\n")

    # ── Render each sample ────────────────────────────────────────────────────
    export_lines: list[str] | None = [] if args.export else None

    for rank, cart_row in enumerate(oscillating, 1):
        idx          = cart_row["idx"]
        train_line_no = cart_row.get("train_line_no", idx)  # fall back to idx for old JSONL files
        raw_line = samples[train_line_no] if train_line_no < len(samples) else None

        render_sample(cart_row, raw_line, rank, len(oscillating), export_lines)

        if not args.no_paginate and args.export is None and rank < len(oscillating):
            try:
                inp = console.input(
                    f"[dim]Press Enter for next, 'q' to quit, 's' to skip to summary: [/dim]"
                ).strip().lower()
                if inp == "q":
                    break
                if inp == "s":
                    break
            except (KeyboardInterrupt, EOFError):
                break

    # ── Summary table ─────────────────────────────────────────────────────────
    console.print(Rule("[bold]Summary[/bold]", style="cyan"))
    table = Table(style="cyan", show_lines=True)
    table.add_column("idx",       style="yellow", no_wrap=True)
    table.add_column("conf",      justify="right")
    table.add_column("var",       justify="right")
    table.add_column("max_rise",  justify="right", style="red")
    table.add_column("total_drop",justify="right", style="green")
    table.add_column("shape",     style="dim")
    table.add_column("spark")

    for r in oscillating:
        table.add_row(
            str(r["idx"]),
            f"{r['confidence']:.3f}",
            f"{r['variability']:.3f}",
            f"{max_rise(r['losses']):.3f}",
            f"{total_drop(r['losses']):.3f}",
            trajectory_label(r["losses"]),
            sparkline(r["losses"]),
        )
    console.print(table)

    # ── Export report ─────────────────────────────────────────────────────────
    if args.export and export_lines is not None:
        args.export.parent.mkdir(parents=True, exist_ok=True)
        args.export.write_text("\n".join(export_lines), encoding="utf-8")
        console.print(f"\n[green]Report exported → {args.export}[/green]")

    console.print(f"\n[bold]{len(oscillating)} oscillating samples[/bold] identified.")
    console.print(
        "Next: review the compressions above and decide:\n"
        "  [red]remove[/red]  — input has no stable compression target\n"
        "  [yellow]re-generate[/yellow] — target was unlucky, re-run generator on this input\n"
        "  [green]keep[/green]     — oscillation was training interference, not a data problem"
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())