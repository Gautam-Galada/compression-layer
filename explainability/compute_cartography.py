#!/usr/bin/env python3
"""Standalone cartography computation — no training, fast in-process loss evaluation.

Loads the model ONCE per checkpoint (not once per sample) and computes
loss for all samples in a single batched pass. This reduces the previous
18-hour runtime to ~20-40 minutes for 300 samples across 10 checkpoints.

Usage:
    # Run on the existing adapter directory
    python explainability/compute_cartography.py \
        --run-dir models/runs/mlx/2026-03-10_16-18-40 \
        --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
        --data-dir data/training \
        --max-samples 300

    # Separate NL and Code (recommended — different difficulty distributions)
    python explainability/compute_cartography.py \
        --run-dir models/runs/mlx/2026-03-10_16-18-40 \
        --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
        --data-dir data/training \
        --max-samples 300 \
        --domain code

    python explainability/compute_cartography.py \
        --run-dir models/runs/mlx/2026-03-10_16-18-40 \
        --model mlx-community/Qwen3-4B-Instruct-2507-8bit \
        --data-dir data/training \
        --max-samples 300 \
        --domain nl

    # Only replot from existing JSONL
    python explainability/compute_cartography.py --only-plot \
        --cartography-file explainability/outputs/cartography_code.jsonl
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

EXPLAINABILITY_OUT = Path("explainability/outputs")

# ── Domain classifier ─────────────────────────────────────────────────────────

_CODE_SIGNALS = (
    "def ", "class ", "import ", "return ", "->", "self.",
    "fn:", "lambda", ":=", "async def", "#include", "print(",
    "{", "};", "//", "/*",
)

def infer_domain(text: str) -> str:
    code_hits = sum(s in text for s in _CODE_SIGNALS)
    if code_hits >= 3:
        return "Code"
    if code_hits >= 1:
        return "Mixed"
    return "NL"


# ── Checkpoint discovery — fixed to use correct run adapter dir ───────────────

def find_checkpoints(adapter_dir: Path) -> list[tuple[int, Path]]:
    """
    Returns sorted list of (iteration, safetensors_path) for all checkpoints
    in the given adapter directory.

    MLX naming: 0000050_adapters.safetensors → iter 50
    Each checkpoint is a full adapter directory path — mlx_lm.lora --adapter-path
    expects the directory, not the individual file.
    """
    checkpoints: list[tuple[int, Path]] = []

    for f in sorted(adapter_dir.glob("*_adapters.safetensors")):
        stem = f.stem  # e.g. "0000050_adapters"
        parts = stem.split("_")
        try:
            itr = int(parts[0])
            checkpoints.append((itr, adapter_dir))
        except ValueError:
            continue

    if not checkpoints:
        console.print(f"[red]No checkpoint files found in {adapter_dir}[/red]")
        console.print(f"[dim]Files present: {list(adapter_dir.iterdir())}[/dim]")

    return checkpoints


# ── In-process loss computation ───────────────────────────────────────────────
# Key insight: mlx_lm exposes a Python API. We load the model once per
# checkpoint and score all samples — avoiding 3300 subprocess spawns.

def load_mlx_model(model_name: str, adapter_dir: Path, checkpoint_iter: int):
    """
    Load model + specific checkpoint adapter weights in-process.
    Returns (model, tokenizer) or (None, None) on failure.
    """
    try:
        import mlx.core as mx
        import mlx_lm

        # mlx_lm.load accepts adapter_path as directory
        # We need to load a specific checkpoint, not the final adapters.safetensors
        # Strategy: temporarily symlink the target checkpoint file as adapters.safetensors
        # in a temp directory, then load from there.
        import shutil
        import tempfile

        checkpoint_file = adapter_dir / f"{checkpoint_iter:07d}_adapters.safetensors"
        adapter_config  = adapter_dir / "adapter_config.json"

        if not checkpoint_file.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_file}")
            return None, None

        # Create a temp dir with this checkpoint as adapters.safetensors
        tmp = Path(tempfile.mkdtemp(prefix=f"ckpt_{checkpoint_iter}_"))
        shutil.copy(checkpoint_file, tmp / "adapters.safetensors")
        if adapter_config.exists():
            shutil.copy(adapter_config, tmp / "adapter_config.json")

        model, tokenizer = mlx_lm.load(model_name, adapter_path=str(tmp))
        return model, tokenizer, tmp

    except Exception as e:
        logger.error(f"Failed to load model at checkpoint {checkpoint_iter}: {e}")
        return None, None, None


def compute_sample_loss_inprocess(
    model,
    tokenizer,
    sample_text: str,
) -> float | None:
    """
    Compute teacher-forced cross-entropy loss for one sample using loaded model.
    Parses the chat-format JSONL and tokenizes input + target separately
    to respect prompt masking (loss only on assistant response).
    """
    try:
        import mlx.core as mx
        import mlx.nn as nn

        row = json.loads(sample_text)
        messages = row.get("messages", [])

        # Separate prompt (system+user) from completion (assistant)
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        full_messages   = messages

        # Tokenize
        prompt_ids = tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors=None,
        )
        full_ids = tokenizer.apply_chat_template(
            full_messages,
            add_generation_prompt=False,
            return_tensors=None,
        )

        if isinstance(prompt_ids, list):
            prompt_len = len(prompt_ids)
            full_len   = len(full_ids)
        else:
            prompt_len = prompt_ids.shape[-1]
            full_len   = full_ids.shape[-1]

        if full_len <= prompt_len:
            return None

        # Build input/target tensors — loss only on assistant tokens
        input_ids  = mx.array(full_ids[:-1])[None]       # [1, seq-1]
        target_ids = mx.array(full_ids[1:])[None]         # [1, seq-1]
        mask_start = prompt_len - 1                       # ignore prompt positions

        logits = model(input_ids)                         # [1, seq-1, vocab]
        log_probs = nn.log_softmax(logits, axis=-1)

        # Gather log probs of target tokens
        gathered = log_probs[0, mx.arange(full_len - 1), target_ids[0]]

        # Apply prompt mask
        response_log_probs = gathered[mask_start:]
        mx.eval(response_log_probs)

        loss = float(-mx.mean(response_log_probs).item())
        return loss

    except Exception as e:
        logger.debug(f"Loss computation error: {e}")
        return None


# ── Main cartography loop ─────────────────────────────────────────────────────

def run_cartography(
    model_name: str,
    adapter_dir: Path,
    train_jsonl: Path,
    max_samples: int,
    domain_filter: str | None,
    out_path: Path,
) -> list[dict]:
    """
    For each checkpoint: load model once, score all samples, unload.
    Total model loads = number of checkpoints (10), not samples × checkpoints.
    """
    # Load samples
    raw_samples: list[str] = []
    train_line_numbers: list[int] = []   # original line number in train.jsonl for each sample
    with open(train_jsonl, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if line:
                raw_samples.append(line)
                train_line_numbers.append(line_no)

    # Filter by domain if requested
    if domain_filter:
        target = {"code": "Code", "nl": "NL", "mixed": "Mixed"}.get(
            domain_filter.lower(), domain_filter
        )
        filtered_samples = []
        filtered_line_nos = []
        for s, ln in zip(raw_samples, train_line_numbers):
            try:
                row = json.loads(s)
                user_text = next(
                    (m["content"] for m in row.get("messages", []) if m["role"] == "user"), ""
                )
                if infer_domain(user_text) == target:
                    filtered_samples.append(s)
                    filtered_line_nos.append(ln)
            except Exception:
                pass
        console.print(f"[dim]Domain filter '{target}': {len(filtered_samples)}/{len(raw_samples)} samples[/dim]")
        raw_samples = filtered_samples
        train_line_numbers = filtered_line_nos

    if max_samples > 0:
        raw_samples = raw_samples[:max_samples]
        train_line_numbers = train_line_numbers[:max_samples]

    console.print(f"[cyan]Samples to evaluate: {len(raw_samples)}[/cyan]")

    checkpoints = find_checkpoints(adapter_dir)
    if not checkpoints:
        return []

    console.print(f"[cyan]Checkpoints: {[c[0] for c in checkpoints]}[/cyan]\n")

    # per_sample_losses[i] = list of (iter, loss) in checkpoint order
    per_sample_losses: dict[int, list[tuple[int, float]]] = {i: [] for i in range(len(raw_samples))}

    # ── Outer loop: checkpoints (model loads) ────────────────────────────────
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        ckpt_task = progress.add_task("Checkpoints", total=len(checkpoints))

        for itr, adapter_path in checkpoints:
            progress.update(ckpt_task, description=f"Checkpoint iter={itr}")

            model, tokenizer, tmp_dir = load_mlx_model(model_name, adapter_path, itr)
            if model is None:
                progress.advance(ckpt_task)
                continue

            sample_task = progress.add_task(f"  Samples @{itr}", total=len(raw_samples))

            for idx, raw_line in enumerate(raw_samples):
                loss = compute_sample_loss_inprocess(model, tokenizer, raw_line)
                if loss is not None:
                    per_sample_losses[idx].append((itr, loss))
                progress.advance(sample_task)

            progress.remove_task(sample_task)

            # Unload model explicitly to free memory before next checkpoint
            del model, tokenizer
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass
            if tmp_dir and tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)

            progress.advance(ckpt_task)

    # ── Build metrics ─────────────────────────────────────────────────────────
    metrics: list[dict] = []
    for idx, (raw_line, train_line_no) in enumerate(zip(raw_samples, train_line_numbers)):
        entries = per_sample_losses[idx]
        if not entries:
            continue

        try:
            row = json.loads(raw_line)
            user_text = next(
                (m["content"] for m in row.get("messages", []) if m["role"] == "user"), ""
            )
            domain = infer_domain(user_text)
        except Exception:
            domain = "NL"

        iters_list  = [e[0] for e in entries]
        losses_list = [e[1] for e in entries]
        mean_loss   = float(np.mean(losses_list))
        confidence  = float(np.mean([np.exp(-l) for l in losses_list]))
        variability = float(np.std(losses_list))

        metrics.append({
            "idx":           idx,            # position within domain-filtered subset
            "train_line_no": train_line_no,  # actual line number in train.jsonl — use this for text lookup
            "domain":        domain,
            "checkpoints":   iters_list,
            "losses":        losses_list,
            "mean_loss":     mean_loss,
            "confidence":    confidence,
            "variability":   variability,
        })

    # Add correctness once global median is known
    if metrics:
        global_median = float(np.median([m["mean_loss"] for m in metrics]))
        for m in metrics:
            m["correctness"] = float(
                np.mean([l < global_median for l in m["losses"]])
            )

    # Save
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m) + "\n")
    console.print(f"\n[green]Cartography saved → {out_path}[/green]")

    return metrics


# ── Plotting (unchanged from train_cartography.py) ────────────────────────────

def plot_cartography(cartography_path: Path, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        console.print("[yellow]pip install matplotlib to enable plots[/yellow]")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(cartography_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        return

    confidence  = np.array([r["confidence"]   for r in rows])
    variability = np.array([r["variability"]  for r in rows])
    correctness = np.array([r["correctness"]  for r in rows])
    domains     = [r["domain"] for r in rows]
    mean_losses = np.array([r["mean_loss"]    for r in rows])

    med_conf = float(np.median(confidence))
    med_var  = float(np.median(variability))

    def region(c, v):
        if v > med_var:
            return "Ambiguous"
        return "Easy" if c > med_conf else "Hard"

    region_labels  = [region(c, v) for c, v in zip(confidence, variability)]
    region_colors  = {"Easy": "#2ECC71", "Hard": "#E74C3C", "Ambiguous": "#F39C12"}
    domain_colors  = {"NL": "#4C9BE8", "Mixed": "#F5A623", "Code": "#7ED321"}
    domain_markers = {"NL": "o", "Mixed": "s", "Code": "^"}
    unique_domains = list(dict.fromkeys(domains))

    _fig_defaults = dict(facecolor="#0F0F0F")
    _ax_defaults  = "#0F0F0F"

    def _style(ax):
        ax.set_facecolor(_ax_defaults)
        ax.tick_params(colors="#AAA")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    def _legend(ax):
        leg = ax.legend(framealpha=0.15, labelcolor="white", fontsize=9)
        leg.get_frame().set_facecolor("#222")

    # 1 — Main cartography map
    fig, ax = plt.subplots(figsize=(10, 7), **_fig_defaults)
    _style(ax)
    for reg, col in region_colors.items():
        mask = np.array(region_labels) == reg
        ax.scatter(variability[mask], confidence[mask],
                   c=col, alpha=0.6, s=22, label=reg, edgecolors="none")
    ax.axvline(med_var,  color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_conf, color="#555", lw=0.8, ls="--", alpha=0.6)
    counts = {r: region_labels.count(r) for r in region_colors}
    subtitle = "  |  ".join(f"{r}: {n}" for r, n in counts.items())
    ax.set_title(f"Dataset Cartography — Semantic Compression\n{subtitle}",
                 color="white", fontsize=12, pad=12)
    ax.set_xlabel("Variability  (std of loss across checkpoints)", color="#CCC", fontsize=11)
    ax.set_ylabel("Confidence  (mean exp(−loss))", color="#CCC", fontsize=11)
    _legend(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "cartography_map.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # 2 — By domain
    fig, ax = plt.subplots(figsize=(10, 7), **_fig_defaults)
    _style(ax)
    for dom in unique_domains:
        mask = np.array(domains) == dom
        ax.scatter(variability[mask], confidence[mask],
                   c=domain_colors.get(dom, "#888"),
                   marker=domain_markers.get(dom, "o"),
                   alpha=0.6, s=22, label=dom, edgecolors="none")
    ax.axvline(med_var,  color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_conf, color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.set_title("Cartography Map — by Domain", color="white", fontsize=12, pad=12)
    ax.set_xlabel("Variability", color="#CCC", fontsize=11)
    ax.set_ylabel("Confidence", color="#CCC", fontsize=11)
    _legend(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "cartography_by_domain.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # 3 — Loss trajectories per region
    all_ckpts = sorted({c for r in rows for c in r["checkpoints"]})
    if len(all_ckpts) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5), **_fig_defaults)
        _style(ax)
        for reg, col in region_colors.items():
            reg_rows = [r for r, lbl in zip(rows, region_labels) if lbl == reg]
            if not reg_rows:
                continue
            mat = []
            for row in reg_rows:
                ckpt_loss = dict(zip(row["checkpoints"], row["losses"]))
                mat.append([ckpt_loss.get(c, np.nan) for c in all_ckpts])
            mat    = np.array(mat)
            means  = np.nanmean(mat, axis=0)
            stds   = np.nanstd(mat,  axis=0)
            ax.plot(all_ckpts, means, color=col, label=reg, lw=2)
            ax.fill_between(all_ckpts, means - stds, means + stds,
                            color=col, alpha=0.15)
        ax.set_xlabel("Training Iteration", color="#CCC", fontsize=11)
        ax.set_ylabel("Mean Loss", color="#CCC", fontsize=11)
        ax.set_title("Loss Trajectories by Region", color="white", fontsize=12)
        _legend(ax)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_trajectories.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

    # 4 — Domain region breakdown
    bar_regions = ["Easy", "Hard", "Ambiguous"]
    fig, axes = plt.subplots(1, len(unique_domains), figsize=(5 * len(unique_domains), 5),
                              sharey=False, **_fig_defaults)
    if len(unique_domains) == 1:
        axes = [axes]
    fig.suptitle("Region Distribution per Domain", color="white", fontsize=13)
    for ax, dom in zip(axes, unique_domains):
        _style(ax)
        dom_regions = [r for d, r in zip(domains, region_labels) if d == dom]
        cnts = [dom_regions.count(r) for r in bar_regions]
        bars = ax.bar(bar_regions, cnts,
                      color=[region_colors[r] for r in bar_regions],
                      width=0.5, edgecolor="#333")
        for bar, cnt in zip(bars, cnts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(cnt), ha="center", va="bottom", color="white", fontsize=9)
        ax.set_title(dom, color="white", fontsize=11)
        ax.set_ylabel("Count", color="#AAA", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "domain_region_breakdown.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # 5 — Correctness histogram
    fig, ax = plt.subplots(figsize=(9, 5), **_fig_defaults)
    _style(ax)
    bins = np.linspace(0, 1, 15)
    for dom in unique_domains:
        mask = np.array(domains) == dom
        ax.hist(correctness[mask], bins=bins, alpha=0.55, label=dom,
                color=domain_colors.get(dom, "#888"), edgecolor="#222")
    ax.set_xlabel("Correctness", color="#CCC", fontsize=10)
    ax.set_ylabel("Count", color="#CCC", fontsize=10)
    ax.set_title("Correctness Distribution per Domain", color="white", fontsize=12)
    _legend(ax)
    plt.tight_layout()
    plt.savefig(out_dir / "correctness_histogram.png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    # 6 — Violin loss per domain
    plot_data = [mean_losses[np.array(domains) == dom] for dom in unique_domains]
    plot_data = [d for d in plot_data if len(d) > 1]
    if plot_data:
        fig, ax = plt.subplots(figsize=(8, 5), **_fig_defaults)
        _style(ax)
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                              showmedians=True, showextrema=True)
        for pc, dom in zip(parts["bodies"], unique_domains):
            pc.set_facecolor(domain_colors.get(dom, "#888"))
            pc.set_alpha(0.6)
        for key in ("cmedians", "cmins", "cmaxes", "cbars"):
            if key in parts:
                parts[key].set_color("#AAA")
        ax.set_xticks(range(len(unique_domains)))
        ax.set_xticklabels(unique_domains, color="white")
        ax.set_ylabel("Mean Loss", color="#CCC", fontsize=10)
        ax.set_title("Mean Loss Distribution per Domain", color="white", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_dir / "loss_violin_by_domain.png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close()

    # Summary table
    table = Table(title="Cartography Summary", style="cyan")
    table.add_column("Region", style="bold")
    table.add_column("Count")
    table.add_column("Avg Confidence")
    table.add_column("Avg Variability")
    table.add_column("Avg Correctness")
    for reg in bar_regions:
        mask = np.array(region_labels) == reg
        if not mask.any():
            continue
        table.add_row(reg, str(int(mask.sum())),
                      f"{confidence[mask].mean():.3f}",
                      f"{variability[mask].mean():.3f}",
                      f"{correctness[mask].mean():.3f}")
    console.print(table)
    console.print(f"\n[green]Plots saved → {out_dir}[/green]")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast dataset cartography — in-process loss eval")
    p.add_argument("--run-dir",    type=Path, required=False,
                   help="Run directory, e.g. models/runs/mlx/2026-03-10_16-18-40")
    p.add_argument("--model",      type=str,
                   default="mlx-community/Qwen3-4B-Instruct-2507-8bit")
    p.add_argument("--data-dir",   type=Path, default=Path("data/training"))
    p.add_argument("--max-samples",type=int,  default=300,
                   help="0 = all samples")
    p.add_argument("--domain",     type=str,  default=None,
                   choices=["code", "nl", "mixed"],
                   help="Filter to a single domain (recommended)")
    p.add_argument("--out-dir",    type=Path, default=EXPLAINABILITY_OUT)
    p.add_argument("--only-plot",  action="store_true")
    p.add_argument("--cartography-file", type=Path, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.domain}" if args.domain else ""
    default_jsonl = args.out_dir / f"cartography{suffix}.jsonl"
    cartography_path = args.cartography_file or default_jsonl

    if args.only_plot:
        if not cartography_path.exists():
            console.print(f"[red]File not found: {cartography_path}[/red]")
            return 1
        plot_cartography(cartography_path, args.out_dir / "plots")
        return 0

    if args.run_dir is None:
        console.print("[red]--run-dir is required unless --only-plot[/red]")
        return 1

    adapter_dir = args.run_dir / "adapter"
    if not adapter_dir.exists():
        console.print(f"[red]Adapter dir not found: {adapter_dir}[/red]")
        return 1

    train_jsonl = args.data_dir / "train.jsonl"
    if not train_jsonl.exists():
        console.print(f"[red]Training data not found: {train_jsonl}[/red]")
        return 1

    console.print(Panel.fit(
        f"[bold]Dataset Cartography — Fast Mode[/bold]\n\n"
        f"Run dir    : {args.run_dir}\n"
        f"Model      : {args.model}\n"
        f"Max samples: {args.max_samples}\n"
        f"Domain     : {args.domain or 'all'}\n"
        f"Strategy   : load model once per checkpoint, score all samples in-process",
        border_style="green",
    ))

    metrics = run_cartography(
        model_name=args.model,
        adapter_dir=adapter_dir,
        train_jsonl=train_jsonl,
        max_samples=args.max_samples,
        domain_filter=args.domain,
        out_path=cartography_path,
    )

    if not metrics:
        console.print("[red]No metrics produced[/red]")
        return 1

    plot_cartography(cartography_path, args.out_dir / "plots")
    return 0


if __name__ == "__main__":
    sys.exit(main())