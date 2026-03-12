#!/usr/bin/env python3
"""Dataset Cartography for Semantic Compression Training.

Runs MLX LoRA training with per-sample loss logging across checkpoints,
then computes cartography metrics (confidence, variability, correctness)
and produces diagnostic plots.

This script lives in explainability/ and imports all training primitives
from src.training (via scripts/train_local.py conventions) — it does NOT
duplicate any training logic.

Cartography workflow:
    1. Train with intermediate checkpoints (save_every controls granularity)
    2. After training, replay every training sample through each checkpoint
       using mlx_lm.generate with teacher-forced loss
    3. Compute per-sample: mean loss (confidence), std loss (variability),
       fraction of steps with loss < median (correctness)
    4. Plot cartography map + diagnostic breakdowns

Usage:
    python explainability/train_cartography.py

    # Custom iterations and checkpoints
    python explainability/train_cartography.py --iters 500 --save-every 50

    # Skip training (use existing run)
    python explainability/train_cartography.py --skip-training --run-dir models/runs/mlx/2026-03-01_20-53-41--iter-500

    # Only plot (cartography JSONL already computed)
    python explainability/train_cartography.py --only-plot --cartography-file explainability/outputs/cartography.jsonl
"""

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

# ── repo root on path ────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table

from src.training import (
    MLXTrainingConfig,
    check_mlx_available,
    train_local,
)
from src.training.train_mlx import load_config_from_yaml, prepare_run_paths, update_latest_symlink
from src.utils.config import get_settings

console = Console()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ── output directory (all artefacts land here) ────────────────────────────────
EXPLAINABILITY_OUT = Path("explainability/outputs")

# ── cartography metric names ──────────────────────────────────────────────────
# confidence  : mean of (1 - normalised_loss) across checkpoints  → high = easy
# variability : std of per-checkpoint loss                         → high = ambiguous
# correctness : fraction of checkpoints where loss < global median → high = easy


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SampleMetrics:
    idx: int
    domain: str          # "NL" | "Mixed" | "Code" — inferred from input text
    losses: list[float]  # one per checkpoint in checkpoint order
    checkpoints: list[int]  # iteration numbers

    @property
    def confidence(self) -> float:
        """Mean probability proxy: mean(exp(-loss)).  Higher = easier."""
        return float(np.mean([np.exp(-l) for l in self.losses]))

    @property
    def variability(self) -> float:
        """Std of per-checkpoint loss.  Higher = more ambiguous."""
        return float(np.std(self.losses))

    @property
    def mean_loss(self) -> float:
        return float(np.mean(self.losses))

    def correctness(self, global_median_loss: float) -> float:
        """Fraction of checkpoints where loss < global median."""
        return float(np.mean([l < global_median_loss for l in self.losses]))

    def to_dict(self, global_median_loss: float) -> dict:
        return {
            "idx": self.idx,
            "domain": self.domain,
            "losses": self.losses,
            "checkpoints": self.checkpoints,
            "confidence": self.confidence,
            "variability": self.variability,
            "mean_loss": self.mean_loss,
            "correctness": self.correctness(global_median_loss),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Domain classifier (lightweight, no model needed)
# ─────────────────────────────────────────────────────────────────────────────

_CODE_SIGNALS = (
    "def ", "class ", "import ", "return ", "->", "self.",
    "fn:", "lambda", ":=", "async def", "#include", "print(",
)


def infer_domain(text: str) -> str:
    code_hits = sum(s in text for s in _CODE_SIGNALS)
    if code_hits >= 3:
        return "Code"
    if code_hits >= 1:
        return "Mixed"
    return "NL"


# ─────────────────────────────────────────────────────────────────────────────
# Per-sample loss computation via mlx_lm
# ─────────────────────────────────────────────────────────────────────────────


def compute_loss_for_sample(
    model: str,
    adapter_path: Path,
    sample_text: str,
    tmp_dir: Path,
) -> float | None:
    """
    Compute teacher-forced loss for a single sample at a given checkpoint.

    Strategy: write a one-line JSONL to a temp dir, run mlx_lm.lora --test,
    parse the reported loss.  This reuses the exact same loss computation
    MLX uses during training — no approximation.

    Returns None on failure.
    """
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_file = tmp_dir / "test.jsonl"
    test_file.write_text(sample_text + "\n", encoding="utf-8")

    cmd = [
        "python", "-m", "mlx_lm.lora",
        "--model", model,
        "--adapter-path", str(adapter_path),
        "--data", str(tmp_dir),
        "--test",
        "--mask-prompt",
        "--batch-size", "1",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        # MLX prints: "Test loss X.XXX, Test ppl Y.YYY."
        for line in (result.stdout + result.stderr).splitlines():
            if "test loss" in line.lower():
                parts = line.lower().split("test loss")
                if len(parts) > 1:
                    token = parts[1].strip().split()[0].rstrip(",.")
                    try:
                        return float(token)
                    except ValueError:
                        pass
    except subprocess.TimeoutExpired:
        logger.warning("Loss computation timed out for sample")
    except Exception as e:
        logger.debug(f"Loss computation error: {e}")
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────


def find_checkpoints(adapter_path: Path) -> list[tuple[int, Path]]:
    """
    Return sorted list of (iteration, adapter_path) for all saved checkpoints.

    MLX saves adapters as:
        adapter_path/
            0000100_adapters.safetensors   (step 100)
            0000200_adapters.safetensors   (step 200)
            adapters.safetensors           (final)
    """
    checkpoints: list[tuple[int, Path]] = []

    for f in adapter_path.glob("*_adapters.safetensors"):
        try:
            itr = int(f.stem.split("_")[0])
            checkpoints.append((itr, adapter_path))
        except ValueError:
            pass

    # Final adapter (use total iters as key)
    final = adapter_path / "adapters.safetensors"
    if final.exists():
        # Infer final iter from config if present
        config_path = adapter_path / "adapter_config.json"
        final_iter = -1  # will be resolved below
        if config_path.exists():
            try:
                cfg = json.loads(config_path.read_text())
                final_iter = cfg.get("total_iters", -1)
            except Exception:
                pass
        checkpoints.append((final_iter, adapter_path))

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


# ─────────────────────────────────────────────────────────────────────────────
# Core cartography computation
# ─────────────────────────────────────────────────────────────────────────────


def compute_cartography(
    model: str,
    adapter_path: Path,
    train_jsonl: Path,
    max_samples: int,
    tmp_base: Path,
) -> list[SampleMetrics]:
    """
    For each training sample, compute loss at every checkpoint.

    Returns list of SampleMetrics, one per sample.
    """
    # Load training samples
    samples: list[str] = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(line)
    if max_samples > 0:
        samples = samples[:max_samples]

    console.print(f"\n[cyan]Computing cartography for {len(samples)} samples[/cyan]")

    checkpoints = find_checkpoints(adapter_path)
    if not checkpoints:
        console.print("[red]No checkpoints found. Did training use --save-every?[/red]")
        return []

    console.print(f"[dim]Checkpoints found: {[c[0] for c in checkpoints]}[/dim]")

    metrics: list[SampleMetrics] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        sample_task = progress.add_task("Samples", total=len(samples))

        for idx, raw_line in enumerate(samples):
            try:
                parsed = json.loads(raw_line)
                # Extract user content for domain inference
                user_text = ""
                for msg in parsed.get("messages", []):
                    if msg.get("role") == "user":
                        user_text = msg.get("content", "")
                        break
                domain = infer_domain(user_text)
            except Exception:
                domain = "NL"

            losses: list[float] = []
            ckpt_iters: list[int] = []

            for itr, ckpt_path in checkpoints:
                tmp_dir = tmp_base / f"sample_{idx:05d}_ckpt_{itr}"
                loss = compute_loss_for_sample(model, ckpt_path, raw_line, tmp_dir)
                if loss is not None:
                    losses.append(loss)
                    ckpt_iters.append(itr)

            if losses:
                metrics.append(SampleMetrics(
                    idx=idx,
                    domain=domain,
                    losses=losses,
                    checkpoints=ckpt_iters,
                ))

            progress.advance(sample_task)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Saving / loading cartography results
# ─────────────────────────────────────────────────────────────────────────────


def save_cartography(metrics: list[SampleMetrics], out_path: Path) -> None:
    global_median = float(np.median([m.mean_loss for m in metrics]))
    with open(out_path, "w", encoding="utf-8") as f:
        for m in metrics:
            f.write(json.dumps(m.to_dict(global_median)) + "\n")
    console.print(f"[green]Cartography saved → {out_path}[/green]")


def load_cartography(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────


def plot_cartography(cartography_path: Path, out_dir: Path) -> None:
    """Generate all cartography diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import LinearSegmentedColormap
    except ImportError:
        console.print("[yellow]matplotlib not installed — skipping plots. pip install matplotlib[/yellow]")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cartography(cartography_path)
    if not rows:
        console.print("[red]No cartography data to plot[/red]")
        return

    confidence   = np.array([r["confidence"]   for r in rows])
    variability  = np.array([r["variability"]   for r in rows])
    correctness  = np.array([r["correctness"]   for r in rows])
    domains      = [r["domain"] for r in rows]
    mean_losses  = np.array([r["mean_loss"]     for r in rows])

    domain_colors = {"NL": "#4C9BE8", "Mixed": "#F5A623", "Code": "#7ED321"}
    domain_markers = {"NL": "o", "Mixed": "s", "Code": "^"}

    # ── Region labels ─────────────────────────────────────────────────────────
    # Following Swayamdipta et al. (2020):
    #   Easy-to-learn   : high confidence, low variability
    #   Hard-to-learn   : low confidence,  low variability
    #   Ambiguous       : high variability (centre of confidence range)

    def region_label(conf: float, var: float, med_conf: float, med_var: float) -> str:
        if var > med_var:
            return "Ambiguous"
        return "Easy" if conf > med_conf else "Hard"

    med_conf = float(np.median(confidence))
    med_var  = float(np.median(variability))
    region_labels = [region_label(c, v, med_conf, med_var) for c, v in zip(confidence, variability)]
    region_colors = {"Easy": "#2ECC71", "Hard": "#E74C3C", "Ambiguous": "#F39C12"}

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 1 — Main cartography map (confidence vs variability, coloured by region)
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#0F0F0F")

    for region, color in region_colors.items():
        mask = np.array(region_labels) == region
        ax.scatter(
            variability[mask], confidence[mask],
            c=color, alpha=0.55, s=18, label=region,
            edgecolors="none",
        )

    # Quadrant lines
    ax.axvline(med_var,  color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_conf, color="#555", lw=0.8, ls="--", alpha=0.6)

    # Quadrant annotations
    _ann_kw = dict(fontsize=8, color="#888", alpha=0.7, style="italic")
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0,
            med_conf * 1.02, "← Easy-to-learn", **_ann_kw)
    ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] > 0 else 0,
            med_conf * 0.96, "← Hard-to-learn", **_ann_kw)

    ax.set_xlabel("Variability  (std of loss across checkpoints)", color="#CCC", fontsize=11)
    ax.set_ylabel("Confidence  (mean exp(−loss))", color="#CCC", fontsize=11)
    ax.set_title("Dataset Cartography — Semantic Compression Training Set",
                 color="white", fontsize=13, pad=12)
    ax.tick_params(colors="#AAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

    legend = ax.legend(framealpha=0.15, labelcolor="white", fontsize=9)
    legend.get_frame().set_facecolor("#222")

    # Sample counts in subtitle
    counts = {r: region_labels.count(r) for r in region_colors}
    subtitle = "  |  ".join(f"{r}: {n}" for r, n in counts.items())
    ax.set_title(f"Dataset Cartography — Semantic Compression Training Set\n{subtitle}",
                 color="white", fontsize=12, pad=12)

    plt.tight_layout()
    p1 = out_dir / "cartography_map.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"  [green]✓[/green] {p1}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 2 — Domain breakdown: region distribution per domain
    # ─────────────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
    fig.patch.set_facecolor("#0F0F0F")
    fig.suptitle("Region Distribution per Domain", color="white", fontsize=13)

    unique_domains = list(dict.fromkeys(domains))
    bar_regions = ["Easy", "Hard", "Ambiguous"]

    for ax, dom in zip(axes, unique_domains):
        ax.set_facecolor("#111")
        dom_regions = [r for d, r in zip(domains, region_labels) if d == dom]
        region_counts = [dom_regions.count(r) for r in bar_regions]
        bars = ax.bar(bar_regions, region_counts,
                      color=[region_colors[r] for r in bar_regions],
                      width=0.5, edgecolor="#333")
        for bar, cnt in zip(bars, region_counts):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5, str(cnt),
                    ha="center", va="bottom", color="white", fontsize=9)
        ax.set_title(dom, color="white", fontsize=11)
        ax.tick_params(colors="#AAA")
        ax.set_ylabel("Count", color="#AAA", fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    plt.tight_layout()
    p2 = out_dir / "domain_region_breakdown.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"  [green]✓[/green] {p2}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 3 — Loss trajectories across checkpoints (mean ± std per region)
    # ─────────────────────────────────────────────────────────────────────────
    # Group by region and compute mean loss per checkpoint
    all_ckpts = sorted({c for r in rows for c in r["checkpoints"]})
    if len(all_ckpts) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_facecolor("#0F0F0F")
        ax.set_facecolor("#111")

        for region, color in region_colors.items():
            region_rows = [r for r, lbl in zip(rows, region_labels) if lbl == region]
            if not region_rows:
                continue
            # Align losses to all_ckpts grid
            loss_matrix = []
            for row in region_rows:
                ckpt_to_loss = dict(zip(row["checkpoints"], row["losses"]))
                loss_matrix.append([ckpt_to_loss.get(c, np.nan) for c in all_ckpts])
            loss_matrix = np.array(loss_matrix)
            means = np.nanmean(loss_matrix, axis=0)
            stds  = np.nanstd(loss_matrix,  axis=0)
            ax.plot(all_ckpts, means, color=color, label=region, lw=2)
            ax.fill_between(all_ckpts, means - stds, means + stds,
                            color=color, alpha=0.15)

        ax.set_xlabel("Training Iteration", color="#CCC", fontsize=11)
        ax.set_ylabel("Mean Loss", color="#CCC", fontsize=11)
        ax.set_title("Loss Trajectories by Region Across Checkpoints",
                     color="white", fontsize=12)
        ax.tick_params(colors="#AAA")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")
        legend = ax.legend(framealpha=0.15, labelcolor="white", fontsize=9)
        legend.get_frame().set_facecolor("#222")
        plt.tight_layout()
        p3 = out_dir / "loss_trajectories.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close()
        console.print(f"  [green]✓[/green] {p3}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 4 — Confidence vs variability, coloured by domain
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#0F0F0F")

    for dom in unique_domains:
        mask = np.array(domains) == dom
        ax.scatter(
            variability[mask], confidence[mask],
            c=domain_colors.get(dom, "#888"),
            marker=domain_markers.get(dom, "o"),
            alpha=0.55, s=18, label=dom,
            edgecolors="none",
        )

    ax.axvline(med_var,  color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_conf, color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel("Variability", color="#CCC", fontsize=11)
    ax.set_ylabel("Confidence", color="#CCC", fontsize=11)
    ax.set_title("Cartography Map — by Domain", color="white", fontsize=12, pad=12)
    ax.tick_params(colors="#AAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    legend = ax.legend(framealpha=0.15, labelcolor="white", fontsize=9)
    legend.get_frame().set_facecolor("#222")
    plt.tight_layout()
    p4 = out_dir / "cartography_by_domain.png"
    plt.savefig(p4, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"  [green]✓[/green] {p4}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 5 — Correctness histogram per domain
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#111")

    bins = np.linspace(0, 1, 15)
    for dom in unique_domains:
        mask = np.array(domains) == dom
        ax.hist(correctness[mask], bins=bins,
                alpha=0.55, label=dom,
                color=domain_colors.get(dom, "#888"),
                edgecolor="#222")

    ax.set_xlabel("Correctness (fraction of checkpoints with loss < median)",
                  color="#CCC", fontsize=10)
    ax.set_ylabel("Count", color="#CCC", fontsize=10)
    ax.set_title("Correctness Distribution per Domain", color="white", fontsize=12)
    ax.tick_params(colors="#AAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    legend = ax.legend(framealpha=0.15, labelcolor="white", fontsize=9)
    legend.get_frame().set_facecolor("#222")
    plt.tight_layout()
    p5 = out_dir / "correctness_histogram.png"
    plt.savefig(p5, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"  [green]✓[/green] {p5}")

    # ─────────────────────────────────────────────────────────────────────────
    # Plot 6 — Mean loss distribution (violin) per domain
    # ─────────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0F0F0F")
    ax.set_facecolor("#111")

    plot_data = [mean_losses[np.array(domains) == dom] for dom in unique_domains]
    parts = ax.violinplot(plot_data, positions=range(len(unique_domains)),
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
    ax.tick_params(colors="#AAA")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.tight_layout()
    p6 = out_dir / "loss_violin_by_domain.png"
    plt.savefig(p6, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    console.print(f"  [green]✓[/green] {p6}")

    # ── Summary table ─────────────────────────────────────────────────────────
    table = Table(title="Cartography Summary", style="cyan")
    table.add_column("Region", style="bold")
    table.add_column("Count")
    table.add_column("Avg Confidence")
    table.add_column("Avg Variability")
    table.add_column("Avg Correctness")

    for region in bar_regions:
        mask = np.array(region_labels) == region
        if not mask.any():
            continue
        table.add_row(
            region,
            str(mask.sum()),
            f"{confidence[mask].mean():.3f}",
            f"{variability[mask].mean():.3f}",
            f"{correctness[mask].mean():.3f}",
        )
    console.print(table)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset Cartography for Semantic Compression",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Training control
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training; use existing run-dir")
    parser.add_argument("--only-plot", action="store_true",
                        help="Skip training + cartography computation; only re-plot")
    parser.add_argument("--run-dir", type=Path, default=None,
                        help="Existing run directory (required with --skip-training)")
    parser.add_argument("--cartography-file", type=Path, default=None,
                        help="Existing cartography JSONL (required with --only-plot)")

    # Training params (mirror train_local.py)
    parser.add_argument("--model", type=str,
                        default="mlx-community/Qwen3-4B-Instruct-2507-8bit")
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-layers", type=int, default=16)
    parser.add_argument("--save-every", type=int, default=100,
                        help="Save checkpoint every N iters (more = finer cartography)")
    parser.add_argument("--config", type=Path, default=None,
                        help="configs/training.yaml override")

    # Cartography params
    parser.add_argument("--max-samples", type=int, default=500,
                        help="Max training samples to compute cartography for (0 = all)")
    parser.add_argument("--data-dir", type=Path, default=Path("data/training"))
    parser.add_argument("--out-dir", type=Path, default=EXPLAINABILITY_OUT)

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cartography_path = args.cartography_file or (args.out_dir / "cartography.jsonl")

    # ── MODE: only plot ───────────────────────────────────────────────────────
    if args.only_plot:
        if not cartography_path.exists():
            console.print(f"[red]Cartography file not found: {cartography_path}[/red]")
            return 1
        console.print(Panel.fit("[bold]Plotting from existing cartography[/bold]",
                                border_style="cyan"))
        plot_cartography(cartography_path, args.out_dir / "plots")
        return 0

    # ── STEP 1: Training ──────────────────────────────────────────────────────
    adapter_path: Path

    if args.skip_training:
        if args.run_dir is None:
            console.print("[red]--run-dir required with --skip-training[/red]")
            return 1
        adapter_path = args.run_dir / "adapter"
        if not adapter_path.exists():
            console.print(f"[red]Adapter not found at {adapter_path}[/red]")
            return 1
        console.print(f"[dim]Skipping training — using adapter at {adapter_path}[/dim]")
    else:
        if not check_mlx_available():
            console.print("[red]MLX not available[/red]")
            return 1

        if args.config and args.config.exists():
            config = load_config_from_yaml(args.config)
        else:
            config = MLXTrainingConfig()

        config.model = args.model
        config.data_dir = args.data_dir
        config.iters = args.iters
        config.batch_size = args.batch_size
        config.learning_rate = args.learning_rate
        config.lora_rank = args.lora_rank
        config.lora_layers = args.lora_layers
        config.save_every = args.save_every  # critical: more checkpoints = richer cartography

        console.print(Panel.fit(
            f"[bold]Cartography Training Run[/bold]\n\n"
            f"Model : {config.model}\n"
            f"Iters : {config.iters}   Save every : {config.save_every}\n"
            f"Checkpoints expected : {config.iters // config.save_every}",
            border_style="green",
        ))

        result = train_local(config)
        if not result.success:
            console.print(f"[red]Training failed: {result.error}[/red]")
            return 1

        adapter_path = result.adapter_path
        console.print(f"[green]Training complete → {adapter_path}[/green]")

    # ── STEP 2: Cartography computation ───────────────────────────────────────
    train_jsonl = args.data_dir / "train.jsonl"
    if not train_jsonl.exists():
        console.print(f"[red]Train JSONL not found: {train_jsonl}[/red]")
        return 1

    tmp_base = args.out_dir / "_tmp_loss_eval"
    console.print(Panel.fit("[bold]Computing per-sample losses across checkpoints[/bold]",
                            border_style="cyan"))

    metrics = compute_cartography(
        model=args.model,
        adapter_path=adapter_path,
        train_jsonl=train_jsonl,
        max_samples=args.max_samples,
        tmp_base=tmp_base,
    )

    if not metrics:
        console.print("[red]No metrics computed — check checkpoints and logs[/red]")
        return 1

    save_cartography(metrics, cartography_path)

    # ── STEP 3: Plots ─────────────────────────────────────────────────────────
    console.print(Panel.fit("[bold]Generating diagnostic plots[/bold]", border_style="magenta"))
    plot_cartography(cartography_path, args.out_dir / "plots")

    console.print(Panel.fit(
        f"[green]✓ Cartography complete[/green]\n\n"
        f"JSONL  : {cartography_path}\n"
        f"Plots  : {args.out_dir / 'plots'}\n\n"
        f"Next: inspect Hard + Ambiguous samples to diagnose training data quality",
        border_style="green",
    ))

    return 0


if __name__ == "__main__":
    sys.exit(main())
