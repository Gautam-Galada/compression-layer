#!/usr/bin/env python3
"""Tinker Cloud Training Log Parser and Visualizer.

Parses metrics.jsonl from Tinker training runs and generates
training curve visualizations including loss curves, throughput,
and key training events (epoch boundaries, best val loss, early stopping).
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class TinkerTrainingMetrics:
    """Container for parsed Tinker training metrics."""

    # Training metrics
    train_steps: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    train_losses_total: list[float] = field(default_factory=list)  # raw sum-reduced
    tokens_per_sec: list[float] = field(default_factory=list)
    train_epochs: list[int] = field(default_factory=list)  # epoch for each train step

    # Validation metrics
    val_steps: list[int] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_epochs: list[int] = field(default_factory=list)

    # Events
    early_stop_step: int | None = None
    best_val_loss: float | None = None

    # Derived
    epoch_boundaries: list[int] = field(default_factory=list)  # steps where epochs change


class TinkerMetricsParser:
    """Parse metrics.jsonl from Tinker training runs."""

    def parse(self, metrics_path: Path) -> TinkerTrainingMetrics:
        """Parse a metrics.jsonl file into TinkerTrainingMetrics.

        Args:
            metrics_path: Path to the metrics.jsonl file.

        Returns:
            TinkerTrainingMetrics with all parsed data.
        """
        metrics = TinkerTrainingMetrics()
        prev_epoch = None

        with open(metrics_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                record_type = record.get("type")

                if record_type == "train":
                    step = record["step"]
                    epoch = record.get("epoch", 1)
                    metrics.train_steps.append(step)
                    metrics.train_losses.append(record["train_loss"])
                    if "train_loss_total" in record:
                        metrics.train_losses_total.append(record["train_loss_total"])
                    metrics.tokens_per_sec.append(record.get("tokens_per_sec", 0.0))
                    metrics.train_epochs.append(epoch)

                    # Detect epoch boundary
                    if prev_epoch is not None and epoch != prev_epoch:
                        metrics.epoch_boundaries.append(step)
                    prev_epoch = epoch

                elif record_type == "val":
                    metrics.val_steps.append(record["step"])
                    metrics.val_losses.append(record["val_loss"])
                    metrics.val_epochs.append(record.get("epoch", 1))

                elif record_type == "early_stop":
                    metrics.early_stop_step = record.get("step")
                    metrics.best_val_loss = record.get("best_val_loss")

        # Find best val loss if not set by early stopping
        if metrics.val_losses and metrics.best_val_loss is None:
            best_idx = int(np.argmin(metrics.val_losses))
            metrics.best_val_loss = metrics.val_losses[best_idx]

        return metrics


class TinkerVisualizer:
    """Generate training curve visualizations from Tinker metrics."""

    def __init__(self, dpi: int = 150, ema_alpha: float = 0.1):
        self.dpi = dpi
        self.ema_alpha = ema_alpha

    def _compute_ema(self, values: list[float], alpha: float) -> list[float]:
        """Compute exponential moving average.

        Args:
            values: Raw values to smooth.
            alpha: Smoothing factor (0-1). Higher = more responsive to recent values.

        Returns:
            List of EMA-smoothed values, same length as input.
        """
        ema = []
        current = values[0] if values else 0.0
        for v in values:
            current = alpha * v + (1 - alpha) * current
            ema.append(current)
        return ema

    def plot(self, metrics: TinkerTrainingMetrics, output_path: Path) -> None:
        """Generate and save the training curves plot.

        Creates a two-subplot figure:
          - Top: Loss curves (train scatter + EMA, val line, epoch boundaries, best val, early stop)
          - Bottom: Throughput (tokens/sec raw + EMA)

        Args:
            metrics: Parsed training metrics.
            output_path: Path to save the output PNG.
        """
        fig, (ax_loss, ax_throughput) = plt.subplots(
            2,
            1,
            figsize=(14, 10),
            height_ratios=[3, 1],
            sharex=True,
        )
        fig.suptitle("Tinker Training Curves", fontsize=14, fontweight="bold")

        # === Top: Loss curves ===
        # Raw train loss (faint scatter)
        if metrics.train_steps and metrics.train_losses:
            ax_loss.scatter(
                metrics.train_steps,
                metrics.train_losses,
                s=4,
                alpha=0.15,
                color="tab:blue",
                label="_nolegend_",
            )
            # EMA smoothed train loss
            ema_losses = self._compute_ema(metrics.train_losses, self.ema_alpha)
            ax_loss.plot(
                metrics.train_steps,
                ema_losses,
                color="tab:blue",
                linewidth=1.5,
                label="Train Loss (EMA)",
            )

        # Val loss
        if metrics.val_steps and metrics.val_losses:
            ax_loss.plot(
                metrics.val_steps,
                metrics.val_losses,
                color="tab:orange",
                linewidth=1.5,
                marker="o",
                markersize=3,
                label="Val Loss",
            )

            # Best val loss star
            if metrics.best_val_loss is not None:
                # Find the closest val loss to best_val_loss (handles rounding differences)
                try:
                    best_idx = metrics.val_losses.index(metrics.best_val_loss)
                except ValueError:
                    # best_val_loss not exact match; find closest
                    best_idx = int(
                        np.argmin([abs(v - metrics.best_val_loss) for v in metrics.val_losses])
                    )
                ax_loss.plot(
                    metrics.val_steps[best_idx],
                    metrics.best_val_loss,
                    marker="*",
                    color="gold",
                    markersize=15,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    zorder=5,
                    label=f"Best Val Loss: {metrics.best_val_loss:.2f}",
                )

        # Epoch boundaries
        for boundary_step in metrics.epoch_boundaries:
            ax_loss.axvline(x=boundary_step, color="gray", linestyle="--", alpha=0.5, linewidth=1)
            ax_loss.text(
                boundary_step,
                ax_loss.get_ylim()[1] * 0.95,
                " Epoch",
                fontsize=8,
                color="gray",
                va="top",
            )

        # Early stopping marker
        if metrics.early_stop_step is not None:
            ax_loss.axvline(
                x=metrics.early_stop_step,
                color="red",
                linestyle="-.",
                alpha=0.7,
                linewidth=1.5,
            )
            ax_loss.text(
                metrics.early_stop_step,
                ax_loss.get_ylim()[1] * 0.85,
                " Early Stop",
                fontsize=8,
                color="red",
                va="top",
            )

        ax_loss.set_ylabel("Loss")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_title("Training & Validation Loss")

        # === Bottom: Throughput ===
        if metrics.train_steps and metrics.tokens_per_sec:
            ax_throughput.plot(
                metrics.train_steps,
                metrics.tokens_per_sec,
                color="tab:green",
                linewidth=0.8,
                alpha=0.5,
            )
            # EMA smoothed
            ema_throughput = self._compute_ema(metrics.tokens_per_sec, self.ema_alpha)
            ax_throughput.plot(
                metrics.train_steps,
                ema_throughput,
                color="tab:green",
                linewidth=1.5,
                label="Tokens/sec (EMA)",
            )
            ax_throughput.legend(loc="upper right")

        for boundary_step in metrics.epoch_boundaries:
            ax_throughput.axvline(
                x=boundary_step, color="gray", linestyle="--", alpha=0.5, linewidth=1
            )

        ax_throughput.set_xlabel("Training Step")
        ax_throughput.set_ylabel("Tokens/sec")
        ax_throughput.grid(True, alpha=0.3)
        ax_throughput.set_title("Throughput")

        plt.tight_layout()
        fig.savefig(output_path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved training curves to {output_path}")


def main() -> None:
    """CLI entry point for Tinker training visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize Tinker training metrics from metrics.jsonl"
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("models/adapters/tinker/metrics.jsonl"),
        help="Path to metrics.jsonl (default: models/adapters/tinker/metrics.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: <metrics-dir>/training_curves.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for output (default: 150)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.1,
        help="EMA smoothing factor (0-1, higher = more responsive) (default: 0.1)",
    )
    args = parser.parse_args()

    if not args.metrics.exists():
        print(f"Error: metrics file not found: {args.metrics}", file=sys.stderr)
        sys.exit(1)

    output = args.output or args.metrics.parent / "training_curves.png"

    parser_obj = TinkerMetricsParser()
    metrics = parser_obj.parse(args.metrics)

    print(f"Parsed {len(metrics.train_steps)} train entries, {len(metrics.val_steps)} val entries")
    if metrics.best_val_loss is not None:
        print(f"Best val loss: {metrics.best_val_loss:.4f}")
    if metrics.epoch_boundaries:
        print(f"Epoch boundaries at steps: {metrics.epoch_boundaries}")
    if metrics.early_stop_step:
        print(f"Early stopping at step: {metrics.early_stop_step}")

    visualizer = TinkerVisualizer(dpi=args.dpi, ema_alpha=args.ema_alpha)
    visualizer.plot(metrics, output)


if __name__ == "__main__":
    main()
