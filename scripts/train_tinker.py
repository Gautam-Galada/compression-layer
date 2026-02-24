#!/usr/bin/env python3
"""CLI script for Tinker cloud training.

Train a compression model on Tinker's cloud infrastructure.

Usage:
    # Train with default settings
    python scripts/train_tinker.py

    # Train with custom model
    python scripts/train_tinker.py --model Qwen/Qwen3-30B-A3B

    # Estimate cost before training
    python scripts/train_tinker.py --estimate-cost

    # Start training without waiting
    python scripts/train_tinker.py --no-wait

    # Check job status
    python scripts/train_tinker.py --status <job_id>
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.training import (
    TinkerClient,
    TinkerLoRAConfig,
    TinkerTrainingConfig,
    estimate_cost,
    train_on_tinker,
)
from src.training.train_tinker import load_config_from_yaml
from src.utils.config import get_settings

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train compression model on Tinker cloud",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Estimate cost before training
  python scripts/train_tinker.py --estimate-cost

  # Train with default Qwen3-8B
  python scripts/train_tinker.py

  # Train with MoE model (cost-efficient)
  python scripts/train_tinker.py --model Qwen/Qwen3-30B-A3B

  # Start training without waiting for completion
  python scripts/train_tinker.py --no-wait

  # Check status of existing job
  python scripts/train_tinker.py --status job_abc123
        """,
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--estimate-cost",
        action="store_true",
        help="Estimate training cost without starting job",
    )
    mode.add_argument(
        "--status",
        type=str,
        metavar="JOB_ID",
        help="Check status of existing training job",
    )

    # Paths
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to training config YAML (default: configs/training.yaml)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to training data directory (default: data/training/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to save adapter (default: models/adapters/tinker/)",
    )

    # Model
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model to fine-tune (default: Qwen/Qwen3-8B)",
    )

    # Training parameters (None means use YAML config value)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: from config, typically 2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (default: from config, typically 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate (default: from config, typically 2e-4)",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=None,
        help="LoRA rank (default: from config, typically 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=None,
        help="LoRA alpha (default: from config, typically 32)",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=None,
        help="LoRA dropout (default: from config, typically 0.05)",
    )
    parser.add_argument(
        "--log-interval-steps",
        type=int,
        default=None,
        help="How often to log train metrics (default: from config, typically 10)",
    )
    parser.add_argument(
        "--checkpoint-interval-steps",
        type=int,
        default=None,
        help="How often to save resumable checkpoints (default: from config, typically 250)",
    )
    parser.add_argument(
        "--eval-interval-steps",
        type=int,
        default=None,
        help="How often to run validation during an epoch (default: from config, typically 250)",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Stop after N evals without improvement; 0 disables (default: from config, typically 5)",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=None,
        help="Min improvement to reset patience (default: from config, typically 0.01)",
    )
    parser.add_argument(
        "--no-eval-at-epoch-end",
        action="store_true",
        help="Disable validation pass at epoch end",
    )
    parser.add_argument(
        "--checkpoint-ttl-seconds",
        type=int,
        default=None,
        help="Optional TTL for saved checkpoints in seconds",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable auto-resume from latest checkpoint in output directory",
    )

    # Job control
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Start training and return immediately (don't wait for completion)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Custom name for uploaded dataset",
    )

    return parser.parse_args()


def count_examples(data_dir: Path) -> int:
    """Count training examples."""
    train_file = data_dir / "train.jsonl"
    if not train_file.exists():
        return 0
    with open(train_file) as f:
        return sum(1 for _ in f)


def print_cost_estimate(config: TinkerTrainingConfig, num_examples: int) -> None:
    """Print cost estimate."""
    cost = estimate_cost(config, num_examples)

    table = Table(title="Cost Estimate")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Model", str(cost["model"]))
    table.add_row("Training Examples", f"{cost['examples']:,}")
    table.add_row("Epochs", str(cost["epochs"]))
    table.add_row("Total Tokens (est.)", f"{cost['total_tokens']:,}")
    table.add_row("Cost/1M Tokens", f"${cost['cost_per_million']:.2f}")
    table.add_row("", "")
    table.add_row("[bold]Estimated Cost[/bold]", f"[bold]${cost['estimated_cost_usd']:.2f}[/bold]")

    console.print(table)

    console.print(
        "\n[dim]Note: Actual cost may vary based on sequence lengths and model efficiency[/dim]"
    )


def print_config(config: TinkerTrainingConfig) -> None:
    """Print training configuration."""
    table = Table(title="Training Configuration")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Model", config.model)
    table.add_row("Dataset Path", str(config.dataset_path))
    table.add_row("Output Path", str(config.output_dir))
    table.add_row("Epochs", str(config.epochs))
    table.add_row("Batch Size", str(config.batch_size))
    table.add_row("Learning Rate", f"{config.learning_rate:.0e}")
    table.add_row("LoRA Rank", str(config.lora.rank))
    table.add_row("LoRA Alpha", str(config.lora.alpha))
    table.add_row("LoRA Dropout", str(config.lora.dropout))
    table.add_row("Log Interval", str(config.log_interval_steps))
    table.add_row("Checkpoint Every", str(config.checkpoint_interval_steps))
    table.add_row("Eval Every", str(config.eval_interval_steps))
    table.add_row("Eval Epoch End", str(config.eval_at_epoch_end))
    table.add_row("Auto Resume", str(config.resume_from_checkpoint))
    # Early stopping
    if config.early_stopping_patience > 0:
        table.add_row("Early Stop Patience", str(config.early_stopping_patience))
        table.add_row("Early Stop Threshold", str(config.early_stopping_threshold))
    else:
        table.add_row("Early Stopping", "Disabled")

    console.print(table)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    settings = get_settings()

    # Check API key
    if not settings.tinker_api_key:
        console.print(
            "[red]Error: TINKER_API_KEY not set.[/red]\nSet it in your .env file or environment."
        )
        return 1

    # Load config from YAML if provided
    if args.config and args.config.exists():
        config = load_config_from_yaml(args.config)
    else:
        config = TinkerTrainingConfig()

    # Override with CLI arguments (only if explicitly provided)
    config.model = args.model
    config.dataset_path = args.data or settings.data_dir / "training"
    config.output_dir = args.output or settings.adapters_dir / "tinker"
    config.wait_for_completion = not args.no_wait
    config.dataset_name = args.dataset_name
    config.eval_at_epoch_end = not args.no_eval_at_epoch_end
    config.checkpoint_ttl_seconds = args.checkpoint_ttl_seconds
    config.resume_from_checkpoint = not args.no_resume

    # Only override training params if explicitly provided via CLI
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.log_interval_steps is not None:
        config.log_interval_steps = args.log_interval_steps
    if args.checkpoint_interval_steps is not None:
        config.checkpoint_interval_steps = args.checkpoint_interval_steps
    if args.eval_interval_steps is not None:
        config.eval_interval_steps = args.eval_interval_steps
    if args.early_stopping_patience is not None:
        config.early_stopping_patience = args.early_stopping_patience
    if args.early_stopping_threshold is not None:
        config.early_stopping_threshold = args.early_stopping_threshold

    # Handle LoRA config - only override if any LoRA arg is provided
    if args.lora_rank is not None or args.lora_alpha is not None or args.lora_dropout is not None:
        config.lora = TinkerLoRAConfig(
            rank=args.lora_rank if args.lora_rank is not None else config.lora.rank,
            alpha=args.lora_alpha if args.lora_alpha is not None else config.lora.alpha,
            dropout=args.lora_dropout if args.lora_dropout is not None else config.lora.dropout,
        )

    # Handle status check
    if args.status:
        return check_status(args.status)

    # Handle cost estimation
    if args.estimate_cost:
        if not config.dataset_path.exists():
            console.print(f"[red]Error: Data directory not found: {config.dataset_path}[/red]")
            return 1
        num_examples = count_examples(config.dataset_path)
        if num_examples == 0:
            console.print("[red]Error: No training examples found[/red]")
            return 1
        print_cost_estimate(config, num_examples)
        return 0

    # Run training
    return run_train(config, settings)


def check_status(job_id: str) -> int:
    """Check status of a training job."""
    console.print(f"[bold]Checking status of job: {job_id}[/bold]\n")

    try:
        client = TinkerClient()
        if not client.is_available:
            console.print("[red]Error: Tinker not available[/red]")
            return 1

        status = client.get_job_status(job_id)

        table = Table(title=f"Job Status: {job_id}")
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Status", status.status)
        table.add_row("Progress", f"{status.progress * 100:.1f}%")
        table.add_row("Current Epoch", str(status.current_epoch))
        if status.current_loss:
            table.add_row("Current Loss", f"{status.current_loss:.4f}")
        if status.error:
            table.add_row("Error", f"[red]{status.error}[/red]")

        console.print(table)

        return 0 if status.status != "failed" else 1

    except Exception as e:
        console.print(f"[red]Error checking status: {e}[/red]")
        return 1


def run_train(config: TinkerTrainingConfig, settings) -> int:
    """Run training on Tinker."""
    console.print(
        Panel.fit(
            "[bold]Tinker Cloud Training[/bold]\n\n"
            "Training compression model on Tinker infrastructure",
            border_style="blue",
        )
    )

    # Print configuration
    print_config(config)

    # Check data
    if not config.dataset_path.exists():
        console.print(f"[red]Error: Data directory not found: {config.dataset_path}[/red]")
        console.print("\nRun this first:")
        console.print("  python scripts/format_training_data.py")
        return 1

    num_examples = count_examples(config.dataset_path)
    if num_examples == 0:
        console.print("[red]Error: No training examples found[/red]")
        return 1

    console.print(f"\n[dim]Training examples: {num_examples:,}[/dim]")

    # Show cost estimate
    console.print("\n[bold]Cost Estimate:[/bold]")
    cost = estimate_cost(config, num_examples)
    console.print(f"  Estimated: ${cost['estimated_cost_usd']:.2f}")

    # Confirm
    if config.wait_for_completion:
        console.print(
            "\n[yellow]Training will start and this script will wait for completion.[/yellow]"
        )
    else:
        console.print(
            "\n[yellow]Training will start. Use --status <job_id> to check progress.[/yellow]"
        )

    # Run training
    console.print("\n[bold]Starting training...[/bold]\n")

    result = train_on_tinker(config, api_key=settings.tinker_api_key)

    if not result.success:
        console.print(f"[red]Training failed: {result.error}[/red]")
        return 1

    # Success
    if result.adapter_path:
        console.print(
            Panel.fit(
                f"[green]✓ Training completed![/green]\n\n"
                f"Job ID: {result.job_id}\n"
                f"Adapter saved to: {result.adapter_path}\n"
                f"Final Loss: {result.final_loss:.4f}"
                if result.final_loss
                else "",
                border_style="green",
            )
        )

        console.print("\n[bold]Next steps:[/bold]")
        console.print("  1. Test locally with MLX:")
        console.print(
            f"     mlx_lm.generate --model mlx-community/Qwen3-8B-Instruct-4bit --adapter-path {result.adapter_path}"
        )
    else:
        console.print(
            Panel.fit(
                f"[green]✓ Training started![/green]\n\n"
                f"Job ID: {result.job_id}\n\n"
                f"Check status with:\n"
                f"  python scripts/train_tinker.py --status {result.job_id}",
                border_style="green",
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
