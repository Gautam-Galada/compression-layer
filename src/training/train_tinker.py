"""Tinker cloud training for compression model.

This module provides functionality to train a compression model on Tinker's
cloud infrastructure. Tinker provides fast and cost-effective fine-tuning
for LLMs with full LoRA support.

Note: Requires TINKER_API_KEY environment variable to be set.

Usage:
    from src.training.train_tinker import train_on_tinker, TinkerTrainingConfig

    config = TinkerTrainingConfig(
        model="Qwen/Qwen3-8B",
        dataset_path=Path("data/training"),
    )
    result = train_on_tinker(config)
"""

import json
import logging
import math
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from src.training.tinker_data import render_chat_example

logger = logging.getLogger(__name__)


@dataclass
class TinkerLoRAConfig:
    """LoRA configuration for Tinker training."""

    rank: int = 64
    alpha: int = 128
    dropout: float = 0.0
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class TinkerTrainingConfig:
    """Configuration for Tinker cloud training."""

    # Model
    model: str = "Qwen/Qwen3-8B"

    # Data paths
    dataset_path: Path = field(default_factory=lambda: Path("data/training"))
    output_dir: Path = field(default_factory=lambda: Path("models/adapters/tinker"))

    # Dataset name for Tinker
    dataset_name: str | None = None  # Auto-generated if not provided

    # LoRA configuration
    lora: TinkerLoRAConfig = field(default_factory=TinkerLoRAConfig)

    # Training parameters
    epochs: int = 3
    batch_size: int = 4
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048

    # Job settings
    wait_for_completion: bool = True
    poll_interval: int = 30  # seconds

    # ServiceClient mode controls
    log_interval_steps: int = 10
    checkpoint_interval_steps: int = 250
    eval_interval_steps: int = 0  # 0 = disabled
    eval_at_epoch_end: bool = True
    checkpoint_ttl_seconds: int | None = None
    resume_from_checkpoint: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "model": self.model,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "warmup_ratio": self.warmup_ratio,
            "max_seq_length": self.max_seq_length,
            "log_interval_steps": self.log_interval_steps,
            "checkpoint_interval_steps": self.checkpoint_interval_steps,
            "eval_interval_steps": self.eval_interval_steps,
            "eval_at_epoch_end": self.eval_at_epoch_end,
            "checkpoint_ttl_seconds": self.checkpoint_ttl_seconds,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "lora": {
                "r": self.lora.rank,
                "alpha": self.lora.alpha,
                "dropout": self.lora.dropout,
                "target_modules": self.lora.target_modules,
            },
        }


@dataclass
class TinkerJobStatus:
    """Status of a Tinker training job."""

    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: float = 0.0  # 0.0 to 1.0
    current_epoch: int = 0
    current_loss: float | None = None
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class TinkerTrainingResult:
    """Result from a Tinker training run."""

    success: bool
    job_id: str | None = None
    adapter_path: Path | None = None
    final_loss: float | None = None
    total_epochs: int = 0
    error: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


class TinkerClient:
    """
    Client for Tinker cloud training API.

    This is a mock/placeholder implementation. When Tinker SDK is available,
    replace with actual SDK calls.
    """

    def __init__(self, api_key: str | None = None):
        """
        Initialize Tinker client.

        Args:
            api_key: Tinker API key (defaults to TINKER_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("TINKER_API_KEY", "")
        if not self.api_key:
            logger.warning("TINKER_API_KEY not set. Tinker operations will fail.")

        self._sdk_available = self._check_sdk()

    def _check_sdk(self) -> bool:
        """Check if Tinker SDK is available."""
        try:
            import importlib.util

            return importlib.util.find_spec("tinker") is not None
        except ImportError:
            return False

    def _load_tinker_module(self) -> ModuleType:
        """Import and return the installed tinker module."""
        return __import__("tinker")

    def has_legacy_client_api(self) -> bool:
        """Whether SDK exposes legacy Client(api_key=...) API."""
        tinker = self._load_tinker_module()
        return callable(getattr(tinker, "Client", None))

    def has_service_client_api(self) -> bool:
        """Whether SDK exposes ServiceClient API."""
        tinker = self._load_tinker_module()
        return callable(getattr(tinker, "ServiceClient", None))

    @property
    def is_available(self) -> bool:
        """Check if Tinker is properly configured."""
        return bool(self.api_key) and self._sdk_available

    def upload_dataset(self, data_dir: Path, name: str) -> str:
        """
        Upload dataset to Tinker.

        Args:
            data_dir: Path to directory with train.jsonl, valid.jsonl, test.jsonl
            name: Dataset name

        Returns:
            Dataset ID
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available. Install with: pip install tinker")

        tinker = self._load_tinker_module()  # Dynamic import to avoid static analysis errors

        if not self.has_legacy_client_api():
            raise RuntimeError(
                "Installed tinker SDK does not expose Client API required for job-style upload. "
                "Use scripts/train_tinker.py in ServiceClient mode (default in this repo) or "
                "install the legacy SDK variant."
            )

        client = tinker.Client(api_key=self.api_key)

        # Tinker expects the data directory path
        dataset_id: str = client.data.upload(str(data_dir), name=name)

        logger.info(f"Uploaded dataset '{name}' with ID: {dataset_id}")
        return dataset_id

    def start_training(
        self,
        config: TinkerTrainingConfig,
        dataset_id: str,
    ) -> str:
        """
        Start a training job on Tinker.

        Args:
            config: Training configuration
            dataset_id: ID of uploaded dataset

        Returns:
            Job ID
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available. Install with: pip install tinker")

        tinker = self._load_tinker_module()

        if not self.has_legacy_client_api():
            raise RuntimeError(
                "Installed tinker SDK does not expose Client API required for job-style training."
            )

        client = tinker.Client(api_key=self.api_key)

        # Start training job
        job = client.train(
            model=config.model,
            dataset=dataset_id,
            lora_rank=config.lora.rank,
            lora_alpha=config.lora.alpha,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
        )

        logger.info(f"Started training job: {job.id}")
        return str(job.id)

    def get_job_status(self, job_id: str) -> TinkerJobStatus:
        """
        Get status of a training job.

        Args:
            job_id: Training job ID

        Returns:
            Job status
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available")

        tinker = self._load_tinker_module()

        if not self.has_legacy_client_api():
            raise RuntimeError(
                "Installed tinker SDK does not expose Client API required for job status polling."
            )

        client = tinker.Client(api_key=self.api_key)
        job = client.get_job(job_id)

        return TinkerJobStatus(
            job_id=job_id,
            status=job.status,
            progress=getattr(job, "progress", 0.0),
            current_epoch=getattr(job, "current_epoch", 0),
            current_loss=getattr(job, "current_loss", None),
            error=getattr(job, "error", None),
            metrics=getattr(job, "metrics", {}),
        )

    def wait_for_completion(
        self,
        job_id: str,
        poll_interval: int = 30,
        callback: Any | None = None,
    ) -> TinkerJobStatus:
        """
        Wait for a training job to complete.

        Args:
            job_id: Training job ID
            poll_interval: Seconds between status checks
            callback: Optional callback(status) called on each poll

        Returns:
            Final job status
        """
        while True:
            status = self.get_job_status(job_id)

            if callback:
                callback(status)

            if status.status in ("completed", "failed"):
                return status

            logger.info(
                f"Job {job_id}: {status.status} "
                f"({status.progress * 100:.1f}%, epoch {status.current_epoch})"
            )

            time.sleep(poll_interval)

    def download_adapter(self, job_id: str, output_path: Path) -> Path:
        """
        Download trained adapter from completed job.

        Args:
            job_id: Training job ID
            output_path: Where to save adapter

        Returns:
            Path to downloaded adapter
        """
        if not self._sdk_available:
            raise RuntimeError("Tinker SDK not available")

        tinker = self._load_tinker_module()

        if not self.has_legacy_client_api():
            raise RuntimeError(
                "Installed tinker SDK does not expose Client API required for adapter download."
            )

        client = tinker.Client(api_key=self.api_key)
        job = client.get_job(job_id)

        output_path.mkdir(parents=True, exist_ok=True)
        job.download_adapter(str(output_path))

        logger.info(f"Downloaded adapter to {output_path}")
        return output_path


def train_on_tinker(
    config: TinkerTrainingConfig,
    api_key: str | None = None,
) -> TinkerTrainingResult:
    """
    Run training on Tinker cloud.

    This function:
    1. Validates configuration and data
    2. Uploads dataset to Tinker
    3. Starts training job
    4. Optionally waits for completion
    5. Downloads adapter

    Args:
        config: Training configuration

    Returns:
        TinkerTrainingResult with success status and adapter path
    """
    client = TinkerClient(api_key=api_key)

    # Check availability
    if not client.is_available:
        return TinkerTrainingResult(
            success=False,
            error="Tinker not available. Ensure TINKER_API_KEY is set and tinker SDK is installed.",
        )

    # Validate data
    if not config.dataset_path.exists():
        return TinkerTrainingResult(
            success=False,
            error=f"Dataset path not found: {config.dataset_path}",
        )

    train_file = config.dataset_path / "train.jsonl"
    if not train_file.exists():
        return TinkerTrainingResult(
            success=False,
            error=f"Training file not found: {train_file}",
        )

    try:
        if client.has_service_client_api() and not client.has_legacy_client_api():
            if not config.wait_for_completion:
                return TinkerTrainingResult(
                    success=False,
                    error="--no-wait is not supported with the installed ServiceClient SDK mode.",
                )
            return _train_with_service_client_sdk(config, api_key=client.api_key)

        # Generate dataset name if not provided
        dataset_name = config.dataset_name or f"compression-{int(time.time())}"

        # Upload dataset
        logger.info(f"Uploading dataset from {config.dataset_path}...")
        dataset_id = client.upload_dataset(config.dataset_path, dataset_name)

        # Start training
        logger.info(f"Starting training job with model {config.model}...")
        job_id = client.start_training(config, dataset_id)

        if not config.wait_for_completion:
            return TinkerTrainingResult(
                success=True,
                job_id=job_id,
                error=None,
            )

        # Wait for completion
        logger.info("Waiting for training to complete...")
        final_status = client.wait_for_completion(
            job_id,
            poll_interval=config.poll_interval,
        )

        if final_status.status != "completed":
            return TinkerTrainingResult(
                success=False,
                job_id=job_id,
                error=final_status.error or f"Training failed with status: {final_status.status}",
            )

        # Download adapter
        logger.info("Downloading trained adapter...")
        adapter_path = client.download_adapter(job_id, config.output_dir)

        return TinkerTrainingResult(
            success=True,
            job_id=job_id,
            adapter_path=adapter_path,
            final_loss=final_status.current_loss,
            total_epochs=config.epochs,
            metrics=final_status.metrics,
        )

    except Exception as e:
        logger.exception("Tinker training failed")
        return TinkerTrainingResult(
            success=False,
            error=str(e),
        )


def _utc_now_iso() -> str:
    """Return UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017


def _append_jsonl(file_path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON object to a JSONL file."""
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _append_train_log_line(train_log_path: Path, line: str) -> None:
    """Append one line to train.log."""
    with open(train_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _write_service_run_state(state_path: Path, state: dict[str, Any]) -> None:
    """Persist service-mode run state and a mlflow-compatible run.json."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)

    run_json_path = state_path.parent / "run.json"
    run_json = {
        "started_at": state.get("started_at", _utc_now_iso()),
        "model": state.get("model"),
        "git_sha": state.get("git_sha", os.environ.get("GITHUB_SHA", "unknown")),
        "data_dir": state.get("data_dir"),
        "lora_rank": state.get("lora_rank"),
        "lora_alpha": state.get("lora_alpha"),
        "batch_size": state.get("batch_size"),
        "learning_rate": state.get("learning_rate"),
        "iters": state.get("total_steps"),
    }
    with open(run_json_path, "w", encoding="utf-8") as f:
        json.dump(run_json, f, indent=2, ensure_ascii=False)


def _load_service_run_state(state_path: Path) -> dict[str, Any]:
    """Load prior run state if present."""
    if not state_path.exists():
        return {}

    try:
        with open(state_path, encoding="utf-8") as f:
            loaded = json.load(f)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        logger.warning("Unable to parse run state at %s; starting fresh", state_path)
        return {}


def _to_sdk_datum(local_datum: Any, tinker_module: ModuleType) -> tuple[Any, int]:
    """Convert local tinker_data datum to SDK Datum and return token count."""
    target_tokens = [int(token) for token in local_datum.loss_fn_inputs["target_tokens"]]
    weights = [float(weight) for weight in local_datum.loss_fn_inputs["weights"]]

    datum = tinker_module.Datum(
        model_input=tinker_module.ModelInput.from_ints(local_datum.model_input.tokens),
        loss_fn_inputs={
            "target_tokens": tinker_module.TensorData(data=target_tokens, dtype="int64"),
            "weights": tinker_module.TensorData(data=weights, dtype="float32"),
        },
    )
    return datum, len(target_tokens)


def _iter_training_batches(
    train_file: Path,
    tokenizer: Any,
    tinker_module: ModuleType,
    batch_size: int,
) -> Iterator[tuple[list[Any], int]]:
    """Yield SDK-ready training batches with token counts from chat JSONL data."""
    batch: list[Any] = []
    batch_tokens = 0

    with open(train_file, encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping invalid JSON at line %s", line_number)
                continue

            messages = record.get("messages") if isinstance(record, dict) else None
            if not isinstance(messages, list) or not messages:
                logger.warning("Skipping malformed record at line %s", line_number)
                continue

            local_datum = render_chat_example(messages, tokenizer)
            sdk_datum, token_count = _to_sdk_datum(local_datum, tinker_module)
            batch.append(sdk_datum)
            batch_tokens += token_count

            if len(batch) >= batch_size:
                yield batch, batch_tokens
                batch = []
                batch_tokens = 0

    if batch:
        yield batch, batch_tokens


def _extract_loss(metrics: dict[str, float]) -> float | None:
    """Extract best-effort scalar loss value from metrics dict."""
    for key, value in metrics.items():
        if "loss" in key.lower():
            return float(value)

    if metrics:
        first_value = next(iter(metrics.values()))
        return float(first_value)

    return None


def _run_validation(
    training_client: Any,
    valid_file: Path,
    tokenizer: Any,
    tinker_module: ModuleType,
    batch_size: int,
) -> tuple[float | None, int]:
    """Run validation pass and return mean validation loss and batch count."""
    if not valid_file.exists() or not hasattr(training_client, "forward"):
        return None, 0

    losses: list[float] = []
    val_batches = 0
    for batch, _ in _iter_training_batches(valid_file, tokenizer, tinker_module, batch_size):
        forward_result = training_client.forward(batch, "cross_entropy").result()
        metrics = getattr(forward_result, "metrics", {}) or {}
        val_loss = _extract_loss(metrics)
        if val_loss is not None:
            losses.append(val_loss)
        val_batches += 1

    if not losses:
        return None, val_batches
    return sum(losses) / len(losses), val_batches


def _train_with_service_client_sdk(
    config: TinkerTrainingConfig,
    api_key: str,
) -> TinkerTrainingResult:
    """Train via the modern Tinker ServiceClient/TrainingClient SDK flow."""
    if config.log_interval_steps <= 0:
        return TinkerTrainingResult(
            success=False,
            error="log_interval_steps must be greater than 0",
        )

    tinker = __import__("tinker")

    train_file = config.dataset_path / "train.jsonl"
    valid_file = config.dataset_path / "valid.jsonl"

    with open(train_file, encoding="utf-8") as train_stream:
        total_examples = sum(1 for line in train_stream if line.strip())
    if total_examples == 0:
        return TinkerTrainingResult(
            success=False, error=f"No training examples found in {train_file}"
        )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    train_log_path = config.output_dir / "train.log"
    metrics_path = config.output_dir / "metrics.jsonl"
    run_state_path = config.output_dir / "tinker_run.json"
    existing_state = _load_service_run_state(run_state_path)
    if existing_state.get("sdk_mode") not in {None, "service_client"}:
        existing_state = {}

    service_client = tinker.ServiceClient(api_key=api_key)

    latest_checkpoint_path = ""
    completed_steps = 0
    if config.resume_from_checkpoint:
        latest_checkpoint_path = str(existing_state.get("latest_checkpoint_path", ""))
        completed_steps = int(existing_state.get("completed_steps", 0) or 0)

    resumed = False
    if latest_checkpoint_path and hasattr(
        service_client, "create_training_client_from_state_with_optimizer"
    ):
        training_client = service_client.create_training_client_from_state_with_optimizer(
            latest_checkpoint_path
        )
        resumed = True
        logger.info("Resuming ServiceClient training from checkpoint: %s", latest_checkpoint_path)
    else:
        if latest_checkpoint_path:
            logger.warning(
                "Resume checkpoint found but SDK cannot restore optimizer state; "
                "starting fresh training for %s",
                config.model,
            )
            latest_checkpoint_path = ""
            completed_steps = 0
        training_client = service_client.create_lora_training_client(
            base_model=config.model,
            rank=config.lora.rank,
        )

    tokenizer = training_client.get_tokenizer()
    info = training_client.get_info()
    training_run_id = str(getattr(info, "model_id", "unknown"))

    steps_per_epoch = math.ceil(total_examples / config.batch_size)
    total_steps = steps_per_epoch * config.epochs
    completed_final_loss_raw = existing_state.get("final_loss")
    completed_final_loss: float | None = None
    if isinstance(completed_final_loss_raw, (int, float)):
        completed_final_loss = float(completed_final_loss_raw)
    if completed_steps >= total_steps:
        return TinkerTrainingResult(
            success=True,
            job_id=training_run_id,
            adapter_path=config.output_dir,
            final_loss=completed_final_loss,
            total_epochs=config.epochs,
            metrics={"message": "Training already complete for configured epochs"},
        )

    checkpoints = existing_state.get("checkpoints", [])
    if not isinstance(checkpoints, list):
        checkpoints = []

    state: dict[str, Any] = {
        "sdk_mode": "service_client",
        "training_run_id": training_run_id,
        "model": config.model,
        "data_dir": str(config.dataset_path),
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "lora_rank": config.lora.rank,
        "lora_alpha": config.lora.alpha,
        "total_examples": total_examples,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "completed_steps": completed_steps,
        "latest_checkpoint_path": latest_checkpoint_path,
        "checkpoints": checkpoints,
        "status": "running",
        "started_at": str(existing_state.get("started_at", _utc_now_iso())),
        "updated_at": _utc_now_iso(),
        "resumed": resumed,
    }
    _write_service_run_state(run_state_path, state)

    current_step = completed_steps
    final_loss: float | None = None
    existing_final_loss_raw = existing_state.get("final_loss")
    if isinstance(existing_final_loss_raw, (int, float)):
        final_loss = float(existing_final_loss_raw)

    logger.info(
        "Starting ServiceClient training: %s examples, %s epochs, %s total steps",
        total_examples,
        config.epochs,
        total_steps,
    )

    start_epoch = (completed_steps // steps_per_epoch) + 1
    skip_batches = completed_steps % steps_per_epoch
    for epoch in range(start_epoch, config.epochs + 1):
        epoch_batch_index = 0
        for batch, batch_tokens in _iter_training_batches(
            train_file,
            tokenizer,
            tinker,
            config.batch_size,
        ):
            epoch_batch_index += 1
            if epoch == start_epoch and epoch_batch_index <= skip_batches:
                continue

            step_start = time.perf_counter()
            fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
            optim_future = training_client.optim_step(
                tinker.AdamParams(learning_rate=config.learning_rate),
            )
            fwdbwd = fwdbwd_future.result()
            optim_future.result()
            step_elapsed = max(time.perf_counter() - step_start, 1e-6)
            tokens_per_sec = batch_tokens / step_elapsed if batch_tokens > 0 else 0.0

            current_step += 1
            metrics = getattr(fwdbwd, "metrics", {}) or {}
            step_loss = _extract_loss(metrics)
            if step_loss is not None:
                final_loss = step_loss

            if current_step % config.log_interval_steps == 0 or current_step == total_steps:
                logger.info(
                    "Epoch %s/%s Step %s/%s Loss %s",
                    epoch,
                    config.epochs,
                    current_step,
                    total_steps,
                    f"{step_loss:.4f}" if step_loss is not None else "n/a",
                )

                if step_loss is not None:
                    _append_train_log_line(
                        train_log_path,
                        (
                            f"Iter {current_step}: Train loss {step_loss:.4f} "
                            f"| Tokens/sec {tokens_per_sec:.1f} | Peak mem 0.0 GB"
                        ),
                    )
                    _append_jsonl(
                        metrics_path,
                        {
                            "timestamp": _utc_now_iso(),
                            "type": "train",
                            "step": current_step,
                            "epoch": epoch,
                            "train_loss": step_loss,
                            "tokens_per_sec": tokens_per_sec,
                        },
                    )

                state["completed_steps"] = current_step
                state["last_train_loss"] = step_loss
                state["updated_at"] = _utc_now_iso()
                _write_service_run_state(run_state_path, state)

            if config.eval_interval_steps > 0 and current_step % config.eval_interval_steps == 0:
                val_loss, val_batches = _run_validation(
                    training_client,
                    valid_file,
                    tokenizer,
                    tinker,
                    config.batch_size,
                )
                if val_loss is not None:
                    _append_train_log_line(
                        train_log_path, f"Iter {current_step}: Val loss {val_loss:.4f}"
                    )
                    _append_jsonl(
                        metrics_path,
                        {
                            "timestamp": _utc_now_iso(),
                            "type": "val",
                            "step": current_step,
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_batches": val_batches,
                        },
                    )
                    state["last_val_loss"] = val_loss
                    state["updated_at"] = _utc_now_iso()
                    _write_service_run_state(run_state_path, state)

            if (
                config.checkpoint_interval_steps > 0
                and current_step % config.checkpoint_interval_steps == 0
            ):
                checkpoint_name = f"step-{current_step:06d}"
                checkpoint_response = training_client.save_state(
                    checkpoint_name,
                    ttl_seconds=config.checkpoint_ttl_seconds,
                ).result()
                checkpoint_path = str(getattr(checkpoint_response, "path", ""))
                if checkpoint_path:
                    checkpoints.append(
                        {
                            "name": checkpoint_name,
                            "step": current_step,
                            "epoch": epoch,
                            "path": checkpoint_path,
                            "created_at": _utc_now_iso(),
                        }
                    )
                    state["completed_steps"] = current_step
                    state["latest_checkpoint_path"] = checkpoint_path
                    state["checkpoints"] = checkpoints
                    state["updated_at"] = _utc_now_iso()
                    _write_service_run_state(run_state_path, state)

        if config.eval_at_epoch_end:
            val_loss, val_batches = _run_validation(
                training_client,
                valid_file,
                tokenizer,
                tinker,
                config.batch_size,
            )
            if val_loss is not None:
                _append_train_log_line(
                    train_log_path, f"Iter {current_step}: Val loss {val_loss:.4f}"
                )
                _append_jsonl(
                    metrics_path,
                    {
                        "timestamp": _utc_now_iso(),
                        "type": "val",
                        "step": current_step,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "val_batches": val_batches,
                    },
                )
                state["last_val_loss"] = val_loss
                state["updated_at"] = _utc_now_iso()
                _write_service_run_state(run_state_path, state)

    if current_step == completed_steps:
        return TinkerTrainingResult(
            success=False,
            error="No valid training examples were found after parsing train.jsonl",
        )

    save_name = "final"
    save_response = training_client.save_state(
        save_name,
        ttl_seconds=config.checkpoint_ttl_seconds,
    ).result()
    checkpoint_path = str(getattr(save_response, "path", ""))
    if checkpoint_path:
        checkpoints.append(
            {
                "name": save_name,
                "step": current_step,
                "epoch": config.epochs,
                "path": checkpoint_path,
                "created_at": _utc_now_iso(),
            }
        )

    state["status"] = "completed"
    state["completed_steps"] = current_step
    state["latest_checkpoint_path"] = checkpoint_path or state.get("latest_checkpoint_path", "")
    state["checkpoints"] = checkpoints
    state["final_loss"] = final_loss
    state["updated_at"] = _utc_now_iso()
    _write_service_run_state(run_state_path, state)

    return TinkerTrainingResult(
        success=True,
        job_id=training_run_id,
        adapter_path=config.output_dir,
        final_loss=final_loss,
        total_epochs=config.epochs,
        metrics={
            "checkpoint_path": checkpoint_path,
            "metadata_path": str(run_state_path),
            "metrics_path": str(metrics_path),
            "train_log_path": str(train_log_path),
        },
    )


def load_config_from_yaml(config_path: Path) -> TinkerTrainingConfig:
    """
    Load Tinker training config from YAML file.

    Args:
        config_path: Path to configs/training.yaml

    Returns:
        TinkerTrainingConfig populated from YAML
    """
    with open(config_path) as f:
        yaml_config = yaml.safe_load(f)

    cloud_config = yaml_config.get("cloud", {})
    lora_config = cloud_config.get("lora", {})
    training_config = cloud_config.get("training", {})

    return TinkerTrainingConfig(
        model=cloud_config.get("model", "Qwen/Qwen3-8B"),
        lora=TinkerLoRAConfig(
            rank=lora_config.get("rank", 64),
            alpha=lora_config.get("alpha", 128),
            target_modules=lora_config.get(
                "target_modules",
                ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ),
        ),
        epochs=training_config.get("epochs", 3),
        batch_size=training_config.get("batch_size", 4),
        learning_rate=training_config.get("learning_rate", 2e-4),
        log_interval_steps=training_config.get("log_interval_steps", 10),
        checkpoint_interval_steps=training_config.get("checkpoint_interval_steps", 250),
        eval_interval_steps=training_config.get("eval_interval_steps", 0),
        eval_at_epoch_end=training_config.get("eval_at_epoch_end", True),
        checkpoint_ttl_seconds=training_config.get("checkpoint_ttl_seconds", None),
        resume_from_checkpoint=training_config.get("resume_from_checkpoint", True),
    )


def estimate_cost(config: TinkerTrainingConfig, num_examples: int) -> dict[str, Any]:
    """
    Estimate training cost on Tinker.

    Based on documentation:
    - Qwen3-8B: $0.40/1M tokens

    Args:
        config: Training configuration
        num_examples: Number of training examples

    Returns:
        Dictionary with cost estimates
    """
    # Rough estimates based on typical compression pair lengths
    AVG_TOKENS_PER_EXAMPLE = 500  # system + user + assistant

    total_tokens = num_examples * AVG_TOKENS_PER_EXAMPLE * config.epochs

    # Cost per model (rough estimates)
    cost_per_million = {
        "Qwen/Qwen3-8B": 0.40,
        "Qwen/Qwen3-4B": 0.20,
        "Qwen/Qwen3-30B-A3B": 0.45,  # MoE, efficient
    }

    rate = cost_per_million.get(config.model, 0.40)
    estimated_cost = (total_tokens / 1_000_000) * rate

    return {
        "total_tokens": total_tokens,
        "cost_per_million": rate,
        "estimated_cost_usd": estimated_cost,
        "model": config.model,
        "epochs": config.epochs,
        "examples": num_examples,
    }
