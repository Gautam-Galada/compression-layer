"""Tests for Tinker training visualizer parser."""

import json
import tempfile
from pathlib import Path

from scripts.visualize_tinker_training import TinkerMetricsParser, TinkerTrainingMetrics


class TestTinkerMetricsParser:
    def _write_metrics(self, records: list[dict]) -> Path:
        """Write records to a temp JSONL file and return the path."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        for record in records:
            tmp.write(json.dumps(record) + "\n")
        tmp.close()
        return Path(tmp.name)

    def test_parse_train_entries(self):
        path = self._write_metrics(
            [
                {
                    "type": "train",
                    "step": 10,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "train",
                    "step": 20,
                    "epoch": 1,
                    "train_loss": 4.0,
                    "tokens_per_sec": 200.0,
                },
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.train_steps == [10, 20]
        assert metrics.train_losses == [5.0, 4.0]
        assert metrics.tokens_per_sec == [100.0, 200.0]
        assert metrics.train_epochs == [1, 1]

    def test_parse_val_entries(self):
        path = self._write_metrics(
            [
                {"type": "val", "step": 100, "epoch": 1, "val_loss": 3.5, "val_batches": 10},
                {"type": "val", "step": 200, "epoch": 1, "val_loss": 3.0, "val_batches": 10},
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.val_steps == [100, 200]
        assert metrics.val_losses == [3.5, 3.0]
        assert metrics.best_val_loss == 3.0

    def test_epoch_boundary_detection(self):
        path = self._write_metrics(
            [
                {
                    "type": "train",
                    "step": 490,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "train",
                    "step": 500,
                    "epoch": 2,
                    "train_loss": 4.0,
                    "tokens_per_sec": 100.0,
                },
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.epoch_boundaries == [500]

    def test_early_stop_parsing(self):
        path = self._write_metrics(
            [
                {
                    "type": "train",
                    "step": 10,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "early_stop",
                    "step": 1500,
                    "best_val_loss": 2.5,
                    "evals_without_improvement": 5,
                },
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.early_stop_step == 1500
        assert metrics.best_val_loss == 2.5

    def test_empty_file(self):
        path = self._write_metrics([])
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.train_steps == []
        assert metrics.val_steps == []
        assert metrics.best_val_loss is None

    def test_mixed_entries(self):
        path = self._write_metrics(
            [
                {
                    "type": "train",
                    "step": 10,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "tokens_per_sec": 100.0,
                },
                {"type": "val", "step": 100, "epoch": 1, "val_loss": 4.0, "val_batches": 10},
                {
                    "type": "train",
                    "step": 110,
                    "epoch": 1,
                    "train_loss": 3.0,
                    "tokens_per_sec": 150.0,
                },
                {"type": "val", "step": 200, "epoch": 1, "val_loss": 3.5, "val_batches": 10},
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert len(metrics.train_steps) == 2
        assert len(metrics.val_steps) == 2
        # min(4.0, 3.5) = 3.5
        assert metrics.best_val_loss == 3.5

    def test_train_loss_total_field(self):
        """New format entries with train_loss_total are captured."""
        path = self._write_metrics(
            [
                {
                    "type": "train",
                    "step": 10,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "train_loss_total": 50.0,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "train",
                    "step": 20,
                    "epoch": 1,
                    "train_loss": 4.0,
                    "tokens_per_sec": 200.0,
                },
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        # Only one entry had train_loss_total
        assert metrics.train_losses_total == [50.0]
        # Both entries still have train_loss
        assert metrics.train_losses == [5.0, 4.0]

    def test_missing_tokens_per_sec_defaults_to_zero(self):
        """Entries without tokens_per_sec default to 0.0."""
        path = self._write_metrics(
            [
                {"type": "train", "step": 10, "epoch": 1, "train_loss": 5.0},
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.tokens_per_sec == [0.0]

    def test_blank_lines_skipped(self):
        """Blank lines in the JSONL file are skipped gracefully."""
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False)
        tmp.write(
            json.dumps(
                {
                    "type": "train",
                    "step": 10,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "tokens_per_sec": 100.0,
                }
            )
            + "\n"
        )
        tmp.write("\n")
        tmp.write(
            json.dumps(
                {
                    "type": "train",
                    "step": 20,
                    "epoch": 1,
                    "train_loss": 4.0,
                    "tokens_per_sec": 200.0,
                }
            )
            + "\n"
        )
        tmp.close()
        metrics = TinkerMetricsParser().parse(Path(tmp.name))
        assert metrics.train_steps == [10, 20]

    def test_multiple_epoch_boundaries(self):
        """Multiple epoch transitions are all captured."""
        path = self._write_metrics(
            [
                {
                    "type": "train",
                    "step": 100,
                    "epoch": 1,
                    "train_loss": 5.0,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "train",
                    "step": 200,
                    "epoch": 2,
                    "train_loss": 4.0,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "train",
                    "step": 300,
                    "epoch": 2,
                    "train_loss": 3.5,
                    "tokens_per_sec": 100.0,
                },
                {
                    "type": "train",
                    "step": 400,
                    "epoch": 3,
                    "train_loss": 3.0,
                    "tokens_per_sec": 100.0,
                },
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.epoch_boundaries == [200, 400]

    def test_early_stop_overrides_best_val_loss(self):
        """Early stop's best_val_loss takes precedence over computed min."""
        path = self._write_metrics(
            [
                {"type": "val", "step": 100, "epoch": 1, "val_loss": 4.0, "val_batches": 10},
                {"type": "val", "step": 200, "epoch": 1, "val_loss": 3.0, "val_batches": 10},
                {
                    "type": "early_stop",
                    "step": 300,
                    "best_val_loss": 2.8,
                    "evals_without_improvement": 3,
                },
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        # early_stop sets best_val_loss directly; parser skips argmin computation
        assert metrics.best_val_loss == 2.8

    def test_default_epoch_when_missing(self):
        """Entries without epoch field default to epoch 1."""
        path = self._write_metrics(
            [
                {"type": "train", "step": 10, "train_loss": 5.0, "tokens_per_sec": 100.0},
                {"type": "val", "step": 100, "val_loss": 3.0, "val_batches": 10},
            ]
        )
        metrics = TinkerMetricsParser().parse(path)
        assert metrics.train_epochs == [1]
        assert metrics.val_epochs == [1]
