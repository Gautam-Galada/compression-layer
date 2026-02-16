import json
import re
import sys
import types
from pathlib import Path

import pytest

import src.training.train_tinker as train_tinker
from src.training.train_tinker import _check_early_stopping


class FakeClient:
    last_api_key = None

    def __init__(self, api_key: str | None = None):
        FakeClient.last_api_key = api_key
        self.api_key = api_key
        self._sdk_available = True

    @property
    def is_available(self) -> bool:
        return True

    def has_legacy_client_api(self) -> bool:
        return True

    def has_service_client_api(self) -> bool:
        return False

    def upload_dataset(self, data_dir: Path, name: str) -> str:
        return "dataset-123"

    def start_training(self, config: train_tinker.TinkerTrainingConfig, dataset_id: str) -> str:
        return "job-123"

    def wait_for_completion(self, job_id: str, poll_interval: int = 30, callback=None):
        return train_tinker.TinkerJobStatus(
            job_id=job_id,
            status="completed",
            current_loss=0.42,
        )

    def download_adapter(self, job_id: str, output_path: Path) -> Path:
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path


def test_train_on_tinker_uses_api_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text("{}\n", encoding="utf-8")

    config = train_tinker.TinkerTrainingConfig(dataset_path=data_dir)

    monkeypatch.setattr(train_tinker, "TinkerClient", FakeClient)

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True
    assert FakeClient.last_api_key == "test-key"


def test_train_on_tinker_supports_service_client_sdk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            return _Future(types.SimpleNamespace(path="tinker://run-123/weights/final"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        init_kwargs = None

        def __init__(self, **kwargs):  # noqa: ARG002
            _ServiceClient.init_kwargs = kwargs

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=tmp_path / "out",
        wait_for_completion=True,
        epochs=1,
        batch_size=1,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True
    assert result.job_id == "run-123"
    assert _ServiceClient.init_kwargs is not None
    assert _ServiceClient.init_kwargs.get("api_key") == "test-key"


def test_train_on_tinker_service_client_errors_are_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )

    class _ServiceModeClient:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key or ""

        @property
        def is_available(self) -> bool:
            return True

        def has_legacy_client_api(self) -> bool:
            return False

        def has_service_client_api(self) -> bool:
            return True

    config = train_tinker.TinkerTrainingConfig(dataset_path=data_dir)
    monkeypatch.setattr(train_tinker, "TinkerClient", _ServiceModeClient)

    def _raise_service_error(config, api_key):  # noqa: ARG001
        raise RuntimeError("service path exploded")

    monkeypatch.setattr(train_tinker, "_train_with_service_client_sdk", _raise_service_error)

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is False
    assert result.error is not None
    assert "service path exploded" in result.error


def test_train_on_tinker_service_client_rejects_no_valid_examples(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        'not-json\n{"foo":"bar"}\n',
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        save_called = False

        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            _TrainingClient.save_called = True
            return _Future(types.SimpleNamespace(path="tinker://run-123/weights/final"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=tmp_path / "out",
        wait_for_completion=True,
        epochs=1,
        batch_size=1,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is False
    assert result.error is not None
    assert "No valid training examples" in result.error
    assert _TrainingClient.save_called is False


def test_train_on_tinker_service_client_writes_metrics_validation_and_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        "\n".join(
            [
                '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u1"},{"role":"assistant","content":"a1"}]}',
                '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u2"},{"role":"assistant","content":"a2"}]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (data_dir / "valid.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"vu"},{"role":"assistant","content":"va"}]}\n',
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        checkpoint_names: list[str] = []
        forward_calls = 0

        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def forward(self, data, loss_fn):  # noqa: ARG002
            _TrainingClient.forward_calls += 1
            return _Future(types.SimpleNamespace(metrics={"loss": 0.25}, loss_fn_outputs=[]))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            _TrainingClient.checkpoint_names.append(name)
            return _Future(types.SimpleNamespace(path=f"tinker://run-123/weights/{name}"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

        def create_training_client_from_state_with_optimizer(self, path: str):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    output_dir = tmp_path / "out"
    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        wait_for_completion=True,
        epochs=1,
        batch_size=1,
        log_interval_steps=1,
        checkpoint_interval_steps=1,
        eval_at_epoch_end=True,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True
    assert (output_dir / "tinker_run.json").exists()
    assert (output_dir / "run.json").exists()
    assert (output_dir / "metrics.jsonl").exists()
    assert (output_dir / "train.log").exists()
    assert _TrainingClient.forward_calls > 0
    assert "final" in _TrainingClient.checkpoint_names

    run_json = json.loads((output_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["model"] == config.model
    assert run_json["lora_rank"] == config.lora.rank
    assert run_json["lora_alpha"] == config.lora.alpha
    assert run_json["iters"] == 2

    train_log_text = (output_dir / "train.log").read_text(encoding="utf-8")
    assert re.search(
        r"Iter (\d+): Train loss ([0-9.]+).*Tokens/sec ([0-9.]+).*Peak mem ([0-9.]+) GB",
        train_log_text,
    )
    assert re.search(r"Iter (\d+): Val loss ([0-9.]+)", train_log_text)

    metric_types = [
        json.loads(line)["type"]
        for line in (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert "train" in metric_types
    assert "val" in metric_types


def test_train_on_tinker_service_client_resumes_from_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "tinker_run.json").write_text(
        json.dumps(
            {
                "sdk_mode": "service_client",
                "latest_checkpoint_path": "tinker://run-123/weights/step-000050",
                "completed_steps": 0,
            }
        ),
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            return _Future(types.SimpleNamespace(path=f"tinker://run-123/weights/{name}"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        resume_path = None

        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

        def create_training_client_from_state_with_optimizer(self, path: str):
            _ServiceClient.resume_path = path
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        wait_for_completion=True,
        epochs=1,
        batch_size=1,
        log_interval_steps=1,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True
    assert _ServiceClient.resume_path == "tinker://run-123/weights/step-000050"


def test_train_on_tinker_service_client_checkpoint_persists_completed_steps(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            if name == "final":
                raise RuntimeError("final save failed")
            return _Future(types.SimpleNamespace(path=f"tinker://run-123/weights/{name}"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    output_dir = tmp_path / "out"
    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        wait_for_completion=True,
        epochs=1,
        batch_size=1,
        log_interval_steps=10,
        checkpoint_interval_steps=1,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is False
    saved_state = json.loads((output_dir / "tinker_run.json").read_text(encoding="utf-8"))
    assert saved_state["completed_steps"] == 1
    assert saved_state["latest_checkpoint_path"].endswith("step-000001")


def test_train_on_tinker_service_client_resets_progress_without_resume_support(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )

    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "tinker_run.json").write_text(
        json.dumps(
            {
                "sdk_mode": "service_client",
                "latest_checkpoint_path": "tinker://run-123/weights/step-000001",
                "completed_steps": 1,
            }
        ),
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        train_steps = 0

        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            _TrainingClient.train_steps += 1
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            return _Future(types.SimpleNamespace(path=f"tinker://run-123/weights/{name}"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        wait_for_completion=True,
        epochs=2,
        batch_size=1,
        log_interval_steps=1,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True
    assert _TrainingClient.train_steps == 2


def test_train_on_tinker_service_client_rejects_zero_log_interval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"a"}]}\n',
        encoding="utf-8",
    )

    class _ServiceModeClient:
        def __init__(self, api_key: str | None = None):
            self.api_key = api_key or ""

        @property
        def is_available(self) -> bool:
            return True

        def has_legacy_client_api(self) -> bool:
            return False

        def has_service_client_api(self) -> bool:
            return True

    monkeypatch.setattr(train_tinker, "TinkerClient", _ServiceModeClient)

    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=tmp_path / "out",
        log_interval_steps=0,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is False
    assert result.error == "log_interval_steps must be greater than 0"


# ---------------------------------------------------------------------------
# _check_early_stopping unit tests
# ---------------------------------------------------------------------------


def _make_config(patience: int = 0, threshold: float = 0.01) -> train_tinker.TinkerTrainingConfig:
    """Helper to build a TinkerTrainingConfig with early stopping knobs."""
    return train_tinker.TinkerTrainingConfig(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold,
    )


def test_check_early_stopping_disabled_when_patience_zero() -> None:
    config = _make_config(patience=0)
    best, count, stop = _check_early_stopping(3.0, 5.0, 0, config, 100)
    assert best == 5.0
    assert count == 0
    assert stop is False


def test_check_early_stopping_noop_when_val_loss_none() -> None:
    config = _make_config(patience=5)
    best, count, stop = _check_early_stopping(None, 5.0, 2, config, 100)
    assert best == 5.0
    assert count == 2
    assert stop is False


def test_check_early_stopping_sets_baseline_on_first_eval() -> None:
    config = _make_config(patience=5)
    best, count, stop = _check_early_stopping(3.0, None, 0, config, 100)
    assert best == 3.0
    assert count == 0
    assert stop is False


def test_check_early_stopping_improvement_resets_counter() -> None:
    config = _make_config(patience=5)
    # 2.9 < 3.0 - 0.01 = 2.99  →  improvement
    best, count, stop = _check_early_stopping(2.9, 3.0, 3, config, 100)
    assert best == 2.9
    assert count == 0
    assert stop is False


def test_check_early_stopping_no_improvement_increments_counter() -> None:
    config = _make_config(patience=5)
    # 3.0 is NOT < 3.0 - 0.01 = 2.99  →  no improvement
    best, count, stop = _check_early_stopping(3.0, 3.0, 2, config, 100)
    assert best == 3.0
    assert count == 3
    assert stop is False


def test_check_early_stopping_triggers_at_patience() -> None:
    config = _make_config(patience=5)
    # evals_without_improvement=4, incremented to 5 = patience  →  stop
    best, count, stop = _check_early_stopping(3.0, 2.5, 4, config, 100)
    assert best == 2.5
    assert count == 5
    assert stop is True


def test_check_early_stopping_threshold_boundary() -> None:
    config = _make_config(patience=5, threshold=0.01)
    # 2.995 is NOT < 3.0 - 0.01 = 2.99  →  NOT an improvement
    best, count, stop = _check_early_stopping(2.995, 3.0, 0, config, 100)
    assert best == 3.0
    assert count == 1
    assert stop is False


# ---------------------------------------------------------------------------
# Early stopping integration test
# ---------------------------------------------------------------------------


def test_train_on_tinker_service_client_stops_early(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    # 4 training examples so there are 4 steps per epoch (batch_size=1)
    train_lines = "\n".join(
        [
            '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u1"},{"role":"assistant","content":"a1"}]}',
            '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u2"},{"role":"assistant","content":"a2"}]}',
            '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u3"},{"role":"assistant","content":"a3"}]}',
            '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u4"},{"role":"assistant","content":"a4"}]}',
        ]
    )
    (data_dir / "train.jsonl").write_text(train_lines + "\n", encoding="utf-8")
    (data_dir / "valid.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"vu"},{"role":"assistant","content":"va"}]}\n',
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    class _TrainingClient:
        forward_call_count = 0

        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": 0.5}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def forward(self, data, loss_fn):  # noqa: ARG002
            _TrainingClient.forward_call_count += 1
            # Return increasing loss to trigger early stopping
            val_loss = 1.0 + _TrainingClient.forward_call_count * 0.5
            return _Future(types.SimpleNamespace(metrics={"loss": val_loss}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            return _Future(types.SimpleNamespace(path=f"tinker://run-123/weights/{name}"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    # Reset class-level counter before test
    _TrainingClient.forward_call_count = 0

    output_dir = tmp_path / "out"
    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        wait_for_completion=True,
        # 4 examples × 3 epochs = 12 steps total with batch_size=1
        epochs=3,
        batch_size=1,
        log_interval_steps=1,
        eval_interval_steps=1,
        eval_at_epoch_end=False,
        early_stopping_patience=2,
        early_stopping_threshold=0.0,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")

    assert result.success is True

    # Should have stopped before all 12 steps
    tinker_run = json.loads((output_dir / "tinker_run.json").read_text(encoding="utf-8"))
    assert tinker_run["early_stopped"] is True
    assert tinker_run["completed_steps"] < 12

    # metrics.jsonl should contain an early_stop entry
    metrics_lines = (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    metric_entries = [json.loads(line) for line in metrics_lines]
    early_stop_entries = [e for e in metric_entries if e["type"] == "early_stop"]
    assert len(early_stop_entries) == 1

    # train.log should mention early stopping
    train_log = (output_dir / "train.log").read_text(encoding="utf-8")
    assert "Early stopping" in train_log


# ---------------------------------------------------------------------------
# Loss normalization test
# ---------------------------------------------------------------------------


def test_train_on_tinker_service_client_loss_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify that train_loss in metrics.jsonl is per-token (raw_loss / completion_tokens)."""
    data_dir = tmp_path / "training"
    data_dir.mkdir()
    # Use longer assistant content "assistant" (9 chars) for more completion tokens
    (data_dir / "train.jsonl").write_text(
        '{"messages":[{"role":"system","content":"s"},{"role":"user","content":"u"},{"role":"assistant","content":"assistant"}]}\n',
        encoding="utf-8",
    )

    class _Future:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class _Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
            tokens = [1 for _ in text]
            if add_special_tokens:
                return [0] + tokens
            return tokens

    raw_loss = 0.9

    class _TrainingClient:
        def get_tokenizer(self):
            return _Tokenizer()

        def forward_backward(self, data, loss_fn):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={"loss": raw_loss}, loss_fn_outputs=[]))

        def optim_step(self, adam_params):  # noqa: ARG002
            return _Future(types.SimpleNamespace(metrics={}))

        def save_state(self, name: str, ttl_seconds=None):  # noqa: ARG002
            return _Future(types.SimpleNamespace(path=f"tinker://run-123/weights/{name}"))

        def get_info(self):
            return types.SimpleNamespace(model_id="run-123")

    class _ServiceClient:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

        def create_lora_training_client(self, base_model: str, rank: int):  # noqa: ARG002
            return _TrainingClient()

    class _Datum:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _ModelInput:
        @classmethod
        def from_ints(cls, tokens):  # noqa: ARG003
            return cls()

    class _TensorData:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    class _AdamParams:
        def __init__(self, **kwargs):  # noqa: ARG002
            pass

    fake_tinker = types.ModuleType("tinker")
    fake_tinker.Client = None
    fake_tinker.ServiceClient = _ServiceClient
    fake_tinker.Datum = _Datum
    fake_tinker.ModelInput = _ModelInput
    fake_tinker.TensorData = _TensorData
    fake_tinker.AdamParams = _AdamParams

    monkeypatch.setattr(train_tinker.TinkerClient, "_check_sdk", lambda self: True)
    monkeypatch.setitem(sys.modules, "tinker", fake_tinker)

    output_dir = tmp_path / "out"
    config = train_tinker.TinkerTrainingConfig(
        dataset_path=data_dir,
        output_dir=output_dir,
        wait_for_completion=True,
        epochs=1,
        batch_size=1,
        log_interval_steps=1,
        eval_at_epoch_end=False,
    )

    result = train_tinker.train_on_tinker(config, api_key="test-key")
    assert result.success is True

    metrics_lines = (output_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    all_entries = [json.loads(line) for line in metrics_lines]
    train_entries = [e for e in all_entries if e["type"] == "train"]
    assert len(train_entries) >= 1

    entry = train_entries[0]

    # With _Tokenizer on messages [system "s", user "u", assistant "assistant"]:
    #   prompt = "s\nu" → encode(add_special=True) = [0,1,1,1] (4 tokens)
    #   completion = "assistant" → encode(add_special=False) = [1]*9 (9 tokens)
    #   weights[1:] has sum = 9 → completion_tokens = 9
    expected_completion_tokens = 9
    expected_per_token_loss = raw_loss / expected_completion_tokens

    assert entry["completion_tokens"] == expected_completion_tokens
    assert entry["train_loss_total"] == pytest.approx(raw_loss)
    assert entry["train_loss"] == pytest.approx(expected_per_token_loss)
    assert entry["train_loss"] < entry["train_loss_total"]
