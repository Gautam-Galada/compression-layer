import sys
import types
from pathlib import Path

import pytest

import src.training.train_tinker as train_tinker


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
