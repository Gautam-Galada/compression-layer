from importlib import import_module
from pkgutil import iter_modules
from typing import Protocol

from .dataset import DownstreamExample


class PrepareFn(Protocol):
    def __call__(
        self,
        *,
        split: str,
        limit: int | None = None,
        seed: int = 42,
        max_chunks: int = 8,
        **kwargs: object,
    ) -> list[DownstreamExample]: ...


PREPARE_REGISTRY: dict[str, PrepareFn] = {}


def discover_benchmark_names() -> tuple[str, ...]:
    package_name = "src.evaluation.downstream.benchmarks"
    package = import_module(package_name)
    benchmark_names: list[str] = []
    for module_info in iter_modules(package.__path__):
        module = import_module(f"{package_name}.{module_info.name}")
        benchmark_name = getattr(module, "BENCHMARK_NAME", None)
        if isinstance(benchmark_name, str):
            benchmark_names.append(benchmark_name)
    return tuple(sorted(benchmark_names))


def load_benchmark_modules() -> None:
    discover_benchmark_names()


def get_prepare_registry() -> dict[str, PrepareFn]:
    load_benchmark_modules()
    missing = sorted(set(discover_benchmark_names()) - set(PREPARE_REGISTRY))
    if missing:
        missing_names = ", ".join(missing)
        raise ValueError(f"Benchmark modules missing prepare registration: {missing_names}")
    return PREPARE_REGISTRY


def get_supported_benchmarks() -> tuple[str, ...]:
    return tuple(sorted(get_prepare_registry()))
