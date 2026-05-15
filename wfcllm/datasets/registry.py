"""Dataset-adapter registry (module-level singleton)."""
from __future__ import annotations

from wfcllm.common.registry import Registry
from wfcllm.datasets.adapter import DatasetAdapter

_REGISTRY: Registry[DatasetAdapter] = Registry("dataset")


def register(name: str):
    def deco(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        _REGISTRY.register(name, cls)
        return cls
    return deco


def get(name: str) -> DatasetAdapter:
    return _REGISTRY.get(name)


def names() -> list[str]:
    return _REGISTRY.names()
