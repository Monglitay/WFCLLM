"""WFCLLM dataset-adapter layer (spec §5.2)."""
from __future__ import annotations

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import get as _registry_get
from wfcllm.datasets.registry import names as _registry_names
from wfcllm.datasets.registry import register

__all__ = ["CodeSample", "DatasetAdapter", "get", "names", "register"]

_BUILTIN_ADAPTERS_REGISTERED = False


def _ensure_builtin_adapters_registered() -> None:
    global _BUILTIN_ADAPTERS_REGISTERED
    if _BUILTIN_ADAPTERS_REGISTERED:
        return

    from wfcllm.datasets import humaneval as _humaneval  # noqa: F401
    from wfcllm.datasets import humanevalpack as _humanevalpack  # noqa: F401
    from wfcllm.datasets import mbpp as _mbpp  # noqa: F401

    _BUILTIN_ADAPTERS_REGISTERED = True


def get(name: str) -> DatasetAdapter:
    _ensure_builtin_adapters_registered()
    return _registry_get(name)


def names() -> list[str]:
    _ensure_builtin_adapters_registered()
    return _registry_names()
