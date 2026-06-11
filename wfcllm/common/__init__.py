from __future__ import annotations

from typing import Any

from wfcllm.datasets.constants import SUPPORTED_DATASETS

__all__ = [
    "SUPPORTED_DATASETS",
    "load_prompts",
]


def __getattr__(name: str) -> Any:
    if name == "load_prompts":
        from wfcllm.datasets.loaders.local import load_prompts

        return load_prompts
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
