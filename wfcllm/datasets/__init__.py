"""WFCLLM dataset-adapter layer (spec §5.2)."""
from __future__ import annotations

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import get, names, register

__all__ = ["CodeSample", "DatasetAdapter", "get", "names", "register"]

# Side-effect imports trigger @register decorators. Adapter modules keep
# Hugging Face-backed loader imports inside methods so package import remains
# lightweight for config-only callers.
from wfcllm.datasets import humaneval as _humaneval  # noqa: F401, E402
from wfcllm.datasets import humanevalpack as _humanevalpack  # noqa: F401, E402
from wfcllm.datasets import mbpp as _mbpp  # noqa: F401, E402
