"""WFCLLM dataset-adapter layer (spec §5.2)."""
from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import get, names, register

__all__ = ["CodeSample", "DatasetAdapter", "get", "names", "register"]

# Side-effect imports trigger @register decorators
from wfcllm.datasets import humaneval as _humaneval  # noqa: F401, E402
from wfcllm.datasets import mbpp as _mbpp  # noqa: F401, E402
