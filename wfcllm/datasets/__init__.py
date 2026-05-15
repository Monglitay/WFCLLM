"""WFCLLM dataset-adapter layer (spec §5.2)."""
from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import get, names, register

__all__ = ["CodeSample", "DatasetAdapter", "get", "names", "register"]
