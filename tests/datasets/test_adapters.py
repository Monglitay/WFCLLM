"""Tests for DatasetAdapter implementations and registry."""
from __future__ import annotations

import pytest


def test_humaneval_registered():
    from wfcllm import datasets
    assert "humaneval" in datasets.names()


def test_mbpp_registered():
    from wfcllm import datasets
    assert "mbpp" in datasets.names()


def test_humanevalpack_registered():
    from wfcllm import datasets
    assert "humanevalpack" in datasets.names()


def test_unknown_dataset_raises():
    from wfcllm import datasets
    with pytest.raises(KeyError, match="unknown dataset"):
        datasets.get("does-not-exist")


def test_humaneval_adapter_supports_python_only():
    from wfcllm import datasets
    adapter = datasets.get("humaneval")
    assert adapter.supports("python")
    assert not adapter.supports("java")


def test_humaneval_adapter_rejects_unsupported_language_at_iter():
    from wfcllm import datasets
    adapter = datasets.get("humaneval")
    with pytest.raises(ValueError, match="java"):
        next(adapter.iter_samples(language="java"))


def test_mbpp_adapter_name():
    from wfcllm import datasets
    assert datasets.get("mbpp").name == "mbpp"
