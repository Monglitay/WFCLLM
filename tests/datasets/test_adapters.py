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
