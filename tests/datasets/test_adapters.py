"""Tests for DatasetAdapter implementations and registry."""
from __future__ import annotations

import pytest


def test_direct_registry_names_include_builtins_after_package_import():
    import wfcllm.datasets  # noqa: F401
    from wfcllm.datasets.registry import names as registry_names

    assert {"humaneval", "mbpp"}.issubset(registry_names())


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


def test_humanevalpack_iter_raises_not_implemented():
    from wfcllm import datasets
    adapter = datasets.get("humanevalpack")
    with pytest.raises(NotImplementedError, match="HumanEvalPack"):
        next(adapter.iter_samples())


def test_humanevalpack_supports_four_languages():
    from wfcllm import datasets
    adapter = datasets.get("humanevalpack")
    assert set(adapter.supported_languages) == {"python", "java", "cpp", "js"}
