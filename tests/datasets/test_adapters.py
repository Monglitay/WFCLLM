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


def test_humanevalpack_supports_cpp_and_java():
    from wfcllm import datasets
    adapter = datasets.get("humanevalpack")
    assert set(adapter.supported_languages) == {"java", "cpp"}


def test_humanevalpack_loads_cpp_samples_from_local_cache(monkeypatch, tmp_path):
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter

    calls = []

    def fake_load_dataset(path, name, **kwargs):
        calls.append((path, name, kwargs))
        return {
            "test": [
                {
                    "task_id": "HumanEvalPack/0",
                    "prompt": "int add(int a, int b) {",
                    "declaration": "int add(int a, int b)",
                    "canonical_solution": " return a + b;\n}",
                    "test": "assert(add(1, 2) == 3);",
                    "entry_point": "add",
                }
            ]
        }

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    adapter = HumanEvalPackAdapter(dataset_path=str(tmp_path))
    samples = list(adapter.iter_samples("cpp"))

    assert calls == [
        (
            "bigcode/humanevalpack",
            "cpp",
            {
                "cache_dir": str(tmp_path / "humanevalpack"),
                "download_mode": "reuse_cache_if_exists",
            },
        )
    ]
    assert len(samples) == 1
    assert samples[0].language == "cpp"
    assert samples[0].task_id == "HumanEvalPack/0"
    assert samples[0].canonical_solution.endswith("}")
    assert samples[0].metadata["entry_point"] == "add"


def test_humanevalpack_get_sample_can_disambiguate_language(monkeypatch, tmp_path):
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter

    def fake_load_dataset(path, name, **kwargs):
        return {
            "test": [
                {
                    "task_id": "HumanEvalPack/1",
                    "prompt": f"{name} prompt",
                    "canonical_solution": f"{name} solution",
                }
            ]
        }

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    adapter = HumanEvalPackAdapter(dataset_path=str(tmp_path))

    sample = adapter.get_sample("HumanEvalPack/1", language="java")

    assert sample.language == "java"
    assert sample.prompt == "java prompt"


def test_humanevalpack_rejects_unsupported_language(tmp_path):
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter

    adapter = HumanEvalPackAdapter(dataset_path=str(tmp_path))
    with pytest.raises(ValueError, match="rust"):
        next(adapter.iter_samples("rust"))
