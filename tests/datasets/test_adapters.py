"""Tests for DatasetAdapter implementations and registry."""
from __future__ import annotations

from unittest.mock import ANY

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


def test_mbpp_generation_uses_only_official_test_split(monkeypatch, tmp_path):
    from wfcllm.datasets.loaders.local import load_mbpp_generation_samples

    def row(task_id):
        return {
            "task_id": task_id,
            "text": f"Double a number, task {task_id}.",
            "test_list": ["assert twice(2) == 4"],
            "code": "canonical solution must never enter the generation prompt",
        }

    monkeypatch.setattr(
        "wfcllm.datasets.loaders.local.load_dataset",
        lambda *_args, **_kwargs: {
            "train": [row(601)],
            "test": [row(11), row(12)],
            "validation": [row(511)],
            "prompt": [row(1)],
        },
    )

    samples = load_mbpp_generation_samples(str(tmp_path))

    assert [item["id"] for item in samples] == ["mbpp/11", "mbpp/12"]
    assert all("canonical solution" not in item["prompt"] for item in samples)
    assert all("twice" in item["prompt"] for item in samples)


def test_mbpp_generation_rejects_missing_test_split(monkeypatch, tmp_path):
    from wfcllm.datasets.loaders.local import load_mbpp_generation_samples

    monkeypatch.setattr(
        "wfcllm.datasets.loaders.local.load_dataset",
        lambda *_args, **_kwargs: {"train": []},
    )

    with pytest.raises(ValueError, match="official test split"):
        load_mbpp_generation_samples(str(tmp_path))


def test_mbpp_adapter_uses_interface_aware_generation_prompt(monkeypatch, tmp_path):
    from wfcllm.datasets.mbpp import MBPPAdapter

    monkeypatch.setattr(
        "wfcllm.datasets.mbpp.load_mbpp_generation_samples",
        lambda _path: [
            {
                "id": "mbpp/1",
                "prompt": (
                    "Natural task.\n\nInterface requirements:\n"
                    "Define the function exactly named `target`."
                ),
                "interface_extraction_status": "interface_aware",
                "interface_function_name": "target",
                "interface_positional_arities": [1],
                "interface_helper_classes": [],
            }
        ],
    )

    sample = next(MBPPAdapter(dataset_path=str(tmp_path)).iter_samples("python"))

    assert sample.prompt.endswith("exactly named `target`.")
    assert sample.metadata == {
        "interface_extraction_status": "interface_aware",
        "interface_function_name": "target",
        "interface_positional_arities": [1],
        "interface_helper_classes": [],
    }


def test_humanevalpack_supports_cpp_and_java():
    from wfcllm import datasets
    adapter = datasets.get("humanevalpack")
    assert set(adapter.supported_languages) == {"java", "cpp", "js"}


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
                "download_config": ANY,
            },
        )
    ]
    assert calls[0][2]["download_config"].local_files_only is True
    assert len(samples) == 1
    assert samples[0].language == "cpp"
    assert samples[0].task_id == "HumanEvalPack/0"
    assert samples[0].prompt == "int add(int a, int b) {"
    assert samples[0].metadata == {}


def test_humanevalpack_smoke_limit_preserves_hf_dataset_rows(
    monkeypatch, tmp_path
):
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter

    class FakeDatasetSplit:
        def __init__(self, rows):
            self._rows = rows

        def __iter__(self):
            return iter(self._rows)

        def __len__(self):
            return len(self._rows)

        def __getitem__(self, index):
            if isinstance(index, slice):
                selected = self._rows[index]
                return {
                    key: [row[key] for row in selected]
                    for key in selected[0]
                }
            return self._rows[index]

        def select(self, indices):
            return FakeDatasetSplit([self._rows[index] for index in indices])

    def fake_load_dataset(*_args, **_kwargs):
        return {
            "test": FakeDatasetSplit(
                [
                    {
                        "task_id": f"HumanEvalPack/{index}",
                        "prompt": f"int task_{index}();",
                    }
                    for index in range(3)
                ]
            )
        }

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setenv("WFCLLM_SMOKE_SAMPLE_COUNT", "2")

    samples = list(HumanEvalPackAdapter(dataset_path=str(tmp_path)).iter_samples("cpp"))

    assert [sample.task_id for sample in samples] == [
        "HumanEvalPack/0",
        "HumanEvalPack/1",
    ]
    assert [sample.prompt for sample in samples] == [
        "int task_0();",
        "int task_1();",
    ]


def test_humanevalpack_prefers_local_jsonl_fixture(monkeypatch, tmp_path):
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter

    def fail_load_dataset(*_args, **_kwargs):
        raise AssertionError("HF loader should not be used when local JSONL exists")

    monkeypatch.setattr("datasets.load_dataset", fail_load_dataset)
    local_dir = tmp_path / "humanevalpack" / "java"
    local_dir.mkdir(parents=True)
    (local_dir / "test.jsonl").write_text(
        '{"task_id":"Java/local-0","prompt":"class Solution {"}\n',
        encoding="utf-8",
    )

    samples = list(HumanEvalPackAdapter(str(tmp_path)).iter_samples("java"))

    assert len(samples) == 1
    assert samples[0].task_id == "Java/local-0"
    assert samples[0].language == "java"
    assert samples[0].metadata == {}


def test_humanevalpack_get_sample_can_disambiguate_language(monkeypatch, tmp_path):
    from wfcllm.datasets.humanevalpack import HumanEvalPackAdapter

    def fake_load_dataset(path, name, **kwargs):
        return {
            "test": [
                {
                    "task_id": "HumanEvalPack/1",
                    "prompt": f"{name} prompt",
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
