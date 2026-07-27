from __future__ import annotations

import argparse

import pytest

from wfcllm.cli.runners import _load_gated_generation_samples
from wfcllm.datasets.adapter import CodeSample


class _FakeAdapter:
    supported_languages = ("cpp",)

    def supports(self, language: str) -> bool:
        return language in self.supported_languages

    def iter_samples(self, language: str | None = None):
        assert language == "cpp"
        for index in range(4):
            yield CodeSample(
                task_id=f"cpp/{index}",
                prompt=f"prompt {index}",
                language="cpp",
            )


def test_load_gated_samples_uses_explicit_language_and_slice(monkeypatch) -> None:
    monkeypatch.setattr("wfcllm.datasets.get", lambda name: _FakeAdapter())
    args = argparse.Namespace(
        language="cpp",
        dataset="humanevalpack",
        dataset_path="data/datasets",
        sample_offset=1,
        sample_limit=2,
    )

    rows = _load_gated_generation_samples(
        args,
        {"language": "cpp", "dataset": "humanevalpack"},
    )

    assert rows == [
        {"id": "cpp/1", "prompt": "prompt 1"},
        {"id": "cpp/2", "prompt": "prompt 2"},
    ]


def test_load_gated_samples_rejects_cli_language_mismatch() -> None:
    args = argparse.Namespace(
        language="java",
        dataset="humanevalpack",
        dataset_path="data/datasets",
        sample_offset=None,
        sample_limit=None,
    )

    with pytest.raises(ValueError, match="--language"):
        _load_gated_generation_samples(
            args,
            {"language": "cpp", "dataset": "humanevalpack"},
        )


def test_load_gated_samples_rejects_unsupported_pair(monkeypatch) -> None:
    monkeypatch.setattr("wfcllm.datasets.get", lambda name: _FakeAdapter())
    args = argparse.Namespace(
        language="java",
        dataset="humanevalpack",
        dataset_path="data/datasets",
        sample_offset=None,
        sample_limit=None,
    )

    with pytest.raises(ValueError, match="does not support"):
        _load_gated_generation_samples(
            args,
            {"language": "java", "dataset": "humanevalpack"},
        )
