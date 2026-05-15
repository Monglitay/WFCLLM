"""HumanEvalPackAdapter — skeleton-only; spec §5.2 defers data loading to a multi-language phase."""
from __future__ import annotations

from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import register


@register("humanevalpack")
class HumanEvalPackAdapter(DatasetAdapter):
    name = "humanevalpack"
    supported_languages: tuple[str, ...] = ("python", "java", "cpp", "js")

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        raise NotImplementedError(
            "HumanEvalPack Adapter 留作接口；多语言阶段实现"
        )

    def get_sample(self, task_id: str) -> CodeSample:
        raise NotImplementedError(
            "HumanEvalPack Adapter 留作接口；多语言阶段实现"
        )
