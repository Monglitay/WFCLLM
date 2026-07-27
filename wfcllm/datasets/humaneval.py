"""HumanEvalAdapter — wraps wfcllm.datasets.loaders.local.load_prompts."""
from __future__ import annotations

from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import register


@register("humaneval")
class HumanEvalAdapter(DatasetAdapter):
    name = "humaneval"
    supported_languages: tuple[str, ...] = ("python",)

    def __init__(self, dataset_path: str = "data/datasets") -> None:
        self._dataset_path = dataset_path

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        from wfcllm.datasets.loaders.local import load_prompts

        if language is not None and language not in self.supported_languages:
            raise ValueError(
                f"HumanEvalAdapter does not support language={language!r}; "
                f"supported={self.supported_languages}"
            )
        rows = load_prompts("humaneval", self._dataset_path)
        for row in rows:
            yield CodeSample(
                task_id=row["id"],
                prompt=row["prompt"],
                language="python",
            )

    def get_sample(self, task_id: str) -> CodeSample:
        for sample in self.iter_samples():
            if sample.task_id == task_id:
                return sample
        raise KeyError(f"HumanEval task_id not found: {task_id!r}")
