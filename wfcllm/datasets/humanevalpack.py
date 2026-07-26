"""Hugging Face cache-backed adapter for C++ and Java HumanEvalPack configs."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.registry import register


@register("humanevalpack")
class HumanEvalPackAdapter(DatasetAdapter):
    name = "humanevalpack"
    supported_languages: tuple[str, ...] = ("cpp", "java", "js")

    def __init__(self, dataset_path: str = "data/datasets") -> None:
        self._dataset_path = Path(dataset_path)

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        languages = self.supported_languages if language is None else (language,)
        for selected_language in languages:
            self._validate_language(selected_language)
            for row in self._load_split(selected_language):
                yield self._normalise(row, selected_language)

    def get_sample(
        self,
        task_id: str,
        language: str | None = None,
    ) -> CodeSample:
        for sample in self.iter_samples(language):
            if sample.task_id == task_id:
                return sample
        qualifier = "" if language is None else f" for language={language!r}"
        raise KeyError(f"HumanEvalPack task_id not found{qualifier}: {task_id!r}")

    def _load_split(self, language: str):
        local = self._local_jsonl_split(language)
        if local is not None:
            return local

        from datasets import load_dataset

        dataset = load_dataset(
            "bigcode/humanevalpack",
            language,
            cache_dir=str(self._dataset_path / "humanevalpack"),
            download_mode="reuse_cache_if_exists",
        )
        if "test" not in dataset:
            raise ValueError(
                f"HumanEvalPack language={language!r} has no 'test' split"
            )
        return dataset["test"]

    def _local_jsonl_split(self, language: str) -> list[dict[str, object]] | None:
        candidates = (
            self._dataset_path / "humanevalpack" / language / "test.jsonl",
            self._dataset_path / "humanevalpack" / f"{language}.jsonl",
        )
        for path in candidates:
            if not path.exists():
                continue
            rows: list[dict[str, object]] = []
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"local HumanEvalPack {language} line {line_number} is invalid JSON"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError(
                        f"local HumanEvalPack {language} line {line_number} must be an object"
                    )
                rows.append(row)
            return rows
        return None

    def _validate_language(self, language: str) -> None:
        if language not in self.supported_languages:
            raise ValueError(
                f"HumanEvalPackAdapter does not support language={language!r}; "
                f"supported={self.supported_languages}"
            )

    @staticmethod
    def _normalise(row, language: str) -> CodeSample:
        required = ("task_id", "prompt", "canonical_solution")
        missing = [name for name in required if name not in row]
        if missing:
            raise ValueError(
                f"malformed HumanEvalPack {language} row; missing fields: {missing}"
            )
        metadata = {
            key: value
            for key, value in dict(row).items()
            if key not in required
        }
        return CodeSample(
            task_id=str(row["task_id"]),
            prompt=str(row["prompt"]),
            canonical_solution=str(row["canonical_solution"]),
            language=language,
            metadata=metadata,
        )
