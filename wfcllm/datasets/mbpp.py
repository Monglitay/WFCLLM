"""MBPPAdapter — wraps wfcllm.datasets.loaders.local.load_prompts."""
from __future__ import annotations

import logging
from typing import Iterator

from wfcllm.datasets.adapter import CodeSample, DatasetAdapter
from wfcllm.datasets.loaders.local import load_mbpp_generation_samples
from wfcllm.datasets.registry import register


logger = logging.getLogger(__name__)


@register("mbpp")
class MBPPAdapter(DatasetAdapter):
    name = "mbpp"
    supported_languages: tuple[str, ...] = ("python",)

    def __init__(self, dataset_path: str = "data/datasets") -> None:
        self._dataset_path = dataset_path

    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]:
        if language is not None and language not in self.supported_languages:
            raise ValueError(
                f"MBPPAdapter does not support language={language!r}; "
                f"supported={self.supported_languages}"
            )
        rows = load_mbpp_generation_samples(self._dataset_path)
        interface_aware = sum(
            row["interface_extraction_status"] == "interface_aware" for row in rows
        )
        logger.info(
            "MBPP interface extraction interface_aware=%d fallback=%d total=%d",
            interface_aware,
            len(rows) - interface_aware,
            len(rows),
        )
        for row in rows:
            yield CodeSample(
                task_id=row["id"],
                prompt=row["prompt"],
                language="python",
                metadata={
                    "interface_extraction_status": row[
                        "interface_extraction_status"
                    ],
                    "interface_function_name": row["interface_function_name"],
                    "interface_positional_arities": row[
                        "interface_positional_arities"
                    ],
                    "interface_helper_classes": row[
                        "interface_helper_classes"
                    ],
                },
            )

    def get_sample(self, task_id: str) -> CodeSample:
        for sample in self.iter_samples():
            if sample.task_id == task_id:
                return sample
        raise KeyError(f"MBPP task_id not found: {task_id!r}")
