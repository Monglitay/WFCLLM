"""Dataset adapter ABC (spec §5.2)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator


@dataclass(frozen=True)
class CodeSample:
    """A code-generation benchmark sample, normalised across datasets."""

    task_id: str
    prompt: str
    canonical_solution: str
    language: str
    metadata: dict = field(default_factory=dict)


class DatasetAdapter(ABC):
    """Pluggable per-dataset strategy for loading code-generation benchmarks."""

    name: str = ""
    supported_languages: tuple[str, ...] = ()

    @abstractmethod
    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]: ...

    @abstractmethod
    def get_sample(self, task_id: str) -> CodeSample: ...

    def task_ids(self, language: str | None = None) -> list[str]:
        return [s.task_id for s in self.iter_samples(language)]

    def supports(self, language: str) -> bool:
        return language in self.supported_languages
