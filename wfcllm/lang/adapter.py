"""Language adapter ABC for the WFCLLM pluggable language layer."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class StatementTypes:
    """Per-language partition of AST node types into simple / compound statement blocks."""

    simple: frozenset[str]
    compound: frozenset[str]

    @property
    def all(self) -> frozenset[str]:
        return self.simple | self.compound


class LanguageAdapter(ABC):
    """Pluggable per-language strategy used by encoder, watermark, and extract phases."""

    name: str = ""

    @abstractmethod
    def statement_types(self) -> StatementTypes: ...

    @abstractmethod
    def extract_blocks(self, source: str) -> list:
        """Parse `source` and return a flat list of statement blocks."""

    @abstractmethod
    def positive_rules(self) -> list:
        """Return the canonical list of positive (semantic-equivalent) transform rules."""

    @abstractmethod
    def negative_rules(self) -> list:
        """Return the canonical list of negative (semantic-breaking) transform rules."""
