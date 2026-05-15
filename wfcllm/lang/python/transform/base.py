"""Base class for all transformation rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def parse_code(source: str) -> Tree:
    """Parse Python source code into a tree-sitter Tree."""
    return _parser.parse(source.encode("utf-8"))


@dataclass
class Match:
    """A location in source code where a rule can be applied."""

    node_type: str
    start_byte: int
    end_byte: int
    original_text: str
    replacement_text: str


class Rule(ABC):
    """Base class for transformation rules."""

    name: str = ""
    category: str = ""
    description: str = ""

    @abstractmethod
    def detect(self, source: str, tree: Tree) -> list[Match]:
        """Find all positions where this rule can be applied."""
        ...

    @abstractmethod
    def apply(self, source: str, matches: list[Match]) -> str:
        """Apply the transformation, returning new source code."""
        ...

    def can_apply(self, source: str) -> bool:
        """Check if this rule can be applied to the given source."""
        tree = parse_code(source)
        return len(self.detect(source, tree)) > 0

    def transform(self, source: str) -> str | None:
        """Detect and apply in one step. Returns None if not applicable."""
        tree = parse_code(source)
        matches = self.detect(source, tree)
        if not matches:
            return None
        return self.apply(source, matches)
