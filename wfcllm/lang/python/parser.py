"""Tree-sitter Python parsing utilities.

Provides a singleton parser and statement block extraction for Python source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language(tspython.language())

# Statement block node type classification
SIMPLE_STATEMENT_TYPES = frozenset({
    "expression_statement",
    "return_statement",
    "assert_statement",
    "import_statement",
    "import_from_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "raise_statement",
    "delete_statement",
    "global_statement",
    "nonlocal_statement",
})

COMPOUND_STATEMENT_TYPES = frozenset({
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
})

STATEMENT_TYPES = SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES


class PythonParser:
    """Singleton tree-sitter Python parser."""

    _instance: PythonParser | None = None
    _parser: Parser

    def __new__(cls) -> PythonParser:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parser = Parser(PY_LANGUAGE)
        return cls._instance

    def parse(self, source: str) -> Tree:
        """Parse Python source code into a tree-sitter Tree."""
        return self._parser.parse(source.encode("utf-8"))


@dataclass
class StatementBlock:
    """A statement block extracted from Python source code."""

    block_id: str
    block_type: Literal["simple", "compound"]
    node_type: str
    source: str
    start_line: int
    end_line: int
    depth: int
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)


def extract_statement_blocks(source: str) -> list[StatementBlock]:
    """Extract all statement blocks from Python source code.

    Returns a flat list of StatementBlock with parent-child references.
    """
    parser = PythonParser()
    tree = parser.parse(source)
    blocks: list[StatementBlock] = []
    _extract_recursive(tree.root_node, source, blocks, depth=0, parent_id=None)
    return blocks


def _extract_recursive(
    node,
    source: str,
    blocks: list[StatementBlock],
    depth: int,
    parent_id: str | None,
) -> None:
    """Recursively walk AST, extracting statement blocks."""
    for child in node.children:
        if child.type not in STATEMENT_TYPES:
            _extract_recursive(child, source, blocks, depth, parent_id)
            continue

        block_id = str(len(blocks))
        is_compound = child.type in COMPOUND_STATEMENT_TYPES

        block = StatementBlock(
            block_id=block_id,
            block_type="compound" if is_compound else "simple",
            node_type=child.type,
            source=child.text.decode("utf-8"),
            start_line=child.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            depth=depth,
            parent_id=parent_id,
        )
        blocks.append(block)

        if is_compound:
            child_count_before = len(blocks)
            _extract_recursive(child, source, blocks, depth + 1, block_id)
            block.children_ids = [str(i) for i in range(child_count_before, len(blocks))]
