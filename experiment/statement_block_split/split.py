"""Split Python code into statement blocks using tree-sitter."""

import json
from datetime import datetime, timezone

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from config import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    STATEMENT_TYPES,
)

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extract_blocks(code: str) -> list[dict]:
    """Parse code and recursively extract all statement blocks.

    Returns a flat list of block dicts with parent-child references.
    """
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node
    blocks: list[dict] = []
    _extract_recursive(root, blocks, depth=0, parent_id=None)
    return blocks


def _extract_recursive(
    node, blocks: list[dict], depth: int, parent_id: int | None
) -> None:
    """Recursively walk AST children, extracting statement blocks."""
    for child in node.children:
        if child.type not in STATEMENT_TYPES:
            # Recurse into non-statement nodes (e.g., 'block', 'elif_clause')
            # to find nested statements, keeping same depth and parent.
            _extract_recursive(child, blocks, depth, parent_id)
            continue

        block_id = len(blocks)
        is_compound = child.type in COMPOUND_STATEMENT_TYPES

        block = {
            "id": block_id,
            "type": "compound" if is_compound else "simple",
            "node_type": child.type,
            "source": child.text.decode("utf-8"),
            "start_line": child.start_point[0] + 1,
            "end_line": child.end_point[0] + 1,
            "depth": depth,
            "parent_id": parent_id,
            "children_ids": [],
        }
        blocks.append(block)

        if is_compound:
            child_count_before = len(blocks)
            _extract_recursive(child, blocks, depth + 1, block_id)
            block["children_ids"] = list(
                range(child_count_before, len(blocks))
            )
