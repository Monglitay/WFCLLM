"""Language-neutral Tree-sitter statement block extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tree_sitter import Parser


@dataclass
class SourceBlock:
    """A statement-like AST node with stable source and nesting metadata."""

    block_id: str
    block_type: Literal["simple", "compound"]
    node_type: str
    source: str
    start_line: int
    end_line: int
    depth: int
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)


def extract_statement_blocks(
    source: str,
    *,
    parser: Parser,
    simple_types: frozenset[str],
    compound_types: frozenset[str],
) -> list[SourceBlock]:
    """Return a flat, source-ordered list of statement-like AST nodes."""
    tree = parser.parse(source.encode("utf-8"))
    blocks: list[SourceBlock] = []
    _extract_recursive(
        tree.root_node,
        blocks=blocks,
        simple_types=simple_types,
        compound_types=compound_types,
        depth=0,
        parent_id=None,
    )
    return blocks


def _extract_recursive(
    node: Any,
    *,
    blocks: list[SourceBlock],
    simple_types: frozenset[str],
    compound_types: frozenset[str],
    depth: int,
    parent_id: str | None,
) -> None:
    statement_types = simple_types | compound_types
    for child in node.children:
        if child.type not in statement_types:
            _extract_recursive(
                child,
                blocks=blocks,
                simple_types=simple_types,
                compound_types=compound_types,
                depth=depth,
                parent_id=parent_id,
            )
            continue

        block_id = str(len(blocks))
        is_compound = child.type in compound_types
        block = SourceBlock(
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
            child_start = len(blocks)
            _extract_recursive(
                child,
                blocks=blocks,
                simple_types=simple_types,
                compound_types=compound_types,
                depth=depth + 1,
                parent_id=block_id,
            )
            # Match the established Python adapter contract: children_ids
            # contains every descendant block, while parent_id identifies the
            # direct parent.
            block.children_ids = [
                candidate.block_id for candidate in blocks[child_start:]
            ]
