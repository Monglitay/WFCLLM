"""AST parsing utilities for node entropy experiment."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def _parse(source: str):
    return _parser.parse(source.encode("utf-8"))


def get_nodes_by_type(source: str, node_type: str) -> list[Node]:
    """Return all AST nodes of the given type, in order of appearance."""
    tree = _parse(source)
    results: list[Node] = []
    _collect(tree.root_node, node_type, results)
    return results


def _collect(node: Node, target_type: str, results: list[Node]) -> None:
    if node.type == target_type:
        results.append(node)
    for child in node.children:
        _collect(child, target_type, results)


def get_node_tokens(source: str, node: Node) -> list[str]:
    """Extract leaf tokens from a tree-sitter node using byte offsets."""
    node_source = source.encode("utf-8")[node.start_byte : node.end_byte]
    # Re-parse the node's source to get clean leaf tokens
    sub_tree = _parser.parse(node_source)
    tokens: list[str] = []
    _collect_leaves(sub_tree.root_node, node_source, tokens)
    return tokens


def _collect_leaves(node: Node, source_bytes: bytes, tokens: list[str]) -> None:
    if not node.children:
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8").strip()
        if text:
            tokens.append(text)
        return
    for child in node.children:
        _collect_leaves(child, source_bytes, tokens)


def get_all_node_types(source: str) -> set[str]:
    """Return all unique node types present in the AST."""
    tree = _parse(source)
    types: set[str] = set()
    _collect_types(tree.root_node, types)
    return types


def _collect_types(node: Node, types: set[str]) -> None:
    types.add(node.type)
    for child in node.children:
        _collect_types(child, types)
