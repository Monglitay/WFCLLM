"""Lightweight JavaScript statement block extraction.

This parser intentionally avoids a new runtime dependency. It recognizes common
top-level and nested JavaScript statements well enough for language registration,
dataset plumbing, and fast semantic unit experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


SIMPLE_STATEMENT_TYPES = frozenset({
    "break_statement",
    "continue_statement",
    "expression_statement",
    "export_statement",
    "import_statement",
    "lexical_declaration",
    "return_statement",
    "throw_statement",
    "variable_declaration",
})

COMPOUND_STATEMENT_TYPES = frozenset({
    "catch_clause",
    "class_declaration",
    "do_statement",
    "else_clause",
    "for_statement",
    "function_declaration",
    "if_statement",
    "switch_statement",
    "try_statement",
    "while_statement",
})

STATEMENT_TYPES = SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES


@dataclass
class StatementBlock:
    """A statement block extracted from JavaScript source code."""

    block_id: str
    block_type: Literal["simple", "compound"]
    node_type: str
    source: str
    start_line: int
    end_line: int
    depth: int
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)


@dataclass
class _OpenBlock:
    block_id: str
    start_depth: int
    brace_depth_after_line: int


def extract_statement_blocks(source: str) -> list[StatementBlock]:
    """Extract JavaScript statement-like blocks with parent-child references."""
    lines = source.splitlines()
    blocks: list[StatementBlock] = []
    open_blocks: list[_OpenBlock] = []
    brace_depth = 0

    for index, line in enumerate(lines):
        line_number = index + 1
        stripped = line.strip()
        leading_closes = len(stripped) - len(stripped.lstrip("}"))

        while open_blocks and brace_depth - leading_closes < open_blocks[-1].brace_depth_after_line:
            open_blocks.pop()

        node_type = _classify_statement(stripped)
        if node_type is not None:
            is_compound = node_type in COMPOUND_STATEMENT_TYPES
            parent_id = open_blocks[-1].block_id if open_blocks else None
            start_depth = len(open_blocks)
            end_line = _find_compound_end(lines, index) if is_compound else line_number
            block_id = str(len(blocks))
            block = StatementBlock(
                block_id=block_id,
                block_type="compound" if is_compound else "simple",
                node_type=node_type,
                source="\n".join(lines[index:end_line]),
                start_line=line_number,
                end_line=end_line,
                depth=start_depth,
                parent_id=parent_id,
            )
            blocks.append(block)
            if parent_id is not None:
                blocks[int(parent_id)].children_ids.append(block_id)

        depth_before_line = brace_depth
        brace_depth += _brace_delta(line)
        if node_type in COMPOUND_STATEMENT_TYPES and "{" in line:
            open_blocks.append(
                _OpenBlock(
                    block_id=str(len(blocks) - 1),
                    start_depth=depth_before_line,
                    brace_depth_after_line=brace_depth,
                )
            )

    return blocks


def _classify_statement(stripped: str) -> str | None:
    if not stripped or stripped.startswith("//"):
        return None

    normalized = stripped.lstrip("}").strip()
    if not normalized:
        return None
    if normalized.startswith("function "):
        return "function_declaration"
    if normalized.startswith("class "):
        return "class_declaration"
    if normalized.startswith("if " ) or normalized.startswith("if("):
        return "if_statement"
    if normalized.startswith("else"):
        return "else_clause"
    if normalized.startswith("for " ) or normalized.startswith("for("):
        return "for_statement"
    if normalized.startswith("while " ) or normalized.startswith("while("):
        return "while_statement"
    if normalized.startswith("do " ) or normalized == "do" or normalized.startswith("do{"):
        return "do_statement"
    if normalized.startswith("switch " ) or normalized.startswith("switch("):
        return "switch_statement"
    if normalized.startswith("try"):
        return "try_statement"
    if normalized.startswith("catch " ) or normalized.startswith("catch("):
        return "catch_clause"
    if normalized.startswith("return") or normalized.startswith("await return"):
        return "return_statement"
    if normalized.startswith(("const ", "let ")):
        return "lexical_declaration"
    if normalized.startswith("var "):
        return "variable_declaration"
    if normalized.startswith("import "):
        return "import_statement"
    if normalized.startswith("export "):
        return "export_statement"
    if normalized.startswith("throw "):
        return "throw_statement"
    if normalized.startswith("break"):
        return "break_statement"
    if normalized.startswith("continue"):
        return "continue_statement"
    if normalized.endswith(";"):
        return "expression_statement"
    return None


def _find_compound_end(lines: list[str], start_index: int) -> int:
    depth = 0
    saw_open = False
    for index in range(start_index, len(lines)):
        line = lines[index]
        depth += _count_braces(line, "{")
        if "{" in line:
            saw_open = True
        depth -= _count_braces(line, "}")
        if saw_open and depth <= 0:
            return index + 1
    return start_index + 1


def _brace_delta(line: str) -> int:
    return _count_braces(line, "{") - _count_braces(line, "}")


def _count_braces(line: str, brace: str) -> int:
    in_single = False
    in_double = False
    in_template = False
    escaped = False
    count = 0
    for char in line:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "'" and not in_double and not in_template:
            in_single = not in_single
            continue
        if char == '"' and not in_single and not in_template:
            in_double = not in_double
            continue
        if char == "`" and not in_single and not in_double:
            in_template = not in_template
            continue
        if char == brace and not in_single and not in_double and not in_template:
            count += 1
    return count
