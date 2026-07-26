"""Tree-sitter statement-unit extraction for non-Python languages."""

from __future__ import annotations

from collections import defaultdict

from tree_sitter import Node

from wfcllm.lang import get as get_language_adapter
from wfcllm.windowing.contracts import StatementUnit
from wfcllm.windowing.python import PythonStatementUnitExtractor


_BODY_TYPES = frozenset(
    {
        "block",
        "class_body",
        "compound_statement",
        "constructor_body",
        "enum_body",
        "interface_body",
        "record_body",
        "switch_block",
    }
)
_HARD_BOUNDARY_TYPES = {
    "cpp": frozenset(
        {
            "break_statement",
            "class_specifier",
            "continue_statement",
            "function_definition",
            "goto_statement",
            "namespace_definition",
            "struct_specifier",
            "throw_statement",
        }
    ),
    "java": frozenset(
        {
            "break_statement",
            "class_declaration",
            "constructor_declaration",
            "continue_statement",
            "enum_declaration",
            "interface_declaration",
            "method_declaration",
            "record_declaration",
            "throw_statement",
        }
    ),
}


class TreeSitterStatementUnitExtractor:
    """Convert registered C++/Java Tree-sitter nodes to formal units."""

    def __init__(self, language: str) -> None:
        if language not in {"cpp", "java"}:
            raise ValueError("Tree-sitter statement extractor supports cpp or java")
        self.language = language
        adapter = get_language_adapter(language)
        statement_types = adapter.statement_types()
        self._simple_types = statement_types.simple
        self._compound_types = statement_types.compound
        if language == "cpp":
            from wfcllm.lang.cpp.parser import CppParser

            self._parser = CppParser()
        else:
            from wfcllm.lang.java.parser import JavaParser

            self._parser = JavaParser()

    def extract(self, source: str, *, function_name: str | None = None) -> list[StatementUnit]:
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if function_name is not None:
            raise ValueError("function_name selection is only supported for Python")
        source_bytes = source.encode("utf-8")
        root = self._parser.parse(source).root_node
        units: list[StatementUnit] = []
        ordinals: defaultdict[tuple[int, int, str], int] = defaultdict(int)

        def walk(
            node: Node,
            parent_path: tuple[str, ...],
            depth: int,
            *,
            skip_before: int | None = None,
        ) -> None:
            owner_key = (node.start_byte, node.end_byte, node.type)
            for child in node.named_children:
                if child.is_missing:
                    continue
                if skip_before is not None and child.end_byte <= skip_before:
                    continue
                compound = child.type in self._compound_types
                simple = child.type in self._simple_types
                if not compound and not simple:
                    walk(child, (*parent_path, child.type), depth, skip_before=skip_before)
                    continue

                ordinal = ordinals[owner_key]
                ordinals[owner_key] += 1
                end_byte = (
                    _compound_header_end(child, source_bytes)
                    if compound
                    else child.end_byte
                )
                if end_byte > child.start_byte:
                    uncertain = _has_uncertainty(child, child.start_byte, end_byte)
                    hard_boundary = (
                        uncertain
                        or child.type in _HARD_BOUNDARY_TYPES[self.language]
                    )
                    units.append(
                        StatementUnit(
                            unit_id=str(len(units)),
                            node_type=child.type,
                            text=source_bytes[child.start_byte:end_byte].decode("utf-8"),
                            start_byte=child.start_byte,
                            end_byte=end_byte,
                            start_line=child.start_point[0] + 1,
                            end_line=_line_for_byte(source_bytes, end_byte),
                            depth=depth,
                            parent_path=parent_path,
                            direct_parent_type=node.type,
                            direct_child_ordinal=ordinal,
                            eligible=not hard_boundary,
                            hard_boundary=hard_boundary,
                            compound_header=compound,
                        )
                    )
                if compound:
                    walk(
                        child,
                        (*parent_path, child.type),
                        depth + 1,
                        skip_before=end_byte,
                    )

        walk(root, (root.type,), 0)
        units.sort(key=lambda unit: (unit.start_byte, unit.end_byte))
        return [
            StatementUnit(**{**unit.__dict__, "unit_id": str(index)})
            for index, unit in enumerate(units)
        ]


class JavaScriptStatementUnitExtractor:
    """Dependency-light JS statement-unit extractor for gated experiments."""

    _HARD_BOUNDARY_TYPES = frozenset(
        {
            "break_statement",
            "class_declaration",
            "continue_statement",
            "export_statement",
            "function_declaration",
            "import_statement",
            "throw_statement",
        }
    )

    def extract(self, source: str, *, function_name: str | None = None) -> list[StatementUnit]:
        if not isinstance(source, str):
            raise ValueError("source must be a string")
        if function_name is not None:
            raise ValueError("function_name selection is only supported for Python")
        from wfcllm.lang.js.parser import COMPOUND_STATEMENT_TYPES, extract_statement_blocks

        blocks = extract_statement_blocks(source)
        source_bytes = source.encode("utf-8")
        line_starts = _line_starts(source)
        source_lines = source.splitlines()
        by_id = {block.block_id: block for block in blocks}
        ordinals: defaultdict[str, int] = defaultdict(int)
        units: list[StatementUnit] = []
        for block in blocks:
            parent = by_id.get(block.parent_id) if block.parent_id is not None else None
            direct_parent = parent.node_type if parent is not None else "program"
            parent_path = _js_parent_path(block, by_id)
            ordinal_key = block.parent_id or "program"
            ordinal = ordinals[ordinal_key]
            ordinals[ordinal_key] += 1
            start_byte = line_starts[max(0, block.start_line - 1)]
            if block.node_type in COMPOUND_STATEMENT_TYPES:
                line = source_lines[block.start_line - 1] if block.start_line - 1 < len(source_lines) else ""
                header_end = line.find("{")
                end_byte = start_byte + (header_end + 1 if header_end >= 0 else len(line.encode("utf-8")))
                end_line = block.start_line
            else:
                end_byte = line_starts[min(block.end_line, len(line_starts) - 1)]
                while end_byte > start_byte and source_bytes[end_byte - 1:end_byte] in {b"\n", b"\r"}:
                    end_byte -= 1
                end_line = block.end_line
            if end_byte <= start_byte:
                continue
            hard_boundary = block.node_type in self._HARD_BOUNDARY_TYPES
            units.append(
                StatementUnit(
                    unit_id=str(len(units)),
                    node_type=block.node_type,
                    text=source_bytes[start_byte:end_byte].decode("utf-8"),
                    start_byte=start_byte,
                    end_byte=end_byte,
                    start_line=block.start_line,
                    end_line=end_line,
                    depth=block.depth,
                    parent_path=parent_path,
                    direct_parent_type=direct_parent,
                    direct_child_ordinal=ordinal,
                    eligible=not hard_boundary,
                    hard_boundary=hard_boundary,
                    compound_header=block.node_type in COMPOUND_STATEMENT_TYPES,
                )
            )
        return units


def _line_starts(source: str) -> list[int]:
    starts = [0]
    total = 0
    for line in source.splitlines(keepends=True):
        total += len(line.encode("utf-8"))
        starts.append(total)
    encoded_len = len(source.encode("utf-8"))
    if starts[-1] != encoded_len:
        starts.append(encoded_len)
    return starts


def _js_parent_path(block, by_id: dict[str, object]) -> tuple[str, ...]:
    chain = []
    parent_id = block.parent_id
    while parent_id is not None and parent_id in by_id:
        parent = by_id[parent_id]
        if not chain or chain[-1] != parent.node_type:
            chain.append(parent.node_type)
        parent_id = parent.parent_id
    return ("program", *reversed(chain))


def get_statement_unit_extractor(language: str):
    if language == "python":
        return PythonStatementUnitExtractor()
    if language == "js":
        return JavaScriptStatementUnitExtractor()
    return TreeSitterStatementUnitExtractor(language)


def _compound_header_end(node: Node, source: bytes) -> int:
    body_candidates = [
        child for child in node.named_children if child.type in _BODY_TYPES
    ]
    if body_candidates:
        body = min(body_candidates, key=lambda child: child.start_byte)
        if body.start_byte < len(source) and source[body.start_byte:body.start_byte + 1] == b"{":
            return body.start_byte + 1
        return body.start_byte
    nested_statements = [
        child
        for child in node.named_children
        if child.type.endswith("statement") and child.start_byte > node.start_byte
    ]
    if nested_statements:
        return min(child.start_byte for child in nested_statements)
    return node.end_byte


def _has_uncertainty(node: Node, start_byte: int, end_byte: int) -> bool:
    if node.type == "ERROR" or node.is_missing:
        return True
    if not node.has_error:
        return False
    return any(
        child.start_byte < end_byte
        and child.end_byte > start_byte
        and _has_uncertainty(child, start_byte, end_byte)
        for child in node.children
    )


def _line_for_byte(source: bytes, end_byte: int) -> int:
    return source[:end_byte].count(b"\n") + 1
