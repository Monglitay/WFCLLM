"""Tree-sitter-based Python statement unit extraction."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

from tree_sitter import Node

from wfcllm.lang.python.parser import PythonParser
from wfcllm.windowing.contracts import StatementUnit

_COMPOUND_TYPES = frozenset(
    {
        "class_definition",
        "for_statement",
        "function_definition",
        "if_statement",
        "match_statement",
        "try_statement",
        "while_statement",
        "with_statement",
    }
)

_CLAUSE_TYPES = frozenset(
    {
        "case_clause",
        "elif_clause",
        "else_clause",
        "except_clause",
        "finally_clause",
    }
)

_FIXED_EXCLUDED_TYPES = frozenset(
    {
        "assert_statement",
        "break_statement",
        "class_definition",
        "continue_statement",
        "delete_statement",
        "function_definition",
        "future_import_statement",
        "global_statement",
        "import_from_statement",
        "import_statement",
        "nonlocal_statement",
        "pass_statement",
        "raise_statement",
    }
)

_IGNORED_NAMED_TYPES = frozenset({"comment"})


class PythonStatementUnitExtractor:
    """Extract parser-defined statement and compound-header units."""

    def __init__(self) -> None:
        self._parser = PythonParser()

    def extract(
        self, source: str, *, function_name: str | None = None
    ) -> list[StatementUnit]:
        """Return source-ordered statement units from *source*.

        When ``function_name`` is provided, only the first function definition
        with that name and the statements nested inside it are returned.
        """
        source_bytes = source.encode("utf-8")
        root = self._parser.parse(source).root_node
        builder = _UnitBuilder(source_bytes)
        builder.walk_container(
            root,
            parent_path=(root.type,),
            depth=0,
        )

        units = builder.units
        if function_name is None:
            return units

        target = _find_function(root, source_bytes, function_name)
        if target is None:
            return []
        selected = [
            unit
            for unit in units
            if target.start_byte <= unit.start_byte and unit.end_byte <= target.end_byte
        ]
        return [replace(unit, unit_id=str(index)) for index, unit in enumerate(selected)]


class _UnitBuilder:
    def __init__(self, source_bytes: bytes) -> None:
        self.source_bytes = source_bytes
        self.units: list[StatementUnit] = []
        self._next_ordinal: defaultdict[tuple[int, int, str], int] = defaultdict(int)

    def walk_container(
        self,
        container: Node,
        *,
        parent_path: tuple[str, ...],
        depth: int,
    ) -> None:
        for child in container.named_children:
            if child.type in _IGNORED_NAMED_TYPES or child.is_missing:
                continue
            if child.type == "decorated_definition":
                self._walk_decorated_definition(
                    child,
                    parent_path=parent_path,
                    depth=depth,
                )
            elif child.type in _COMPOUND_TYPES:
                self._walk_compound(
                    child,
                    owner=container,
                    parent_path=parent_path,
                    depth=depth,
                )
            elif child.type in _CLAUSE_TYPES:
                self._walk_clause(
                    child,
                    owner=container,
                    parent_path=parent_path,
                    depth=depth,
                )
            else:
                self._append_unit(
                    child,
                    owner=container,
                    parent_path=parent_path,
                    depth=depth,
                    compound_header=False,
                )

    def _walk_decorated_definition(
        self,
        node: Node,
        *,
        parent_path: tuple[str, ...],
        depth: int,
    ) -> None:
        decorated_path = (*parent_path, node.type)
        for child in node.named_children:
            if child.type in {"function_definition", "class_definition"}:
                self._walk_compound(
                    child,
                    owner=node,
                    parent_path=decorated_path,
                    depth=depth,
                )

    def _walk_compound(
        self,
        node: Node,
        *,
        owner: Node,
        parent_path: tuple[str, ...],
        depth: int,
    ) -> None:
        self._append_unit(
            node,
            owner=owner,
            parent_path=parent_path,
            depth=depth,
            compound_header=True,
        )
        child_path = (*parent_path, node.type)
        for child in node.named_children:
            if child.is_missing:
                continue
            if child.type == "block":
                self.walk_container(
                    child,
                    parent_path=(*child_path, child.type),
                    depth=depth + 1,
                )
            elif child.type in _CLAUSE_TYPES:
                self._walk_clause(
                    child,
                    owner=node,
                    parent_path=child_path,
                    depth=depth,
                )
            elif child.type == "ERROR":
                self._append_unit(
                    child,
                    owner=node,
                    parent_path=child_path,
                    depth=depth + 1,
                    compound_header=False,
                )

    def _walk_clause(
        self,
        node: Node,
        *,
        owner: Node,
        parent_path: tuple[str, ...],
        depth: int,
    ) -> None:
        self._append_unit(
            node,
            owner=owner,
            parent_path=parent_path,
            depth=depth,
            compound_header=True,
        )
        clause_path = (*parent_path, node.type)
        for child in node.named_children:
            if child.is_missing:
                continue
            if child.type == "block":
                self.walk_container(
                    child,
                    parent_path=(*clause_path, child.type),
                    depth=depth + 1,
                )
            elif child.type in _CLAUSE_TYPES:
                self._walk_clause(
                    child,
                    owner=node,
                    parent_path=clause_path,
                    depth=depth,
                )
            elif child.type == "ERROR":
                self._append_unit(
                    child,
                    owner=node,
                    parent_path=clause_path,
                    depth=depth + 1,
                    compound_header=False,
                )

    def _append_unit(
        self,
        node: Node,
        *,
        owner: Node,
        parent_path: tuple[str, ...],
        depth: int,
        compound_header: bool,
    ) -> None:
        start_byte = node.start_byte
        end_byte = _header_end_byte(node) if compound_header else node.end_byte
        if end_byte <= start_byte:
            return
        ordinal_key = (owner.start_byte, owner.end_byte, owner.type)
        ordinal = self._next_ordinal[ordinal_key]
        self._next_ordinal[ordinal_key] += 1
        eligible = (
            node.type not in _FIXED_EXCLUDED_TYPES
            and not _has_parse_uncertainty(node, start_byte, end_byte)
        )

        self.units.append(
            StatementUnit(
                unit_id=str(len(self.units)),
                node_type=node.type,
                text=self.source_bytes[start_byte:end_byte].decode("utf-8"),
                start_byte=start_byte,
                end_byte=end_byte,
                start_line=node.start_point[0] + 1,
                end_line=_end_line(node, end_byte),
                depth=depth,
                parent_path=parent_path,
                direct_parent_type=owner.type,
                direct_child_ordinal=ordinal,
                eligible=eligible,
                hard_boundary=not eligible,
                compound_header=compound_header,
            )
        )


def _header_end_byte(node: Node) -> int:
    for child in node.children:
        if not child.is_named and child.type == ":":
            return child.end_byte
    raise ValueError(f"compound node {node.type!r} has no header colon")


def _end_line(node: Node, end_byte: int) -> int:
    if end_byte == node.end_byte:
        return node.end_point[0] + 1
    for child in node.children:
        if child.end_byte == end_byte:
            return child.end_point[0] + 1
    raise ValueError(f"node {node.type!r} has an invalid unit end byte")


def _has_parse_uncertainty(node: Node, start_byte: int, end_byte: int) -> bool:
    if not node.has_error and node.type != "ERROR" and not node.is_missing:
        return False
    if node.type == "ERROR" or node.is_missing:
        return True
    for child in node.children:
        if child.is_missing:
            if start_byte <= child.start_byte <= end_byte:
                return True
        elif child.start_byte < end_byte and child.end_byte > start_byte:
            if _has_parse_uncertainty(child, start_byte, end_byte):
                return True
    return False


def _find_function(root: Node, source_bytes: bytes, name: str) -> Node | None:
    if root.type == "function_definition":
        name_node = root.child_by_field_name("name")
        if name_node is not None:
            node_name = source_bytes[name_node.start_byte : name_node.end_byte].decode(
                "utf-8"
            )
            if node_name == name:
                return root
    for child in root.named_children:
        found = _find_function(child, source_bytes, name)
        if found is not None:
            return found
    return None
