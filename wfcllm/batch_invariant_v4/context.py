from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


CONTEXT_SCHEMA_VERSION = "wfcllm-batch-invariant-structural-context/v4"
_BUILTINS = frozenset(
    {
        "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float",
        "int", "len", "list", "map", "max", "min", "range", "reversed",
        "round", "set", "sorted", "str", "sum", "tuple", "zip",
    }
)


@dataclass(frozen=True)
class ContextConfig:
    schema_version: str = CONTEXT_SCHEMA_VERSION
    max_unit_bytes: int = 4096
    max_context_bytes: int = 8192
    global_ordinal_keying: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError("context schema version mismatch")
        if self.max_unit_bytes <= 0 or self.max_context_bytes <= 0:
            raise ValueError("context byte limits must be positive")
        if self.global_ordinal_keying is not False:
            raise ValueError("global ordinal keying must be false")


class _IdentifierNormalizer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.mapping: dict[str, str] = {}

    def _slot(self, name: str) -> str:
        if name in _BUILTINS:
            return f"builtin_{name}"
        if name not in self.mapping:
            self.mapping[name] = f"identifier_{len(self.mapping)}"
        return self.mapping[name]

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id=self._slot(node.id), ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        annotation = self.visit(node.annotation) if node.annotation is not None else None
        return ast.copy_location(
            ast.arg(
                arg=self._slot(node.arg),
                annotation=annotation,
                type_comment=node.type_comment,
            ),
            node,
        )

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.FunctionDef)
        normalized.name = self._slot(node.name)
        return normalized

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        normalized = self.generic_visit(node)
        assert isinstance(normalized, ast.AsyncFunctionDef)
        normalized.name = self._slot(node.name)
        return normalized


def _parse_fragment(source: str) -> ast.Module | None:
    if source == "<BOS>":
        return None
    return ast.parse(source)


def _dataflow(previous: ast.Module | None, current: ast.Module) -> tuple[str, ...]:
    stores: dict[str, str] = {}
    edges: list[str] = []
    for segment, tree in (("previous", previous), ("current", current)):
        if tree is None:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Name) or node.id.startswith("builtin_"):
                continue
            if isinstance(node.ctx, ast.Store):
                stores[node.id] = segment
            elif isinstance(node.ctx, ast.Load) and node.id in stores:
                edges.append(f"{node.id}:{stores[node.id]}->{segment}")
    return tuple(sorted(edges))


def _shell_source(statement: ast.stmt) -> str:
    shell = copy.deepcopy(statement)
    for field in ("body", "orelse", "finalbody"):
        value = getattr(shell, field, None)
        if isinstance(value, list):
            setattr(shell, field, [ast.Pass()] if value else [])
    handlers = getattr(shell, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            if isinstance(handler, ast.ExceptHandler):
                handler.body = [ast.Pass()]
    cases = getattr(shell, "cases", None)
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, ast.match_case):
                case.body = [ast.Pass()]
    ast.fix_missing_locations(shell)
    return ast.unparse(shell).strip().replace("\r\n", "\n")


def _child_groups(
    statement: ast.stmt,
) -> tuple[tuple[str, str, tuple[ast.stmt, ...]], ...]:
    groups: list[tuple[str, str, tuple[ast.stmt, ...]]] = []
    parent = type(statement).__name__
    for field in ("body", "orelse", "finalbody"):
        value = getattr(statement, field, None)
        if isinstance(value, list) and value and all(isinstance(x, ast.stmt) for x in value):
            groups.append((parent, field, tuple(value)))
    handlers = getattr(statement, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            if isinstance(handler, ast.ExceptHandler) and handler.body:
                groups.append(("ExceptHandler", "body", tuple(handler.body)))
    cases = getattr(statement, "cases", None)
    if isinstance(cases, list):
        for case in cases:
            if isinstance(case, ast.match_case) and case.body:
                groups.append(("match_case", "body", tuple(case.body)))
    return tuple(groups)


@dataclass(frozen=True)
class StructuralContext:
    unit_id: str
    role: str
    context_sha256: str
    representation_sha256: str
    representation_json: str
    representation_bytes: tuple[int, ...]


@dataclass(frozen=True)
class ContextExtraction:
    parse_ok: bool
    contexts: tuple[StructuralContext, ...]
    erasure_counts: dict[str, int]


def _context(role: str, previous: str, current: str) -> StructuralContext:
    previous_tree = _parse_fragment(previous)
    current_tree = _parse_fragment(current)
    assert current_tree is not None
    normalizer = _IdentifierNormalizer()
    normalized_previous = normalizer.visit(previous_tree) if previous_tree else None
    normalized_current = normalizer.visit(current_tree)
    assert isinstance(normalized_current, ast.Module)
    if normalized_previous is not None:
        assert isinstance(normalized_previous, ast.Module)
    payload = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "role": role,
        "previous_ast": (
            "<BOS>"
            if normalized_previous is None
            else ast.dump(normalized_previous, annotate_fields=True, include_attributes=False)
        ),
        "current_ast": ast.dump(
            normalized_current,
            annotate_fields=True,
            include_attributes=False,
        ),
        "bounded_dataflow": _dataflow(normalized_previous, normalized_current),
    }
    representation = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    representation_sha256 = hashlib.sha256(representation.encode("utf-8")).hexdigest()
    context_sha256 = representation_sha256
    unit_id = hashlib.sha256(
        (CONTEXT_SCHEMA_VERSION + "\0" + role + "\0" + representation_sha256).encode(
            "utf-8"
        )
    ).hexdigest()
    return StructuralContext(
        unit_id=unit_id,
        role=role,
        context_sha256=context_sha256,
        representation_sha256=representation_sha256,
        representation_json=representation,
        representation_bytes=tuple(bytes.fromhex(representation_sha256)),
    )


class StructuralContextExtractor:
    def __init__(self, config: ContextConfig) -> None:
        self.config = config

    def extract(self, final_code: str) -> ContextExtraction:
        try:
            tree = ast.parse(final_code)
        except (SyntaxError, ValueError, TypeError):
            return ContextExtraction(False, (), {"parse_failure": 1})
        candidates: list[StructuralContext] = []
        erasures: Counter[str] = Counter()

        def visit_group(
            statements: Sequence[ast.stmt],
            *,
            parent: str,
            field: str,
        ) -> None:
            previous = "<BOS>"
            for statement in statements:
                current = _shell_source(statement)
                oversize = False
                if len(current.encode("utf-8")) > self.config.max_unit_bytes:
                    erasures["unit_too_large"] += 1
                    oversize = True
                role = f"{type(statement).__name__}|{parent}|{field}"
                if (
                    len((role + "\0" + previous + "\0" + current).encode("utf-8"))
                    > self.config.max_context_bytes
                ):
                    erasures["context_too_large"] += 1
                    oversize = True
                if not oversize:
                    candidates.append(_context(role, previous, current))
                previous = current
                for child_parent, child_field, children in _child_groups(statement):
                    visit_group(children, parent=child_parent, field=child_field)

        for function in tree.body:
            if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit_group(
                    function.body,
                    parent=type(function).__name__,
                    field="body",
                )
        counts = Counter(context.unit_id for context in candidates)
        duplicates = {unit_id for unit_id, count in counts.items() if count > 1}
        contexts = tuple(
            context for context in candidates if context.unit_id not in duplicates
        )
        if duplicates:
            erasures["duplicate_unit_id"] += sum(
                context.unit_id in duplicates for context in candidates
            )
        return ContextExtraction(True, contexts, dict(sorted(erasures.items())))


class IncrementalStructuralTracker:
    def __init__(self, extractor: StructuralContextExtractor) -> None:
        self.extractor = extractor
        self.current = ContextExtraction(False, (), {})

    def observe(self, code: str) -> ContextExtraction:
        self.current = self.extractor.extract(code)
        return self.current
