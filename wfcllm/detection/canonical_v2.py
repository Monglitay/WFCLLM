from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import dataclass

from wfcllm.semantic.rules import _normalize_candidate_text

CANONICAL_UNIT_SCHEMA_VERSION = "wfcllm-canonical-unit/v2"

_COMPOUND_STATEMENT_TYPES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.Try,
    ast.With,
    ast.AsyncWith,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)
if hasattr(ast, "Match"):
    _COMPOUND_STATEMENT_TYPES = (*_COMPOUND_STATEMENT_TYPES, ast.Match)


@dataclass(frozen=True)
class CanonicalUnit:
    schema_version: str
    unit_id: str
    normalized_text: str
    normalized_text_hash: str
    node_type: str
    parent_node_type: str
    ordinal: int
    start_line: int
    end_line: int


@dataclass(frozen=True)
class CanonicalExtraction:
    units: tuple[CanonicalUnit, ...]
    duplicate_count: int
    parse_valid: bool
    target_function_name: str | None


@dataclass(frozen=True)
class _RawUnit:
    normalized_text: str
    normalized_text_hash: str
    node_type: str
    parent_node_type: str
    start_line: int
    end_line: int
    col_offset: int
    end_col_offset: int


def extract_canonical_units(final_code: str) -> CanonicalExtraction:
    if not isinstance(final_code, str):
        raise ValueError("final_code must be a string")
    try:
        tree = ast.parse(final_code)
    except (SyntaxError, ValueError, TypeError):
        return CanonicalExtraction(
            units=(),
            duplicate_count=0,
            parse_valid=False,
            target_function_name=None,
        )

    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return CanonicalExtraction(
            units=(),
            duplicate_count=0,
            parse_valid=True,
            target_function_name=None,
        )

    target = functions[-1]
    raw_units = sorted(
        _owned_simple_statements(final_code, target.body, "function_definition"),
        key=lambda item: (
            item.start_line,
            item.col_offset,
            item.end_line,
            item.end_col_offset,
        ),
    )

    seen: set[tuple[str, str, str]] = set()
    unique_units: list[_RawUnit] = []
    duplicate_count = 0
    for raw in raw_units:
        dedup_key = (
            raw.normalized_text,
            raw.node_type,
            raw.parent_node_type,
        )
        if dedup_key in seen:
            duplicate_count += 1
            continue
        seen.add(dedup_key)
        unique_units.append(raw)

    units = tuple(
        _build_unit(raw, ordinal)
        for ordinal, raw in enumerate(unique_units)
    )
    return CanonicalExtraction(
        units=units,
        duplicate_count=duplicate_count,
        parse_valid=True,
        target_function_name=target.name,
    )


def _owned_simple_statements(
    source: str,
    statements: list[ast.stmt],
    parent_node_type: str,
) -> list[_RawUnit]:
    units: list[_RawUnit] = []
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(statement, _COMPOUND_STATEMENT_TYPES):
            child_parent = _node_type(statement)
            for child_body in _child_statement_bodies(statement):
                units.extend(
                    _owned_simple_statements(source, child_body, child_parent)
                )
            continue

        segment = ast.get_source_segment(source, statement)
        if segment is None:
            continue
        normalized_text = _normalize_candidate_text(segment)
        if not normalized_text:
            continue
        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        units.append(
            _RawUnit(
                normalized_text=normalized_text,
                normalized_text_hash=text_hash,
                node_type=_node_type(statement),
                parent_node_type=parent_node_type,
                start_line=int(getattr(statement, "lineno", 0)),
                end_line=int(getattr(statement, "end_lineno", 0)),
                col_offset=int(getattr(statement, "col_offset", 0)),
                end_col_offset=int(getattr(statement, "end_col_offset", 0)),
            )
        )
    return units


def _child_statement_bodies(statement: ast.stmt) -> list[list[ast.stmt]]:
    bodies: list[list[ast.stmt]] = []
    for field_name in ("body", "orelse", "finalbody"):
        value = getattr(statement, field_name, None)
        if isinstance(value, list):
            bodies.append([item for item in value if isinstance(item, ast.stmt)])
    handlers = getattr(statement, "handlers", None)
    if isinstance(handlers, list):
        for handler in handlers:
            body = getattr(handler, "body", None)
            if isinstance(body, list):
                bodies.append([item for item in body if isinstance(item, ast.stmt)])
    cases = getattr(statement, "cases", None)
    if isinstance(cases, list):
        for case in cases:
            body = getattr(case, "body", None)
            if isinstance(body, list):
                bodies.append([item for item in body if isinstance(item, ast.stmt)])
    return bodies


def _build_unit(raw: _RawUnit, ordinal: int) -> CanonicalUnit:
    payload = {
        "schema_version": CANONICAL_UNIT_SCHEMA_VERSION,
        "normalized_text_hash": raw.normalized_text_hash,
        "node_type": raw.node_type,
        "parent_node_type": raw.parent_node_type,
        "ordinal": ordinal,
    }
    unit_id = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return CanonicalUnit(
        schema_version=CANONICAL_UNIT_SCHEMA_VERSION,
        unit_id=unit_id,
        normalized_text=raw.normalized_text,
        normalized_text_hash=raw.normalized_text_hash,
        node_type=raw.node_type,
        parent_node_type=raw.parent_node_type,
        ordinal=ordinal,
        start_line=raw.start_line,
        end_line=raw.end_line,
    )


def _node_type(node: ast.AST) -> str:
    name = type(node).__name__
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
