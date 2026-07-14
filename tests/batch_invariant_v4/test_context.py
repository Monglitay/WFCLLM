from __future__ import annotations

import ast

from wfcllm.batch_invariant_v4.context import (
    ContextConfig,
    IncrementalStructuralTracker,
    StructuralContextExtractor,
)


def _extract(code: str):
    return StructuralContextExtractor(ContextConfig()).extract(code)


def test_format_comments_and_scoped_identifier_rename_are_canonical() -> None:
    left = _extract(
        "def f(value):\n    result = value + 1\n    return result\n"
    )
    right = _extract(
        "def renamed(source): # comment\n  output=source+1\n  return output\n"
    )

    assert left.parse_ok and right.parse_ok
    assert left.contexts == right.contexts


def test_semicolon_and_ast_roundtrip_normalize() -> None:
    semicolon = _extract("def f(x):\n    y=x+1; return y\n")
    roundtrip = _extract(ast.unparse(ast.parse("def f(x):\n y=x+1\n return y\n")))

    assert semicolon.contexts == roundtrip.contexts


def test_nested_compound_units_are_shells_without_child_duplication() -> None:
    extraction = _extract(
        "def f(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        if value > 0:\n"
        "            total += value\n"
        "    return total\n"
    )

    roles = {context.role: context for context in extraction.contexts}
    assert len(extraction.contexts) == 5
    assert "For|FunctionDef|body" in roles
    assert "If|For|body" in roles
    assert "AugAssign|If|body" in roles
    assert "AugAssign" not in roles["For|FunctionDef|body"].representation_json
    assert "AugAssign" not in roles["If|For|body"].representation_json


def test_parse_failure_oversize_and_duplicate_are_explicit_erasures() -> None:
    malformed = _extract("def broken(:\n")
    assert malformed.parse_ok is False
    assert malformed.erasure_counts == {"parse_failure": 1}

    oversize = StructuralContextExtractor(
        ContextConfig(max_unit_bytes=8, max_context_bytes=16)
    ).extract("def f():\n    return 'too long'\n")
    assert oversize.contexts == ()
    assert oversize.erasure_counts["unit_too_large"] == 1

    duplicate = _extract(
        "def first():\n    return 1\n\n"
        "def second():\n    return 1\n"
    )
    assert duplicate.contexts == ()
    assert duplicate.erasure_counts == {"duplicate_unit_id": 2}


def test_insertion_deletion_and_reorder_change_evidence_without_ordinals() -> None:
    base = _extract("def f(source):\n    value = source + 1\n    result = value * 2\n")
    inserted = _extract(
        "def f(source):\n    value = source + 1\n    middle = value - 3\n    result = value * 2\n"
    )
    deleted = _extract("def f(source):\n    result = source * 2\n")
    reordered = _extract(
        "def f(source):\n    result = source * 2\n    value = source + 1\n"
    )

    assert base.contexts != inserted.contexts
    assert base.contexts != deleted.contexts
    assert base.contexts != reordered.contexts
    assert all("ordinal" not in context.representation_json for context in base.contexts)


def test_tracker_rollback_replaces_stale_parse_state() -> None:
    tracker = IncrementalStructuralTracker(StructuralContextExtractor(ContextConfig()))
    complete = "def f():\n    value = 1\n    return value\n"
    broken = "def f(:\n"
    changed = "def f():\n    value = 2\n    return value\n"

    assert tracker.observe(complete).parse_ok is True
    assert tracker.observe(broken).contexts == ()
    final = tracker.observe(changed)
    assert final.parse_ok is True
    assert tracker.current == final
    assert final != _extract(complete)
