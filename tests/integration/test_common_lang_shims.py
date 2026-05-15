"""Tests that wfcllm.common.* deprecated import paths still work via shims (Phase 6)."""
from __future__ import annotations

import warnings


def test_old_ast_parser_emits_deprecation_warning():
    import importlib
    import sys

    sys.modules.pop("wfcllm.common.ast_parser", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("wfcllm.common.ast_parser")
    msgs = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert msgs, f"expected DeprecationWarning, got: {[w.category for w in caught]}"


def test_old_ast_parser_symbol_identity():
    """Old path must re-export the SAME object as the new path (preserves singletons)."""
    import wfcllm.common.ast_parser as old
    import wfcllm.lang.python.parser as new
    assert old.PythonParser is new.PythonParser
    assert old.extract_statement_blocks is new.extract_statement_blocks
    assert old.StatementBlock is new.StatementBlock


def test_old_ast_parser_constants_round_trip():
    import wfcllm.common.ast_parser as old
    import wfcllm.lang.python.parser as new
    assert old.SIMPLE_STATEMENT_TYPES == new.SIMPLE_STATEMENT_TYPES
    assert old.COMPOUND_STATEMENT_TYPES == new.COMPOUND_STATEMENT_TYPES
    assert old.STATEMENT_TYPES == new.STATEMENT_TYPES
