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


def test_old_transform_base_symbol_identity():
    import wfcllm.common.transform.base as old
    import wfcllm.lang.python.transform.base as new
    assert old.Rule is new.Rule
    assert old.Match is new.Match
    assert old.parse_code is new.parse_code


def test_old_transform_engine_symbol_identity():
    import wfcllm.common.transform.engine as old
    import wfcllm.lang.python.transform.engine as new
    assert old.TransformEngine is new.TransformEngine


def test_old_positive_rules_symbol_identity():
    """Every public rule class re-exported from the old positive aggregator must be identity-equal to the new one."""
    import wfcllm.common.transform.positive as old
    import wfcllm.lang.python.transform.positive as new
    assert old.get_all_positive_rules.__code__ is new.get_all_positive_rules.__code__ \
        or old.get_all_positive_rules is new.get_all_positive_rules
    # Spot-check a few representative classes — full equivalence is enforced by
    # tests/common/transform/test_positive_rules.py.
    for sym in ("VariableRename", "FixSpacing", "ListInit", "BranchFlip", "DeMorgan"):
        assert getattr(old, sym) is getattr(new, sym), sym


def test_old_positive_rules_count_equals_new():
    import wfcllm.common.transform.positive as old
    import wfcllm.lang.python.transform.positive as new
    assert len(old.get_all_positive_rules()) == len(new.get_all_positive_rules())


def test_old_negative_rules_symbol_identity():
    import wfcllm.common.transform.negative as old
    import wfcllm.lang.python.transform.negative as new
    for sym in ("MinMaxFlip", "OffByOne", "EqNeqFlip", "SliceStepFlip", "ExceptionSwallow", "SysExitFlip"):
        assert getattr(old, sym) is getattr(new, sym), sym


def test_old_negative_rules_count_equals_new():
    import wfcllm.common.transform.negative as old
    import wfcllm.lang.python.transform.negative as new
    assert len(old.get_all_negative_rules()) == len(new.get_all_negative_rules())


def test_old_transform_package_symbol_identity():
    import wfcllm.common.transform as old
    import wfcllm.lang.python.transform as new
    assert old.Rule is new.Rule
    assert old.Match is new.Match
    assert old.parse_code is new.parse_code
    assert old.TransformEngine is new.TransformEngine


def test_old_dataset_loader_symbol_identity():
    import wfcllm.common.dataset_loader as old
    import wfcllm.datasets.loaders.local as new
    assert old.load_prompts is new.load_prompts
    assert old.load_reference_solutions is new.load_reference_solutions
    assert old.SUPPORTED_DATASETS == new.SUPPORTED_DATASETS
