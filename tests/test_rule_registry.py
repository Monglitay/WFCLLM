"""Tests for rule registry."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules import get_all_rules


def test_all_rules_registered():
    rules = get_all_rules()
    assert len(rules) >= 29  # all implemented rules
    names = [r.name for r in rules]
    assert "explicit_default_print" in names
    assert "list_init" in names
    assert "operand_swap" in names
    assert "branch_flip" in names
    assert "variable_rename" in names


def test_no_duplicate_names():
    rules = get_all_rules()
    names = [r.name for r in rules]
    assert len(names) == len(set(names))
