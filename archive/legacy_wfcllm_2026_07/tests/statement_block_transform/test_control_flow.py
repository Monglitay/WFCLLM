"""Tests for control flow transformation rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.control_flow import LoopConvert, IterationConvert, ComprehensionConvert, BranchFlip


def test_for_to_while():
    rule = LoopConvert()
    source = "for i in range(n):\n    print(i)"
    result = rule.transform(source)
    assert "while" in result
    assert "i = 0" in result

def test_index_iteration():
    rule = IterationConvert()
    source = "for x in lst:\n    print(x)"
    result = rule.transform(source)
    assert "range(len(lst))" in result

def test_comprehension_to_map():
    rule = ComprehensionConvert()
    source = "[f(x) for x in lst]"
    result = rule.transform(source)
    assert "map(" in result

def test_branch_flip():
    rule = BranchFlip()
    source = "if condition:\n    x = 1\nelse:\n    x = 2"
    result = rule.transform(source)
    assert "not condition" in result
    assert result.index("x = 2") < result.index("x = 1")
