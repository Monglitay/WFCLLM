"""Tests for simple expression/logic transformation rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.expression_logic import OperandSwap, ComparisonFlip, UnarySimplify


def test_operand_swap_add():
    rule = OperandSwap()
    result = rule.transform("a + b")
    assert result == "b + a"

def test_operand_swap_multiply():
    rule = OperandSwap()
    result = rule.transform("a * b")
    assert result == "b * a"

def test_operand_swap_no_match_subtract():
    rule = OperandSwap()
    assert not rule.can_apply("a - b")

def test_comparison_flip_lte():
    rule = ComparisonFlip()
    result = rule.transform("n <= right")
    assert result == "right >= n"

def test_comparison_flip_gt():
    rule = ComparisonFlip()
    result = rule.transform("a > b")
    assert result == "b < a"

def test_unary_simplify_double_not():
    rule = UnarySimplify()
    result = rule.transform("not not x")
    assert result == "x"
