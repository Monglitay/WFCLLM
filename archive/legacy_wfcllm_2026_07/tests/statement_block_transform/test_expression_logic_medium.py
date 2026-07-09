"""Tests for medium expression/logic rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.expression_logic import DeMorgan, ArithmeticAssociativity


def test_demorgan_and():
    rule = DeMorgan()
    result = rule.transform("if x and y:\n    pass")
    assert "not (not x or not y)" in result

def test_demorgan_or():
    rule = DeMorgan()
    result = rule.transform("if x or y:\n    pass")
    assert "not (not x and not y)" in result

def test_arithmetic_distribute():
    rule = ArithmeticAssociativity()
    result = rule.transform("x = a * (b + c)")
    assert result == "x = a * b + a * c"
