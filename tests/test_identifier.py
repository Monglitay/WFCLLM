"""Tests for identifier transformation rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.identifier import VariableRename, NameObfuscation


def test_snake_to_camel():
    rule = VariableRename()
    result = rule.transform("total_sum = a + b\nprint(total_sum)")
    assert "totalSum" in result

def test_camel_to_snake():
    rule = VariableRename()
    result = rule.transform("totalSum = a + b\nprint(totalSum)")
    assert "total_sum" in result

def test_name_obfuscation():
    rule = NameObfuscation()
    result = rule.transform("def calculate_total(items):\n    return sum(items)")
    # Function name should be changed but code should remain valid
    assert "calculate_total" not in result
    assert "return sum(items)" in result
