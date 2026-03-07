"""Tests for formatting transformation rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.formatting import FixSpacing, FixCommentSymbols


def test_fix_spacing_add():
    rule = FixSpacing()
    result = rule.transform("result=a+b")
    assert result == "result = a + b"

def test_fix_comment_symbols():
    rule = FixCommentSymbols()
    result = rule.transform("##### Hello")
    assert result == "# Hello"
