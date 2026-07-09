"""Tests for advanced API call rules (LNA, TPF)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.api_calls import LibraryAliasReplace, ThirdPartyFuncReplace


def test_np_to_numpy():
    rule = LibraryAliasReplace()
    result = rule.transform("np.sum(x)")
    assert result == "numpy.sum(x)"

def test_numpy_to_np():
    rule = LibraryAliasReplace()
    result = rule.transform("numpy.sum(x)")
    assert result == "np.sum(x)"

def test_builtin_to_numpy():
    rule = ThirdPartyFuncReplace()
    result = rule.transform("max(x)")
    assert result == "np.max(x)" or result == "numpy.max(x)"
