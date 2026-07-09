"""Tests for syntax/init transformation rules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.syntax_init import ListInit, DictInit, TypeCheck, StringFormat


def test_list_init_empty():
    rule = ListInit()
    assert rule.transform("x = []") == "x = list()"

def test_list_init_nonempty_no_match():
    rule = ListInit()
    assert not rule.can_apply("x = [1, 2]")

def test_list_init_reverse():
    rule = ListInit()
    assert rule.transform("x = list()") == "x = []"

def test_dict_init_empty():
    rule = DictInit()
    assert rule.transform("x = {}") == "x = dict()"

def test_dict_init_reverse():
    rule = DictInit()
    assert rule.transform("x = dict()") == "x = {}"

def test_type_check_isinstance_to_type():
    rule = TypeCheck()
    result = rule.transform("isinstance(x, int)")
    assert result == "type(x) == int"

def test_type_check_type_to_isinstance():
    rule = TypeCheck()
    result = rule.transform("type(x) == int")
    assert result == "isinstance(x, int)"

def test_string_format_percent_to_format():
    rule = StringFormat()
    result = rule.transform("'%s' % x")
    assert result == "'{}'.format(x)"
