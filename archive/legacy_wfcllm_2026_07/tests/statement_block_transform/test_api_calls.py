"""Tests for API call transformation rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.api_calls import (
    ExplicitDefaultPrint,
    ExplicitDefaultRange,
    ExplicitDefaultOpen,
    ExplicitDefaultSorted,
    ExplicitDefaultMinMax,
    ExplicitDefaultZip,
    ExplicitDefaultRandomSeed,
    ExplicitDefaultHtmlEscape,
    ExplicitDefaultRound,
    ExplicitDefaultJsonDump,
)


# --- Print ---
def test_print_detect():
    rule = ExplicitDefaultPrint()
    assert rule.can_apply("print(x)")
    assert rule.can_apply("print(x, y)")
    assert not rule.can_apply("x = 1")

def test_print_already_has_flush():
    rule = ExplicitDefaultPrint()
    assert not rule.can_apply("print(x, flush=True)")

def test_print_apply():
    rule = ExplicitDefaultPrint()
    result = rule.transform("print(x)")
    assert result == "print(x, flush=False)"

def test_print_multi_args():
    rule = ExplicitDefaultPrint()
    result = rule.transform("print(x, y)")
    assert result == "print(x, y, flush=False)"


# --- Range ---
def test_range_single_arg():
    rule = ExplicitDefaultRange()
    result = rule.transform("range(n)")
    assert result == "range(0, n)"

def test_range_two_args_no_match():
    rule = ExplicitDefaultRange()
    assert not rule.can_apply("range(0, n)")


# --- Open ---
def test_open_single_arg():
    rule = ExplicitDefaultOpen()
    result = rule.transform("open(f)")
    assert result == "open(f, closefd=True)"

def test_open_already_has_closefd():
    rule = ExplicitDefaultOpen()
    assert not rule.can_apply("open(f, closefd=False)")


# --- Sorted ---
def test_sorted_apply():
    rule = ExplicitDefaultSorted()
    result = rule.transform("sorted(x)")
    assert result == "sorted(x, reverse=False)"

def test_sorted_already_has_reverse():
    rule = ExplicitDefaultSorted()
    assert not rule.can_apply("sorted(x, reverse=True)")


# --- Min/Max ---
def test_min_apply():
    rule = ExplicitDefaultMinMax()
    result = rule.transform("min(x)")
    assert result == "min(x, key=None)"

def test_max_apply():
    rule = ExplicitDefaultMinMax()
    result = rule.transform("max(x)")
    assert result == "max(x, key=None)"


# --- Zip ---
def test_zip_apply():
    rule = ExplicitDefaultZip()
    result = rule.transform("zip(x, y)")
    assert result == "zip(x, y, strict=False)"


# --- Random.seed ---
def test_random_seed_apply():
    rule = ExplicitDefaultRandomSeed()
    result = rule.transform("random.seed(x)")
    assert result == "random.seed(x, version=2)"


# --- Html.escape ---
def test_html_escape_apply():
    rule = ExplicitDefaultHtmlEscape()
    result = rule.transform("html.escape(x)")
    assert result == "html.escape(x, quote=True)"


# --- Round ---
def test_round_apply():
    rule = ExplicitDefaultRound()
    result = rule.transform("round(x)")
    assert result == "round(x, ndigits=None)"


# --- Json.dump ---
def test_json_dump_apply():
    rule = ExplicitDefaultJsonDump()
    result = rule.transform("json.dump(x)")
    assert result == "json.dump(x, indent=None)"
