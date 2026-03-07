"""Tests for TransformEngine."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from engine import TransformEngine
from rules.base import Match, Rule, parse_code


class AddCommentRule(Rule):
    name = "add_comment"
    category = "test"
    description = "Add comment to pass"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "pass_statement":
                matches.append(Match("pass_statement", node.start_byte, node.end_byte, "pass", "pass  # noop"))
            for c in node.children:
                walk(c)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            source = source[:m.start_byte] + m.replacement_text + source[m.end_byte:]
        return source


class ListInitRule(Rule):
    name = "list_init"
    category = "test"
    description = "[] -> list()"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "list" and len(node.children) == 2:  # empty list []
                matches.append(Match("list", node.start_byte, node.end_byte, "[]", "list()"))
            for c in node.children:
                walk(c)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            source = source[:m.start_byte] + m.replacement_text + source[m.end_byte:]
        return source


def test_get_applicable_rules_none():
    engine = TransformEngine(rules=[AddCommentRule()], max_perm_len=5, max_variants=100)
    applicable = engine.get_applicable_rules("x = 1")
    assert applicable == []


def test_get_applicable_rules_one():
    engine = TransformEngine(rules=[AddCommentRule()], max_perm_len=5, max_variants=100)
    applicable = engine.get_applicable_rules("pass")
    assert len(applicable) == 1
    assert applicable[0].name == "add_comment"


def test_generate_variants_single_rule():
    engine = TransformEngine(rules=[AddCommentRule()], max_perm_len=5, max_variants=100)
    variants = engine.generate_variants("pass")
    assert len(variants) == 1
    assert variants[0]["rules_applied"] == ["add_comment"]
    assert variants[0]["transformed_source"] == "pass  # noop"


def test_generate_variants_two_rules():
    engine = TransformEngine(
        rules=[AddCommentRule(), ListInitRule()],
        max_perm_len=5,
        max_variants=100,
    )
    source = "x = []\npass"
    variants = engine.generate_variants(source)
    # 2 single-rule + 2 two-rule permutations = 4
    assert len(variants) == 4
    rule_sets = [tuple(v["rules_applied"]) for v in variants]
    assert ("add_comment",) in rule_sets
    assert ("list_init",) in rule_sets
    assert ("add_comment", "list_init") in rule_sets
    assert ("list_init", "add_comment") in rule_sets


def test_max_variants_truncation():
    engine = TransformEngine(
        rules=[AddCommentRule(), ListInitRule()],
        max_perm_len=5,
        max_variants=2,  # truncate to 2
    )
    source = "x = []\npass"
    variants = engine.generate_variants(source)
    assert len(variants) == 2  # truncated


def test_max_perm_length():
    engine = TransformEngine(
        rules=[AddCommentRule(), ListInitRule()],
        max_perm_len=1,  # only single-rule permutations
        max_variants=100,
    )
    source = "x = []\npass"
    variants = engine.generate_variants(source)
    assert len(variants) == 2  # only single-rule
    for v in variants:
        assert len(v["rules_applied"]) == 1


def test_no_applicable_rules():
    engine = TransformEngine(rules=[AddCommentRule()], max_perm_len=5, max_variants=100)
    variants = engine.generate_variants("x = 1")
    assert variants == []
