"""Tests for Rule base class."""

import sys
from pathlib import Path

# Add experiment dir to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from rules.base import Match, Rule, parse_code


class DummyRule(Rule):
    """A test rule that replaces 'pass' with 'pass  # noop'."""
    name = "dummy_pass"
    category = "test"
    description = "Add comment to pass statements"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "pass_statement":
                matches.append(Match(
                    node_type="pass_statement",
                    start_byte=node.start_byte,
                    end_byte=node.end_byte,
                    original_text=source[node.start_byte:node.end_byte],
                    replacement_text="pass  # noop",
                ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


def test_parse_code():
    tree = parse_code("x = 1")
    assert tree.root_node.type == "module"


def test_dummy_rule_detect():
    rule = DummyRule()
    tree = parse_code("pass")
    matches = rule.detect("pass", tree)
    assert len(matches) == 1
    assert matches[0].node_type == "pass_statement"


def test_dummy_rule_apply():
    rule = DummyRule()
    result = rule.transform("pass")
    assert result == "pass  # noop"


def test_dummy_rule_no_match():
    rule = DummyRule()
    assert rule.can_apply("x = 1") is False
    assert rule.transform("x = 1") is None
