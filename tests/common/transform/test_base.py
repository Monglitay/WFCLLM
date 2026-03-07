"""Tests for wfcllm.common.transform.base."""

from wfcllm.common.transform.base import Match, Rule, parse_code


class TestParseCode:
    def test_returns_tree(self):
        tree = parse_code("x = 1")
        assert tree.root_node.type == "module"

    def test_empty_source(self):
        tree = parse_code("")
        assert tree.root_node.type == "module"


class TestMatch:
    def test_fields(self):
        m = Match("call", 0, 5, "print", "print")
        assert m.node_type == "call"
        assert m.start_byte == 0
        assert m.end_byte == 5


class _DummyRule(Rule):
    """A test rule that replaces 'hello' with 'world'."""
    name = "dummy"
    category = "test"
    description = "test rule"

    def detect(self, source, tree):
        matches = []
        idx = source.find("hello")
        if idx >= 0:
            matches.append(Match("identifier", idx, idx + 5, "hello", "world"))
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda x: x.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class TestRule:
    def test_can_apply_true(self):
        r = _DummyRule()
        assert r.can_apply('x = "hello"') is True

    def test_can_apply_false(self):
        r = _DummyRule()
        assert r.can_apply("x = 1") is False

    def test_transform(self):
        r = _DummyRule()
        result = r.transform('x = "hello"')
        assert result == 'x = "world"'

    def test_transform_not_applicable(self):
        r = _DummyRule()
        assert r.transform("x = 1") is None
