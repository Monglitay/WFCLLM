"""Tests for wfcllm.common.transform.engine."""

from wfcllm.common.transform.base import Match, Rule, parse_code
from wfcllm.common.transform.engine import TransformEngine


class _AddOneRule(Rule):
    """Replace '1' with '2' in source."""
    name = "add_one"
    category = "test"
    description = "test"

    def detect(self, source, tree):
        matches = []
        for i, ch in enumerate(source):
            if ch == "1":
                matches.append(Match("integer", i, i + 1, "1", "2"))
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda x: x.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class _AddExcl(Rule):
    """Append '!' to source by replacing last char."""
    name = "add_excl"
    category = "test"
    description = "test"

    def detect(self, source, tree):
        if source.endswith("\n"):
            return []
        return [Match("module", len(source), len(source), "", "!")]

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda x: x.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class TestTransformEngine:
    def test_get_applicable_rules(self):
        engine = TransformEngine(rules=[_AddOneRule(), _AddExcl()])
        applicable = engine.get_applicable_rules("x = 1")
        assert len(applicable) == 2

    def test_no_applicable_rules(self):
        engine = TransformEngine(rules=[_AddOneRule()])
        applicable = engine.get_applicable_rules("x = 0")
        assert len(applicable) == 0

    def test_generate_variants(self):
        engine = TransformEngine(
            rules=[_AddOneRule()],
            max_perm_len=1,
            max_variants=10,
        )
        variants = engine.generate_variants("x = 1")
        assert len(variants) >= 1
        assert variants[0]["transformed_source"] == "x = 2"
        assert variants[0]["rules_applied"] == ["add_one"]
        assert variants[0]["sample_type"] == "positive"

    def test_negative_mode(self):
        engine = TransformEngine(
            rules=[_AddOneRule()],
            max_perm_len=1,
            max_variants=10,
            mode="negative",
        )
        variants = engine.generate_variants("x = 1")
        assert variants[0]["sample_type"] == "negative"

    def test_max_variants_limit(self):
        engine = TransformEngine(
            rules=[_AddOneRule(), _AddExcl()],
            max_perm_len=5,
            max_variants=3,
        )
        variants = engine.generate_variants("x = 1")
        assert len(variants) <= 3

    def test_empty_source(self):
        engine = TransformEngine(rules=[_AddOneRule()])
        variants = engine.generate_variants("")
        assert variants == []
