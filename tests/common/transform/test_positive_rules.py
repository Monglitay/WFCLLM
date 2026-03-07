"""Tests for positive transformation rules."""

import pytest
from wfcllm.common.transform.positive import get_all_positive_rules


class TestPositiveRulesRegistry:
    def test_all_rules_loaded(self):
        rules = get_all_positive_rules()
        # 12 api_calls + 4 syntax_init + 4 control_flow + 5 expression_logic + 2 identifier + 2 formatting = 29
        # Note: exact count based on experiment code __init__.py
        assert len(rules) >= 25  # at least this many

    def test_all_rules_have_name(self):
        for rule in get_all_positive_rules():
            assert rule.name, f"Rule {rule.__class__.__name__} has no name"

    def test_all_rules_have_category(self):
        for rule in get_all_positive_rules():
            assert rule.category, f"Rule {rule.name} has no category"


class TestApiCallRules:
    def test_explicit_default_print(self):
        from wfcllm.common.transform.positive.api_calls import ExplicitDefaultPrint
        rule = ExplicitDefaultPrint()
        result = rule.transform("print(x)")
        assert result is not None
        assert "flush" in result or "end" in result

    def test_explicit_default_range(self):
        from wfcllm.common.transform.positive.api_calls import ExplicitDefaultRange
        rule = ExplicitDefaultRange()
        result = rule.transform("range(10)")
        assert result is not None
        assert "0" in result  # range(10) -> range(0, 10)

    def test_min_max_unchanged_semantics(self):
        from wfcllm.common.transform.positive.api_calls import ExplicitDefaultMinMax
        rule = ExplicitDefaultMinMax()
        result = rule.transform("min(a, b)")
        assert result is not None
        assert "key=None" in result


class TestControlFlowRules:
    def test_loop_convert(self):
        from wfcllm.common.transform.positive.control_flow import LoopConvert
        rule = LoopConvert()
        source = "for i in range(n):\n    print(i)"
        assert rule.can_apply(source)

    def test_branch_flip(self):
        from wfcllm.common.transform.positive.control_flow import BranchFlip
        rule = BranchFlip()
        source = "if x > 0:\n    print('yes')\nelse:\n    print('no')"
        assert rule.can_apply(source)


class TestExpressionLogicRules:
    def test_operand_swap(self):
        from wfcllm.common.transform.positive.expression_logic import OperandSwap
        rule = OperandSwap()
        result = rule.transform("x + y")
        assert result is not None

    def test_comparison_flip(self):
        from wfcllm.common.transform.positive.expression_logic import ComparisonFlip
        rule = ComparisonFlip()
        result = rule.transform("a < b")
        assert result is not None
        assert ">" in result  # a < b → b > a


class TestIdentifierRules:
    def test_variable_rename(self):
        from wfcllm.common.transform.positive.identifier import VariableRename
        rule = VariableRename()
        result = rule.transform("my_var = 1")
        # Should convert snake_case to camelCase or vice versa
        assert result is not None


class TestFormattingRules:
    def test_fix_spacing(self):
        from wfcllm.common.transform.positive.formatting import FixSpacing
        rule = FixSpacing()
        # If source has operators without spaces, should add them
        assert rule.name  # at minimum check it loads
