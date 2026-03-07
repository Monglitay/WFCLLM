"""Tests for negative transformation rules."""

import pytest
from wfcllm.common.transform.negative import get_all_negative_rules


class TestNegativeRulesRegistry:
    def test_all_rules_loaded(self):
        rules = get_all_negative_rules()
        # 7 api + 5 control_flow + 6 expression + 1 identifier + 2 data_structure + 1 exception + 1 system = 23
        assert len(rules) == 23

    def test_all_rules_have_name(self):
        for rule in get_all_negative_rules():
            assert rule.name, f"Rule {rule.__class__.__name__} has no name"
            assert rule.name.startswith("neg_"), f"Negative rule {rule.name} should start with 'neg_'"


class TestNegativeApiCalls:
    def test_min_max_flip(self):
        from wfcllm.common.transform.negative.api_calls import MinMaxFlip
        rule = MinMaxFlip()
        result = rule.transform("x = min(a, b)")
        assert result is not None
        assert "max" in result

    def test_any_all_flip(self):
        from wfcllm.common.transform.negative.api_calls import AnyAllFlip
        rule = AnyAllFlip()
        result = rule.transform("if any(lst):\n    pass")
        assert result is not None
        assert "all" in result

    def test_ceil_floor_flip(self):
        from wfcllm.common.transform.negative.api_calls import CeilFloorFlip
        rule = CeilFloorFlip()
        result = rule.transform("x = math.ceil(y)")
        assert result is not None
        assert "floor" in result


class TestNegativeControlFlow:
    def test_off_by_one(self):
        from wfcllm.common.transform.negative.control_flow import OffByOne
        rule = OffByOne()
        result = rule.transform("for i in range(n):\n    pass")
        assert result is not None
        assert "n - 1" in result

    def test_break_continue_swap(self):
        from wfcllm.common.transform.negative.control_flow import BreakContinueSwap
        rule = BreakContinueSwap()
        result = rule.transform("for i in range(10):\n    break")
        assert result is not None
        assert "continue" in result


class TestNegativeExpressionLogic:
    def test_eq_neq_flip(self):
        from wfcllm.common.transform.negative.expression_logic import EqNeqFlip
        rule = EqNeqFlip()
        result = rule.transform("x == y")
        assert result is not None
        assert "!=" in result

    def test_and_or_swap(self):
        from wfcllm.common.transform.negative.expression_logic import AndOrSwap
        rule = AndOrSwap()
        result = rule.transform("a and b")
        assert result is not None
        assert "or" in result
