"""Negative transform rules registry."""

from rules.base import Rule
from rules.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from rules.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)


def get_all_negative_rules() -> list[Rule]:
    return [
        # API (7)
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        # Control flow (5)
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
    ]
