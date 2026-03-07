"""Negative transform rules registry."""

from rules.base import Rule
from rules.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from rules.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from rules.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)
from rules.negative.identifier import ScopeVarCorrupt
from rules.negative.data_structure import SliceStepFlip, DictViewSwap
from rules.negative.exception import ExceptionSwallow
from rules.negative.system import SysExitFlip


def get_all_negative_rules() -> list[Rule]:
    return [
        # API (7)
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        # Control flow (5)
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
        # Expression logic (6)
        EqNeqFlip(), ArithmeticOpReplace(), AndOrSwap(), BoundsNarrow(), AugAssignCorrupt(), ShiftFlip(),
        # Identifier (1)
        ScopeVarCorrupt(),
        # Data structure (2)
        SliceStepFlip(), DictViewSwap(),
        # Exception (1)
        ExceptionSwallow(),
        # System (1)
        SysExitFlip(),
    ]
