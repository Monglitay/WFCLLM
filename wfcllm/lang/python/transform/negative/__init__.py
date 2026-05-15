"""Negative (semantic-breaking) transformation rules registry."""

from wfcllm.lang.python.transform.base import Rule
from wfcllm.lang.python.transform.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from wfcllm.lang.python.transform.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from wfcllm.lang.python.transform.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)
from wfcllm.lang.python.transform.negative.identifier import ScopeVarCorrupt
from wfcllm.lang.python.transform.negative.data_structure import SliceStepFlip, DictViewSwap
from wfcllm.lang.python.transform.negative.exception import ExceptionSwallow
from wfcllm.lang.python.transform.negative.system import SysExitFlip


def get_all_negative_rules() -> list[Rule]:
    return [
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
        EqNeqFlip(), ArithmeticOpReplace(), AndOrSwap(), BoundsNarrow(), AugAssignCorrupt(), ShiftFlip(),
        ScopeVarCorrupt(),
        SliceStepFlip(), DictViewSwap(),
        ExceptionSwallow(),
        SysExitFlip(),
    ]
