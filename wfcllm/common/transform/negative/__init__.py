"""Negative (semantic-breaking) transformation rules registry."""

from wfcllm.common.transform.base import Rule
from wfcllm.common.transform.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from wfcllm.common.transform.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from wfcllm.common.transform.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)
from wfcllm.common.transform.negative.identifier import ScopeVarCorrupt
from wfcllm.common.transform.negative.data_structure import SliceStepFlip, DictViewSwap
from wfcllm.common.transform.negative.exception import ExceptionSwallow
from wfcllm.common.transform.negative.system import SysExitFlip


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
