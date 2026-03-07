"""Positive (semantic-equivalent) transformation rules registry."""

from wfcllm.common.transform.base import Rule
from wfcllm.common.transform.positive.api_calls import (
    ExplicitDefaultPrint, ExplicitDefaultRange, ExplicitDefaultOpen,
    ExplicitDefaultSorted, ExplicitDefaultMinMax, ExplicitDefaultZip,
    ExplicitDefaultRandomSeed, ExplicitDefaultHtmlEscape,
    ExplicitDefaultRound, ExplicitDefaultJsonDump,
    LibraryAliasReplace, ThirdPartyFuncReplace,
)
from wfcllm.common.transform.positive.syntax_init import ListInit, DictInit, TypeCheck, StringFormat
from wfcllm.common.transform.positive.control_flow import LoopConvert, IterationConvert, ComprehensionConvert, BranchFlip
from wfcllm.common.transform.positive.expression_logic import (
    OperandSwap, ComparisonFlip, UnarySimplify, DeMorgan, ArithmeticAssociativity,
)
from wfcllm.common.transform.positive.identifier import VariableRename, NameObfuscation
from wfcllm.common.transform.positive.formatting import FixSpacing, FixCommentSymbols

_ALL_POSITIVE_RULES: list[Rule] = [
    ExplicitDefaultPrint(), ExplicitDefaultRange(), ExplicitDefaultOpen(),
    ExplicitDefaultSorted(), ExplicitDefaultMinMax(), ExplicitDefaultZip(),
    ExplicitDefaultRandomSeed(), ExplicitDefaultHtmlEscape(),
    ExplicitDefaultRound(), ExplicitDefaultJsonDump(),
    LibraryAliasReplace(), ThirdPartyFuncReplace(),
    ListInit(), DictInit(), TypeCheck(), StringFormat(),
    LoopConvert(), IterationConvert(), ComprehensionConvert(), BranchFlip(),
    OperandSwap(), ComparisonFlip(), UnarySimplify(), DeMorgan(), ArithmeticAssociativity(),
    VariableRename(), NameObfuscation(),
    FixSpacing(), FixCommentSymbols(),
]


def get_all_positive_rules() -> list[Rule]:
    return list(_ALL_POSITIVE_RULES)
