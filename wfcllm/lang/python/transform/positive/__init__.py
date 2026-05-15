"""Positive (semantic-equivalent) transformation rules registry."""

from wfcllm.lang.python.transform.base import Rule
from wfcllm.lang.python.transform.positive.api_calls import (
    ExplicitDefaultPrint, ExplicitDefaultRange, ExplicitDefaultOpen,
    ExplicitDefaultSorted, ExplicitDefaultMinMax, ExplicitDefaultZip,
    ExplicitDefaultRandomSeed, ExplicitDefaultHtmlEscape,
    ExplicitDefaultRound, ExplicitDefaultJsonDump,
    LibraryAliasReplace, ThirdPartyFuncReplace,
)
from wfcllm.lang.python.transform.positive.syntax_init import ListInit, DictInit, TypeCheck, StringFormat
from wfcllm.lang.python.transform.positive.control_flow import LoopConvert, IterationConvert, ComprehensionConvert, BranchFlip
from wfcllm.lang.python.transform.positive.expression_logic import (
    OperandSwap, ComparisonFlip, UnarySimplify, DeMorgan, ArithmeticAssociativity,
)
from wfcllm.lang.python.transform.positive.identifier import VariableRename, NameObfuscation
from wfcllm.lang.python.transform.positive.formatting import FixSpacing, FixCommentSymbols

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
