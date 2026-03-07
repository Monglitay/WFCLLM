"""Transform rules registry."""

from .base import Match, Rule, parse_code
from .api_calls import (
    ExplicitDefaultPrint, ExplicitDefaultRange, ExplicitDefaultOpen,
    ExplicitDefaultSorted, ExplicitDefaultMinMax, ExplicitDefaultZip,
    ExplicitDefaultRandomSeed, ExplicitDefaultHtmlEscape,
    ExplicitDefaultRound, ExplicitDefaultJsonDump,
    LibraryAliasReplace, ThirdPartyFuncReplace,
)
from .syntax_init import ListInit, DictInit, TypeCheck, StringFormat
from .control_flow import LoopConvert, IterationConvert, ComprehensionConvert, BranchFlip
from .expression_logic import (
    OperandSwap, ComparisonFlip, UnarySimplify, DeMorgan, ArithmeticAssociativity,
)
from .identifier import VariableRename, NameObfuscation
from .formatting import FixSpacing, FixCommentSymbols

ALL_RULES: list[Rule] = [
    # API calls (12)
    ExplicitDefaultPrint(), ExplicitDefaultRange(), ExplicitDefaultOpen(),
    ExplicitDefaultSorted(), ExplicitDefaultMinMax(), ExplicitDefaultZip(),
    ExplicitDefaultRandomSeed(), ExplicitDefaultHtmlEscape(),
    ExplicitDefaultRound(), ExplicitDefaultJsonDump(),
    LibraryAliasReplace(), ThirdPartyFuncReplace(),
    # Syntax init (4)
    ListInit(), DictInit(), TypeCheck(), StringFormat(),
    # Control flow (4)
    LoopConvert(), IterationConvert(), ComprehensionConvert(), BranchFlip(),
    # Expression logic (5)
    OperandSwap(), ComparisonFlip(), UnarySimplify(), DeMorgan(), ArithmeticAssociativity(),
    # Identifier (2)
    VariableRename(), NameObfuscation(),
    # Formatting (2)
    FixSpacing(), FixCommentSymbols(),
]


def get_all_rules() -> list[Rule]:
    return list(ALL_RULES)
