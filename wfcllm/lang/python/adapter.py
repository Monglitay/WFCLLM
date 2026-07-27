"""PythonAdapter — wraps the existing tree-sitter Python parsing + transform rules."""
from __future__ import annotations

from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.python.parser import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    StatementBlock,
    extract_statement_blocks,
)
from wfcllm.lang.registry import register


_STATEMENT_TYPES = StatementTypes(
    simple=SIMPLE_STATEMENT_TYPES,
    compound=COMPOUND_STATEMENT_TYPES,
)


@register("python")
class PythonAdapter(LanguageAdapter):
    """Tree-sitter-based Python adapter."""

    name = "python"

    def statement_types(self) -> StatementTypes:
        return _STATEMENT_TYPES

    def extract_blocks(self, source: str) -> list[StatementBlock]:
        return extract_statement_blocks(source)

    def positive_rules(self) -> list:
        from wfcllm.lang.python.transform.positive import get_all_positive_rules
        return get_all_positive_rules()

    def negative_rules(self) -> list:
        from wfcllm.lang.python.transform.negative import get_all_negative_rules
        return get_all_negative_rules()
