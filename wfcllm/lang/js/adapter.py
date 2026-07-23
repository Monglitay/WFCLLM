"""JavaScript language adapter."""
from __future__ import annotations

from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.js.parser import (
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


@register("js")
class JavaScriptAdapter(LanguageAdapter):
    """Dependency-light JavaScript adapter."""

    name = "js"

    def statement_types(self) -> StatementTypes:
        return _STATEMENT_TYPES

    def extract_blocks(self, source: str) -> list[StatementBlock]:
        return extract_statement_blocks(source)

    def positive_rules(self) -> list:
        return []

    def negative_rules(self) -> list:
        return []
