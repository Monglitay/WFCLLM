"""Java implementation of the pluggable language adapter."""
from __future__ import annotations

from wfcllm.lang.adapter import LanguageAdapter, StatementTypes
from wfcllm.lang.java.parser import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    extract_statement_blocks,
)
from wfcllm.lang.registry import register
from wfcllm.lang.tree_sitter import SourceBlock

_STATEMENT_TYPES = StatementTypes(
    simple=SIMPLE_STATEMENT_TYPES,
    compound=COMPOUND_STATEMENT_TYPES,
)


@register("java")
class JavaAdapter(LanguageAdapter):
    name = "java"

    def statement_types(self) -> StatementTypes:
        return _STATEMENT_TYPES

    def extract_blocks(self, source: str) -> list[SourceBlock]:
        return extract_statement_blocks(source)

    def positive_rules(self) -> list:
        return []

    def negative_rules(self) -> list:
        return []

