"""Tree-sitter Java parsing and statement block extraction."""
from __future__ import annotations

import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Tree

from wfcllm.lang.tree_sitter import SourceBlock, extract_statement_blocks as extract

JAVA_LANGUAGE = Language(tsjava.language())

SIMPLE_STATEMENT_TYPES = frozenset(
    {
        "assert_statement",
        "break_statement",
        "continue_statement",
        "expression_statement",
        "local_variable_declaration",
        "return_statement",
        "throw_statement",
        "yield_statement",
    }
)

COMPOUND_STATEMENT_TYPES = frozenset(
    {
        "class_declaration",
        "constructor_declaration",
        "do_statement",
        "enhanced_for_statement",
        "enum_declaration",
        "for_statement",
        "if_statement",
        "interface_declaration",
        "method_declaration",
        "record_declaration",
        "switch_expression",
        "synchronized_statement",
        "try_statement",
        "while_statement",
    }
)


class JavaParser:
    """Singleton Tree-sitter Java parser."""

    _instance: JavaParser | None = None
    _parser: Parser

    def __new__(cls) -> JavaParser:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parser = Parser(JAVA_LANGUAGE)
        return cls._instance

    def parse(self, source: str) -> Tree:
        return self._parser.parse(source.encode("utf-8"))

    @property
    def raw_parser(self) -> Parser:
        return self._parser


def extract_statement_blocks(source: str) -> list[SourceBlock]:
    return extract(
        source,
        parser=JavaParser().raw_parser,
        simple_types=SIMPLE_STATEMENT_TYPES,
        compound_types=COMPOUND_STATEMENT_TYPES,
    )

