"""Tree-sitter C++ parsing and statement block extraction."""
from __future__ import annotations

import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser, Tree

from wfcllm.lang.tree_sitter import SourceBlock, extract_statement_blocks as extract

CPP_LANGUAGE = Language(tscpp.language())

SIMPLE_STATEMENT_TYPES = frozenset(
    {
        "break_statement",
        "co_return_statement",
        "continue_statement",
        "declaration",
        "expression_statement",
        "goto_statement",
        "return_statement",
        "throw_statement",
    }
)

COMPOUND_STATEMENT_TYPES = frozenset(
    {
        "class_specifier",
        "do_statement",
        "for_range_loop",
        "for_statement",
        "function_definition",
        "if_statement",
        "namespace_definition",
        "struct_specifier",
        "switch_statement",
        "try_statement",
        "while_statement",
    }
)


class CppParser:
    """Singleton Tree-sitter C++ parser."""

    _instance: CppParser | None = None
    _parser: Parser

    def __new__(cls) -> CppParser:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parser = Parser(CPP_LANGUAGE)
        return cls._instance

    def parse(self, source: str) -> Tree:
        return self._parser.parse(source.encode("utf-8"))

    @property
    def raw_parser(self) -> Parser:
        return self._parser


def extract_statement_blocks(source: str) -> list[SourceBlock]:
    return extract(
        source,
        parser=CppParser().raw_parser,
        simple_types=SIMPLE_STATEMENT_TYPES,
        compound_types=COMPOUND_STATEMENT_TYPES,
    )

