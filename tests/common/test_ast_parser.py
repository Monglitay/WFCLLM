"""Tests for wfcllm.common.ast_parser."""

import pytest
from wfcllm.common.ast_parser import PythonParser, StatementBlock, extract_statement_blocks


class TestPythonParser:
    def test_singleton(self):
        p1 = PythonParser()
        p2 = PythonParser()
        assert p1 is p2

    def test_parse_returns_tree(self):
        p = PythonParser()
        tree = p.parse("x = 1")
        assert tree.root_node.type == "module"

    def test_parse_empty(self):
        p = PythonParser()
        tree = p.parse("")
        assert tree.root_node.type == "module"


class TestExtractStatementBlocks:
    def test_single_simple_statement(self):
        blocks = extract_statement_blocks("x = 1")
        assert len(blocks) == 1
        b = blocks[0]
        assert b.block_type == "simple"
        assert b.node_type == "expression_statement"
        assert b.source == "x = 1"
        assert b.depth == 0
        assert b.parent_id is None

    def test_compound_statement(self):
        code = "for i in range(10):\n    print(i)"
        blocks = extract_statement_blocks(code)
        # Should have for_statement (compound) + expression_statement (simple child)
        compound = [b for b in blocks if b.block_type == "compound"]
        simple = [b for b in blocks if b.block_type == "simple"]
        assert len(compound) == 1
        assert compound[0].node_type == "for_statement"
        assert len(simple) == 1
        assert simple[0].parent_id == compound[0].block_id

    def test_nested_depth(self):
        code = "if True:\n    for i in range(3):\n        print(i)"
        blocks = extract_statement_blocks(code)
        depths = [b.depth for b in blocks]
        assert 0 in depths
        assert 1 in depths
        assert 2 in depths

    def test_children_ids(self):
        code = "if True:\n    x = 1\n    y = 2"
        blocks = extract_statement_blocks(code)
        parent = [b for b in blocks if b.block_type == "compound"][0]
        assert len(parent.children_ids) == 2

    def test_multiple_simple_statements(self):
        code = "x = 1\ny = 2\nz = 3"
        blocks = extract_statement_blocks(code)
        assert len(blocks) == 3
        assert all(b.block_type == "simple" for b in blocks)
        assert all(b.depth == 0 for b in blocks)

    def test_function_definition(self):
        code = "def foo(x):\n    return x + 1"
        blocks = extract_statement_blocks(code)
        func_blocks = [b for b in blocks if b.node_type == "function_definition"]
        assert len(func_blocks) == 1
        assert func_blocks[0].block_type == "compound"

    def test_empty_code(self):
        blocks = extract_statement_blocks("")
        assert blocks == []

    def test_line_numbers(self):
        code = "x = 1\ny = 2"
        blocks = extract_statement_blocks(code)
        assert blocks[0].start_line == 1
        assert blocks[1].start_line == 2
