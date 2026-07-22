from __future__ import annotations


def test_java_adapter_extracts_nested_statement_blocks():
    from wfcllm import lang

    source = """class Solution {
    int addPositive(int x) {
        int total = 0;
        if (x > 0) {
            total = x;
        }
        return total;
    }
}
"""
    adapter = lang.get("java")
    blocks = adapter.extract_blocks(source)

    assert adapter.name == "java"
    assert {block.node_type for block in blocks} >= {
        "class_declaration",
        "method_declaration",
        "local_variable_declaration",
        "if_statement",
        "expression_statement",
        "return_statement",
    }
    assert any(block.depth > 0 and block.parent_id is not None for block in blocks)
    method = next(block for block in blocks if block.node_type == "method_declaration")
    nested_assignment = next(
        block
        for block in blocks
        if block.node_type == "expression_statement" and "total = x" in block.source
    )
    assert nested_assignment.block_id in method.children_ids
    assert nested_assignment.parent_id != method.block_id
    assert adapter.positive_rules() == []
    assert adapter.negative_rules() == []


def test_java_statement_types_are_language_specific():
    from wfcllm import lang

    types = lang.get("java").statement_types()
    assert "local_variable_declaration" in types.simple
    assert "method_declaration" in types.compound
