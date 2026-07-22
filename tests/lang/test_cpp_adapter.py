from __future__ import annotations


def test_cpp_adapter_extracts_nested_statement_blocks():
    from wfcllm import lang

    source = """int add_positive(int x) {
    int total = 0;
    if (x > 0) {
        total = x;
    }
    return total;
}
"""
    adapter = lang.get("cpp")
    blocks = adapter.extract_blocks(source)

    assert adapter.name == "cpp"
    assert {block.node_type for block in blocks} >= {
        "function_definition",
        "declaration",
        "if_statement",
        "expression_statement",
        "return_statement",
    }
    assert any(block.depth > 0 and block.parent_id is not None for block in blocks)
    function = next(block for block in blocks if block.node_type == "function_definition")
    nested_assignment = next(
        block
        for block in blocks
        if block.node_type == "expression_statement" and "total = x" in block.source
    )
    assert nested_assignment.block_id in function.children_ids
    assert nested_assignment.parent_id != function.block_id
    assert adapter.positive_rules() == []
    assert adapter.negative_rules() == []


def test_cpp_statement_types_are_language_specific():
    from wfcllm import lang

    types = lang.get("cpp").statement_types()
    assert "declaration" in types.simple
    assert "function_definition" in types.compound
