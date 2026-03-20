"""Tests for canonical block contracts."""

from hashlib import sha256

from wfcllm.watermark.entropy import NodeEntropyEstimator
from wfcllm.common.block_contract import build_block_contracts


def test_build_block_contracts_is_deterministic():
    code = "def add(a, b):\n    total = a + b\n    return total\n"
    first = build_block_contracts(code)
    second = build_block_contracts(code)

    assert first == second
    assert len(first) > 0


def test_build_block_contracts_only_emits_simple_blocks():
    code = (
        "def add(a, b):\n"
        "    total = a + b\n"
        "    return total\n"
        "if True:\n"
        "    result = add(1, 2)\n"
    )
    contracts = build_block_contracts(code)
    node_types = {contract.node_type for contract in contracts}

    assert "function_definition" not in node_types
    assert "if_statement" not in node_types
    assert "expression_statement" in node_types
    assert "return_statement" in node_types


def test_build_block_contracts_resolves_compound_parent_node_type():
    code = (
        "def f(x):\n"
        "    if x:\n"
        "        return 1\n"
        "    return 0\n"
    )
    contracts = build_block_contracts(code)

    nested_return = next(
        contract
        for contract in contracts
        if contract.node_type == "return_statement" and contract.start_line == 3
    )
    assert nested_return.parent_node_type == "if_statement"


def test_build_block_contracts_root_simple_block_payload_is_canonical():
    code = "x = 1\n"
    contracts = build_block_contracts(code)

    assert len(contracts) == 1
    contract = contracts[0]
    estimator = NodeEntropyEstimator()
    expected_source = "x = 1"

    assert contract.parent_node_type == "module"
    assert contract.block_text_hash == sha256(expected_source.encode("utf-8")).hexdigest()
    assert contract.start_line == 1
    assert contract.end_line == 1
    assert contract.entropy_units == estimator.estimate_block_entropy_units(expected_source)
    assert contract.gamma_target == 0.0
    assert contract.k == 0
    assert contract.gamma_effective == 0.0
