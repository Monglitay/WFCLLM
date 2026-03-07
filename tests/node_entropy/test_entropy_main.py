# tests/node_entropy/test_entropy_main.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiment" / "node_entropy"))

from entropy_main import (
    align_nodes_by_type,
    collect_token_sequences_for_block,
    aggregate_by_node_type,
)


# --- align_nodes_by_type ---

def test_align_nodes_equal_count():
    original = ["nodeA0", "nodeA1"]  # 2 for_statement nodes (mocked as strings)
    variants = [["nodeA0v", "nodeA1v"]]
    result = align_nodes_by_type(original, variants)
    # result[0] = [original[0], variants[0][0]]
    # result[1] = [original[1], variants[0][1]]
    assert result[0] == ["nodeA0", "nodeA0v"]
    assert result[1] == ["nodeA1", "nodeA1v"]


def test_align_nodes_variant_has_more():
    original = ["n0"]
    variants = [["n0v", "n1v"]]  # variant has 2 nodes, original has 1
    result = align_nodes_by_type(original, variants)
    assert len(result) == 2
    assert result[0] == ["n0", "n0v"]
    assert result[1] == ["n1v"]  # extra variant node is standalone


def test_align_nodes_variant_has_fewer():
    original = ["n0", "n1"]
    variants = [["n0v"]]  # variant has 1, original has 2
    result = align_nodes_by_type(original, variants)
    assert len(result) == 2
    assert result[0] == ["n0", "n0v"]
    assert result[1] == ["n1"]  # n1 has no variant counterpart


def test_aggregate_by_node_type_merges_entropy():
    # Two node types, multiple entropy values each
    all_entropies = {
        "for_statement": [0.0, 1.0, 2.0],
        "if_statement": [1.5],
    }
    result = aggregate_by_node_type(all_entropies)
    assert result["for_statement"]["mean_entropy"] == 1.0
    assert result["for_statement"]["sample_count"] == 3
    assert result["if_statement"]["mean_entropy"] == 1.5
