# tests/node_entropy/test_ast_utils.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiment" / "node_entropy"))

from ast_utils import get_node_tokens, get_nodes_by_type

SIMPLE = "x = 1"
FOR_CODE = "for i in range(10):\n    x = i"


def test_get_nodes_by_type_finds_for():
    result = get_nodes_by_type(FOR_CODE, "for_statement")
    assert len(result) == 1


def test_get_nodes_by_type_empty_when_absent():
    result = get_nodes_by_type(SIMPLE, "for_statement")
    assert result == []


def test_get_node_tokens_returns_strings():
    nodes = get_nodes_by_type(FOR_CODE, "for_statement")
    tokens = get_node_tokens(FOR_CODE, nodes[0])
    assert isinstance(tokens, list)
    assert all(isinstance(t, str) for t in tokens)
    assert "for" in tokens
    assert "i" in tokens


def test_get_node_tokens_simple_assignment():
    nodes = get_nodes_by_type(SIMPLE, "expression_statement")
    tokens = get_node_tokens(SIMPLE, nodes[0])
    assert "x" in tokens
    assert "=" in tokens
    assert "1" in tokens
