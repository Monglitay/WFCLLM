# tests/node_entropy/test_entropy.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "experiment" / "node_entropy"))

import math
from entropy import token_entropy, node_entropy


def test_token_entropy_uniform():
    # 4 equally likely tokens → entropy = log2(4) = 2.0
    tokens = ["a", "b", "c", "d"]
    assert abs(token_entropy(tokens) - 2.0) < 1e-9


def test_token_entropy_certain():
    # Only one token → entropy = 0
    tokens = ["x", "x", "x"]
    assert token_entropy(tokens) == 0.0


def test_token_entropy_two_equal():
    # 2 equally likely → entropy = 1.0
    tokens = ["a", "b"]
    assert abs(token_entropy(tokens) - 1.0) < 1e-9


def test_node_entropy_single_position():
    # Each token_sequence is a 1-token list
    # position 0: ["a","a","b"] → 2/3, 1/3 → H = -(2/3*log2(2/3) + 1/3*log2(1/3))
    token_sequences = [["a"], ["a"], ["b"]]
    expected = -(2/3 * math.log2(2/3) + 1/3 * math.log2(1/3))
    assert abs(node_entropy(token_sequences) - expected) < 1e-9


def test_node_entropy_multiple_positions():
    # 2 positions: pos0=[a,a], pos1=[b,c]
    # H(pos0) = 0.0, H(pos1) = 1.0 → mean = 0.5
    token_sequences = [["a", "b"], ["a", "c"]]
    assert abs(node_entropy(token_sequences) - 0.5) < 1e-9


def test_node_entropy_empty():
    assert node_entropy([]) == 0.0
