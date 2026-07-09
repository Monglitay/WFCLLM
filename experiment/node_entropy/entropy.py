"""Shannon entropy calculation for AST node token distributions."""

from __future__ import annotations

import math
from collections import Counter


def token_entropy(tokens: list[str]) -> float:
    """Compute Shannon entropy (bits) of a list of token observations.

    Each token in the list is treated as one sample from the distribution.
    H = -sum(p * log2(p)) for each unique token p.
    """
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    return -sum(
        (c / total) * math.log2(c / total)
        for c in counts.values()
    )


def node_entropy(token_sequences: list[list[str]]) -> float:
    """Compute H_Node for a set of token sequences representing one node.

    Each sequence is one variant's token list for the node.
    Aligns by position: position i collects token_sequences[j][i] for all j.
    Sequences shorter than the max length contribute only to positions they cover.

    Returns the mean Shannon entropy across all positions.
    H_Node = (1/k) * sum_i H(tokens at position i)
    """
    if not token_sequences:
        return 0.0

    max_len = max(len(seq) for seq in token_sequences)
    if max_len == 0:
        return 0.0

    total_entropy = 0.0
    for i in range(max_len):
        tokens_at_pos = [seq[i] for seq in token_sequences if i < len(seq)]
        total_entropy += token_entropy(tokens_at_pos)

    return total_entropy / max_len
