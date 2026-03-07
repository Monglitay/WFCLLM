"""Main pipeline for node entropy experiment."""

from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

from ast_utils import get_node_tokens, get_nodes_by_type, get_all_node_types
from entropy import node_entropy


def align_nodes_by_type(
    original_nodes: list,
    variant_node_lists: list[list],
) -> list[list]:
    """Align nodes from original and variants by position index.

    Args:
        original_nodes: Nodes of one type from the original code.
        variant_node_lists: For each variant, the list of nodes of the same type.

    Returns:
        List of groups. Each group contains the original node (if present)
        followed by the corresponding variant nodes (if present).
        Extra variant nodes become standalone groups.
    """
    max_len = max(
        len(original_nodes),
        max((len(v) for v in variant_node_lists), default=0),
    )
    groups: list[list] = []
    for i in range(max_len):
        group = []
        if i < len(original_nodes):
            group.append(original_nodes[i])
        for variant_nodes in variant_node_lists:
            if i < len(variant_nodes):
                group.append(variant_nodes[i])
        groups.append(group)
    return groups


def collect_token_sequences_for_block(
    block: dict,
) -> dict[str, list[list[list[str]]]]:
    """For one block, collect per-node-type token sequence groups.

    Returns:
        {node_type: [[tokens_variant0, tokens_variant1, ...], ...]}
        Each inner list is a group of token sequences for one aligned node position.
    """
    return _collect_token_groups_for_block(block)


def _collect_token_groups_for_block(
    block: dict,
) -> dict[str, list[list[list[str]]]]:
    """Collect token sequence groups per node type for one block.

    Returns {node_type: [group0_token_seqs, group1_token_seqs, ...]}
    where each group_token_seqs is a list of token lists (one per variant/original).
    """
    original_source = block.get("original_source", "")
    variants = block.get("variants", [])

    if not original_source:
        return {}

    node_types = get_all_node_types(original_source)
    result: dict[str, list[list[list[str]]]] = {}

    for node_type in node_types:
        original_nodes = get_nodes_by_type(original_source, node_type)
        if not original_nodes:
            continue

        # Get token seqs from original nodes
        original_token_seqs = [
            get_node_tokens(original_source, n) for n in original_nodes
        ]

        # Get token seqs from each variant
        variant_token_seq_lists: list[list[list[str]]] = []
        for variant in variants:
            vsrc = variant.get("transformed_source", "")
            if not vsrc:
                continue
            vnodes = get_nodes_by_type(vsrc, node_type)
            variant_token_seq_lists.append(
                [get_node_tokens(vsrc, n) for n in vnodes]
            )

        # Align by position
        groups = align_nodes_by_type(original_token_seqs, variant_token_seq_lists)
        result[node_type] = groups

    return result


def aggregate_by_node_type(
    all_entropies: dict[str, list[float]],
) -> dict[str, dict]:
    """Compute mean, std, count for each node type's entropy values."""
    result = {}
    for node_type, values in all_entropies.items():
        result[node_type] = {
            "mean_entropy": statistics.mean(values),
            "std_entropy": statistics.stdev(values) if len(values) > 1 else 0.0,
            "sample_count": len(values),
            "entropy_values": values,
        }
    return result


def run(input_file: Path, output_file: Path) -> None:
    """Main pipeline: load data, compute entropy, write results."""
    print(f"Loading {input_file}...")
    with open(input_file, encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    print(f"Processing {len(samples)} samples...")

    # {node_type: [entropy_value, ...]}
    all_entropies: dict[str, list[float]] = {}

    for sample_idx, sample in enumerate(samples):
        for block in sample.get("blocks", []):
            token_groups = _collect_token_groups_for_block(block)
            for node_type, groups in token_groups.items():
                for token_seqs in groups:
                    h = node_entropy(token_seqs)
                    all_entropies.setdefault(node_type, []).append(h)

        if (sample_idx + 1) % 100 == 0:
            print(f"  {sample_idx + 1}/{len(samples)}")

    node_type_stats = aggregate_by_node_type(all_entropies)

    output = {
        "metadata": {
            "input_file": str(input_file),
            "total_samples": len(samples),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "node_type_entropy": node_type_stats,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {output_file}")
    print(f"  Node types analyzed: {len(node_type_stats)}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import INPUT_FILE, OUTPUT_FILE
    run(INPUT_FILE, OUTPUT_FILE)
