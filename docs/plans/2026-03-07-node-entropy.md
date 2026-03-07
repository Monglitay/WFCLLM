# Node Entropy Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Compute AST node entropy H_Node for each node type by treating transformed code variants as a token probability distribution, then produce per-node-type statistical summaries.

**Architecture:** Parse original code and all variants with tree-sitter, align nodes of the same type by order of occurrence, collect token sequences per node instance across variants, compute Shannon entropy per token position, average across positions to get H_Node per node instance, then aggregate by node type.

**Tech Stack:** Python 3.11+, tree-sitter, tree-sitter-python, math/collections stdlib. No new dependencies needed.

---

### Task 1: Scaffold directory and config

**Files:**
- Create: `experiment/node_entropy/__init__.py`
- Create: `experiment/node_entropy/config.py`
- Create: `experiment/node_entropy/results/.gitkeep`

**Step 1: Create the directory**

```bash
mkdir -p experiment/node_entropy/results
```

**Step 2: Write `__init__.py`**

```python
```
(empty file)

**Step 3: Write `config.py`**

```python
"""Configuration for node entropy experiment."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "node_entropy_results.json"

INPUT_FILE = (
    PROJECT_ROOT
    / "experiment"
    / "statement_block_transform"
    / "results"
    / "mbpp_blocks_transformed.json"
)
```

**Step 4: Commit**

```bash
git add experiment/node_entropy/
git commit -m "feat: scaffold node_entropy experiment directory"
```

---

### Task 2: Write `ast_utils.py` — AST parsing and token extraction

**Files:**
- Create: `experiment/node_entropy/ast_utils.py`
- Create: `tests/node_entropy/test_ast_utils.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/test_ast_utils.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'ast_utils'`

**Step 3: Write `ast_utils.py`**

```python
"""AST parsing utilities for node entropy experiment."""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def _parse(source: str):
    return _parser.parse(source.encode("utf-8"))


def get_nodes_by_type(source: str, node_type: str) -> list[Node]:
    """Return all AST nodes of the given type, in order of appearance."""
    tree = _parse(source)
    results: list[Node] = []
    _collect(tree.root_node, node_type, results)
    return results


def _collect(node: Node, target_type: str, results: list[Node]) -> None:
    if node.type == target_type:
        results.append(node)
    for child in node.children:
        _collect(child, target_type, results)


def get_node_tokens(source: str, node: Node) -> list[str]:
    """Extract leaf tokens from a tree-sitter node using byte offsets."""
    node_source = source.encode("utf-8")[node.start_byte : node.end_byte]
    # Re-parse the node's source to get clean leaf tokens
    sub_tree = _parser.parse(node_source)
    tokens: list[str] = []
    _collect_leaves(sub_tree.root_node, node_source, tokens)
    return tokens


def _collect_leaves(node: Node, source_bytes: bytes, tokens: list[str]) -> None:
    if not node.children:
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8").strip()
        if text:
            tokens.append(text)
        return
    for child in node.children:
        _collect_leaves(child, source_bytes, tokens)


def get_all_node_types(source: str) -> set[str]:
    """Return all unique node types present in the AST."""
    tree = _parse(source)
    types: set[str] = set()
    _collect_types(tree.root_node, types)
    return types


def _collect_types(node: Node, types: set[str]) -> None:
    types.add(node.type)
    for child in node.children:
        _collect_types(child, types)
```

**Step 4: Run tests to verify they pass**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/test_ast_utils.py -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
mkdir -p tests/node_entropy && touch tests/node_entropy/__init__.py
git add experiment/node_entropy/ast_utils.py tests/node_entropy/
git commit -m "feat: add ast_utils for node token extraction"
```

---

### Task 3: Write `entropy.py` — Shannon entropy calculation

**Files:**
- Create: `experiment/node_entropy/entropy.py`
- Create: `tests/node_entropy/test_entropy.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/test_entropy.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'entropy'`

**Step 3: Write `entropy.py`**

```python
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
```

**Step 4: Run tests to verify they pass**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/test_entropy.py -v
```
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add experiment/node_entropy/entropy.py tests/node_entropy/test_entropy.py
git commit -m "feat: add entropy computation for AST node token distributions"
```

---

### Task 4: Write `entropy_main.py` — main pipeline

**Files:**
- Create: `experiment/node_entropy/entropy_main.py`
- Create: `tests/node_entropy/test_entropy_main.py`

**Step 1: Write the failing tests**

```python
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
```

**Step 2: Run tests to verify they fail**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/test_entropy_main.py -v
```
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write `entropy_main.py`**

```python
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
    original_source = block.get("original_source", "")
    variants = block.get("variants", [])

    if not original_source:
        return {}

    # Find all node types in the original
    from ast_utils import get_all_node_types
    node_types = get_all_node_types(original_source)

    result: dict[str, list[list[list[str]]]] = {}

    for node_type in node_types:
        original_nodes = get_nodes_by_type(original_source, node_type)
        if not original_nodes:
            continue

        # Collect variant node lists
        variant_node_lists = []
        for variant in variants:
            vsrc = variant.get("transformed_source", "")
            if vsrc:
                variant_node_lists.append(get_nodes_by_type(vsrc, node_type))

        # Align nodes
        groups = align_nodes_by_type(original_nodes, variant_node_lists)

        # Extract token sequences per group
        token_groups: list[list[list[str]]] = []
        for group_nodes in groups:
            token_seqs = []
            for node in group_nodes:
                # node is either from original or variant source
                # We need the source — store source alongside node
                token_seqs.append(node)  # will replace below
            token_groups.append(token_seqs)

        result[node_type] = token_groups

    return result


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
```

**Step 4: Run tests to verify they pass**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/test_entropy_main.py -v
```
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add experiment/node_entropy/entropy_main.py tests/node_entropy/test_entropy_main.py
git commit -m "feat: add entropy_main pipeline for node entropy computation"
```

---

### Task 5: Run the full pipeline and verify output

**Step 1: Run the pipeline**

```bash
cd experiment/node_entropy && python entropy_main.py
```
Expected: prints progress, ends with "Done. Output: .../node_entropy_results.json"

**Step 2: Verify output structure**

```bash
python3 -c "
import json
with open('experiment/node_entropy/results/node_entropy_results.json') as f:
    d = json.load(f)
print('Metadata:', d['metadata'])
nte = d['node_type_entropy']
print(f'Node types: {len(nte)}')
for k, v in sorted(nte.items(), key=lambda x: -x[1]['mean_entropy'])[:5]:
    print(f'  {k}: mean={v[\"mean_entropy\"]:.4f}, count={v[\"sample_count\"]}')
"
```
Expected: Output showing node types with entropy values, top 5 by mean entropy.

**Step 3: Commit results**

```bash
git add experiment/node_entropy/results/node_entropy_results.json
git commit -m "feat: add node entropy experiment results"
```

---

### Task 6: Run all tests to confirm nothing is broken

**Step 1: Run full test suite from project root**

```bash
python -m pytest tests/ -v --ignore=tests/node_entropy
```
Expected: All existing tests PASS (node_entropy tests also pass if run with path fix)

**Step 2: Run node entropy tests**

```bash
cd experiment/node_entropy && python -m pytest ../../tests/node_entropy/ -v
```
Expected: All node entropy tests PASS

**Step 3: Final commit if any fixes were needed**

```bash
git add -A
git commit -m "test: verify all tests pass after node entropy experiment"
```
