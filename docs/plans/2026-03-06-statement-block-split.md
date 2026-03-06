# Statement Block Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split MBPP dataset code samples into structured statement blocks using tree-sitter, outputting JSON with block types, nesting, and metadata.

**Architecture:** A Python script uses tree-sitter to parse each MBPP code sample into an AST, then recursively extracts statement blocks (simple vs compound) with parent-child relationships. Results are saved as a single JSON file.

**Tech Stack:** Python 3.13, tree-sitter, tree-sitter-python, Hugging Face datasets, conda WFCLLM environment.

---

### Task 1: Install Dependencies

**Step 1: Install tree-sitter and datasets packages**

Run:
```bash
conda run -n WFCLLM pip install tree-sitter tree-sitter-python datasets
```

Expected: packages install successfully.

**Step 2: Verify imports work**

Run:
```bash
conda run -n WFCLLM python -c "import tree_sitter_python as tspython; from tree_sitter import Language, Parser; from datasets import load_dataset; print('OK')"
```

Expected: prints `OK`.

---

### Task 2: Create config.py

**Files:**
- Create: `experiment/statement_block_split/config.py`

**Step 1: Create directory structure**

Run:
```bash
mkdir -p experiment/statement_block_split/results
```

**Step 2: Write config.py**

```python
"""Configuration for statement block split experiment."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
OUTPUT_FILE = RESULTS_DIR / "mbpp_blocks.json"

# Dataset
DATASET_NAME = "google-research-datasets/mbpp"
DATASET_CONFIG = "full"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "mbpp"

# Statement block node types
SIMPLE_STATEMENT_TYPES = frozenset({
    "expression_statement",
    "return_statement",
    "assert_statement",
    "import_statement",
    "import_from_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "raise_statement",
    "delete_statement",
    "global_statement",
    "nonlocal_statement",
})

COMPOUND_STATEMENT_TYPES = frozenset({
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
})

STATEMENT_TYPES = SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES
```

**Step 3: Verify config loads**

Run:
```bash
cd /home/monglitay/PycharmProjects/WFCLLM && conda run -n WFCLLM python -c "
import sys; sys.path.insert(0, '.')
from experiment.statement_block_split.config import *
print('RESULTS_DIR:', RESULTS_DIR)
print('Simple types:', len(SIMPLE_STATEMENT_TYPES))
print('Compound types:', len(COMPOUND_STATEMENT_TYPES))
"
```

Expected: prints paths and counts (12 simple, 8 compound).

**Step 4: Commit**

```bash
git add experiment/statement_block_split/config.py
git commit -m "feat: add config for statement block split experiment"
```

---

### Task 3: Write split.py — Core Parsing Logic

**Files:**
- Create: `experiment/statement_block_split/split.py`

**Step 1: Write the block extraction function**

```python
"""Split Python code into statement blocks using tree-sitter."""

import json
from datetime import datetime, timezone

import tree_sitter_python as tspython
from tree_sitter import Language, Parser

from config import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    STATEMENT_TYPES,
)

PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


def extract_blocks(code: str) -> list[dict]:
    """Parse code and recursively extract all statement blocks.

    Returns a flat list of block dicts with parent-child references.
    """
    tree = parser.parse(code.encode("utf-8"))
    root = tree.root_node
    blocks: list[dict] = []
    _extract_recursive(root, blocks, depth=0, parent_id=None)
    return blocks


def _extract_recursive(
    node, blocks: list[dict], depth: int, parent_id: int | None
) -> None:
    """Recursively walk AST children, extracting statement blocks."""
    for child in node.children:
        if child.type not in STATEMENT_TYPES:
            continue

        block_id = len(blocks)
        is_compound = child.type in COMPOUND_STATEMENT_TYPES

        block = {
            "id": block_id,
            "type": "compound" if is_compound else "simple",
            "node_type": child.type,
            "source": child.text.decode("utf-8"),
            "start_line": child.start_point[0] + 1,
            "end_line": child.end_point[0] + 1,
            "depth": depth,
            "parent_id": parent_id,
            "children_ids": [],
        }
        blocks.append(block)

        if is_compound:
            child_count_before = len(blocks)
            _extract_recursive(child, blocks, depth + 1, block_id)
            block["children_ids"] = list(
                range(child_count_before, len(blocks))
            )
```

**Step 2: Verify parsing works on a simple example**

Run:
```bash
cd /home/monglitay/PycharmProjects/WFCLLM/experiment/statement_block_split && conda run -n WFCLLM python -c "
from split import extract_blocks
import json

code = '''
def foo(x):
    if x > 0:
        return x
    return -x

print(foo(1))
'''.strip()

blocks = extract_blocks(code)
for b in blocks:
    print(f'[{b[\"id\"]}] depth={b[\"depth\"]} {b[\"type\"]:8s} {b[\"node_type\"]:25s} children={b[\"children_ids\"]}')
"
```

Expected output (approximately):
```
[0] depth=0 compound function_definition       children=[1, 2, 3]
[1] depth=1 compound if_statement              children=[2]
[2] depth=2 simple   return_statement           children=[]
[3] depth=1 simple   return_statement           children=[]
[4] depth=0 simple   expression_statement       children=[]
```

**Step 3: Commit**

```bash
git add experiment/statement_block_split/split.py
git commit -m "feat: add core block extraction logic with tree-sitter"
```

---

### Task 4: Add Dataset Processing to split.py

**Files:**
- Modify: `experiment/statement_block_split/split.py`

**Step 1: Add the dataset processing and main function**

Append to `split.py`:

```python
def process_sample(sample: dict) -> dict:
    """Process a single MBPP sample, returning structured block data."""
    code = sample["code"]
    blocks = extract_blocks(code)

    simple_count = sum(1 for b in blocks if b["type"] == "simple")
    compound_count = sum(1 for b in blocks if b["type"] == "compound")
    max_depth = max((b["depth"] for b in blocks), default=0)

    return {
        "task_id": sample["task_id"],
        "prompt": sample["text"],
        "original_code": code,
        "blocks": blocks,
        "stats": {
            "total_blocks": len(blocks),
            "simple_blocks": simple_count,
            "compound_blocks": compound_count,
            "max_depth": max_depth,
        },
    }


def main(limit: int | None = None) -> None:
    """Load MBPP dataset, process all samples, write JSON output.

    Args:
        limit: If set, only process first N samples (for validation).
    """
    from datasets import load_dataset

    from config import (
        DATA_CACHE_DIR,
        DATASET_CONFIG,
        DATASET_NAME,
        OUTPUT_FILE,
        RESULTS_DIR,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset {DATASET_NAME} ({DATASET_CONFIG})...")
    ds = load_dataset(DATASET_NAME, DATASET_CONFIG, cache_dir=str(DATA_CACHE_DIR))

    all_samples = []
    for split_name in ds:
        all_samples.extend(ds[split_name])

    if limit is not None:
        all_samples = all_samples[:limit]

    print(f"Processing {len(all_samples)} samples...")
    results = []
    failed = 0
    for i, sample in enumerate(all_samples):
        try:
            result = process_sample(sample)
            results.append(result)
        except Exception as e:
            print(f"  FAILED task_id={sample.get('task_id', '?')}: {e}")
            failed += 1

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(all_samples)}")

    output = {
        "metadata": {
            "dataset": "mbpp",
            "split": "full (all splits combined)",
            "total_samples": len(all_samples),
            "processed": len(results),
            "failed": failed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "samples": results,
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {OUTPUT_FILE}")
    print(f"  Processed: {len(results)}, Failed: {failed}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Split MBPP code into statement blocks")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N samples")
    args = ap.parse_args()
    main(limit=args.limit)
```

**Step 2: Validate with first 10 samples**

Run:
```bash
cd /home/monglitay/PycharmProjects/WFCLLM/experiment/statement_block_split && conda run -n WFCLLM python split.py --limit 10
```

Expected: prints progress, outputs `results/mbpp_blocks.json` with 10 samples.

**Step 3: Inspect output**

Run:
```bash
cd /home/monglitay/PycharmProjects/WFCLLM/experiment/statement_block_split && conda run -n WFCLLM python -c "
import json
with open('results/mbpp_blocks.json') as f:
    data = json.load(f)
print('Metadata:', json.dumps(data['metadata'], indent=2))
print()
for s in data['samples'][:3]:
    print(f'task_id={s[\"task_id\"]}: {s[\"stats\"][\"total_blocks\"]} blocks (simple={s[\"stats\"][\"simple_blocks\"]}, compound={s[\"stats\"][\"compound_blocks\"]}, max_depth={s[\"stats\"][\"max_depth\"]})')
    for b in s['blocks'][:3]:
        print(f'  [{b[\"id\"]}] {b[\"type\"]:8s} {b[\"node_type\"]}')
    print()
"
```

Expected: metadata with 10 processed, sample block summaries printed.

**Step 4: Commit**

```bash
git add experiment/statement_block_split/split.py
git commit -m "feat: add dataset processing and CLI for block splitting"
```

---

### Task 5: Run Full Dataset

**Step 1: Run on all samples**

Run:
```bash
cd /home/monglitay/PycharmProjects/WFCLLM/experiment/statement_block_split && conda run -n WFCLLM python split.py
```

Expected: processes ~974 samples, outputs full `results/mbpp_blocks.json`.

**Step 2: Verify output completeness**

Run:
```bash
cd /home/monglitay/PycharmProjects/WFCLLM/experiment/statement_block_split && conda run -n WFCLLM python -c "
import json
with open('results/mbpp_blocks.json') as f:
    data = json.load(f)
m = data['metadata']
print(f'Total: {m[\"total_samples\"]}, Processed: {m[\"processed\"]}, Failed: {m[\"failed\"]}')
total_blocks = sum(s['stats']['total_blocks'] for s in data['samples'])
print(f'Total blocks across all samples: {total_blocks}')
"
```

Expected: ~974 total, most processed, few or zero failed.

**Step 3: Commit results**

```bash
git add experiment/statement_block_split/results/mbpp_blocks.json
git commit -m "data: add MBPP statement block split results"
```
