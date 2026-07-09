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
            # Recurse into non-statement nodes (e.g., 'block', 'elif_clause')
            # to find nested statements, keeping same depth and parent.
            _extract_recursive(child, blocks, depth, parent_id)
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
