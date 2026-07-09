"""Main transform pipeline: load blocks, apply rules, save variants."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from engine import TransformEngine
from rules import get_all_rules


def process_blocks(
    input_path: Path,
    output_path: Path,
    max_perm_len: int = 5,
    max_variants: int = 1000,
    mode: str = "positive",
) -> None:
    """Load blocks JSON, transform each block, write output JSON."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    from rules.negative import get_all_negative_rules

    if mode == "negative":
        rules = get_all_negative_rules()
    else:
        rules = get_all_rules()

    engine = TransformEngine(
        rules=rules,
        max_perm_len=max_perm_len,
        max_variants=max_variants,
        mode=mode,
    )

    total_blocks = 0
    transformed_blocks = 0
    total_variants = 0

    output_samples = []
    samples = data["samples"]

    for i, sample in enumerate(samples):
        output_blocks = []
        for block in sample["blocks"]:
            total_blocks += 1
            source = block["source"]

            applicable = engine.get_applicable_rules(source)
            variants = engine.generate_variants(source) if applicable else []

            if variants:
                transformed_blocks += 1
                total_variants += len(variants)

            output_blocks.append({
                "block_id": block["id"],
                "original_source": source,
                "block_type": block["type"],
                "node_type": block["node_type"],
                "applicable_rules": [r.name for r in applicable],
                "variants": variants,
            })

        output_samples.append({
            "task_id": sample["task_id"],
            "blocks": output_blocks,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(samples)} samples")

    output = {
        "metadata": {
            "source_file": str(input_path.name),
            "mode": mode,
            "total_blocks": total_blocks,
            "transformed_blocks": transformed_blocks,
            "total_variants": total_variants,
            "max_permutation_length": max_perm_len,
            "max_variants_per_block": max_variants,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "samples": output_samples,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Done. Output: {output_path}")
    print(f"  Total blocks: {total_blocks}")
    print(f"  Transformed: {transformed_blocks}")
    print(f"  Total variants: {total_variants}")


def main(limit: int | None = None, max_perm_len: int = 5, max_variants: int = 1000, mode: str = "positive") -> None:
    """Entry point with CLI args."""
    from config import INPUT_FILE, OUTPUT_FILE, OUTPUT_FILE_NEGATIVE

    output = OUTPUT_FILE_NEGATIVE if mode == "negative" else OUTPUT_FILE

    if limit is not None:
        # Load, truncate, save to temp, process
        with open(INPUT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        data["samples"] = data["samples"][:limit]

        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(data, tmp, ensure_ascii=False)
            tmp_path = Path(tmp.name)
        process_blocks(tmp_path, output, max_perm_len, max_variants, mode)
        tmp_path.unlink()
    else:
        process_blocks(INPUT_FILE, output, max_perm_len, max_variants, mode)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Transform statement blocks with semantic-equivalent rules")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N samples")
    ap.add_argument("--max-perm-len", type=int, default=5, help="Max permutation length")
    ap.add_argument("--max-variants", type=int, default=1000, help="Max variants per block")
    ap.add_argument("--mode", choices=["positive", "negative"], default="positive",
                    help="Transform mode: positive (semantic-equivalent) or negative (semantic-breaking)")
    args = ap.parse_args()
    main(limit=args.limit, max_perm_len=args.max_perm_len, max_variants=args.max_variants, mode=args.mode)
