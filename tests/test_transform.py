"""Tests for main transform pipeline."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiment" / "statement_block_transform"))

from transform import process_blocks


def test_process_blocks_small():
    """Process a minimal input and verify output structure."""
    input_data = {
        "metadata": {"dataset": "mbpp", "total_samples": 1, "processed": 1, "failed": 0},
        "samples": [
            {
                "task_id": 1,
                "prompt": "test",
                "original_code": "print(x)",
                "blocks": [
                    {
                        "id": 0,
                        "type": "simple",
                        "node_type": "expression_statement",
                        "source": "print(x)",
                        "start_line": 1,
                        "end_line": 1,
                        "depth": 0,
                        "parent_id": None,
                        "children_ids": [],
                    }
                ],
                "stats": {"total_blocks": 1, "simple_blocks": 1, "compound_blocks": 0, "max_depth": 0},
            }
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(input_data, f)
        input_path = Path(f.name)

    output_path = input_path.with_name("output.json")
    try:
        process_blocks(input_path, output_path, max_perm_len=2, max_variants=10)
        with open(output_path) as f:
            result = json.load(f)

        assert "metadata" in result
        assert "samples" in result
        assert len(result["samples"]) == 1
        sample = result["samples"][0]
        assert sample["task_id"] == 1
        assert len(sample["blocks"]) >= 1
        block = sample["blocks"][0]
        assert "block_id" in block
        assert "original_source" in block
        assert "applicable_rules" in block
        assert "variants" in block
        # print(x) should match at least ExplicitDefaultPrint
        assert len(block["variants"]) >= 1
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
