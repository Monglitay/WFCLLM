from __future__ import annotations

import json
from pathlib import Path

from wfcllm.diagnostics.split_v4 import (
    NegativeSourceRow,
    freeze_negative_split,
    write_frozen_negative_split,
)


def _row(index: int, code: str) -> NegativeSourceRow:
    return NegativeSourceRow(
        sample_id=f"MBPP/{600 + index}",
        source_split="train",
        prompt=f"problem {index}",
        final_code=code,
    )


def test_freeze_negative_split_deduplicates_and_excludes_overlap() -> None:
    rows = (
        _row(1, "def a():\n    return 1\n"),
        _row(2, "def a( ): # same AST\n return 1\n"),
        _row(3, "def b():\n    return 2\n"),
        _row(4, "def c():\n    return 3\n"),
        _row(5, "def d():\n    return 4\n"),
        _row(6, "def excluded():\n    return 5\n"),
        _row(7, "def humaneval_overlap():\n    return 6\n"),
    )

    frozen = freeze_negative_split(
        rows,
        excluded_final_codes=("def excluded():\n    return 5\n",),
        humaneval_prompts=("def unrelated_prompt():\n",),
        humaneval_final_codes=("def humaneval_overlap():\n    return 6\n",),
        calibration_count=2,
        seed=20260714,
    )

    assert len(frozen.calibration) == 2
    assert len(frozen.heldout) == 2
    assert set(row.sample_id for row in frozen.calibration).isdisjoint(
        row.sample_id for row in frozen.heldout
    )
    assert frozen.audit["source_row_count"] == 7
    assert frozen.audit["ast_duplicate_rows_removed"] == 1
    assert frozen.audit["excluded_source_overlap_rows_removed"] == 1
    assert frozen.audit["humaneval_final_overlap_rows_removed"] == 1
    assert frozen.audit["humaneval_prompt_overlap_rows_removed"] == 0
    assert frozen.audit["eligible_unique_row_count"] == 4


def test_freeze_negative_split_is_domain_deterministic_and_writer_hashes_files(
    tmp_path: Path,
) -> None:
    rows = tuple(
        _row(index, f"def f_{index}():\n    return {index}\n")
        for index in range(1, 7)
    )
    left = freeze_negative_split(
        rows,
        excluded_final_codes=(),
        humaneval_prompts=(),
        humaneval_final_codes=(),
        calibration_count=3,
        seed=20260714,
    )
    right = freeze_negative_split(
        tuple(reversed(rows)),
        excluded_final_codes=(),
        humaneval_prompts=(),
        humaneval_final_codes=(),
        calibration_count=3,
        seed=20260714,
    )

    assert left == right
    manifest = write_frozen_negative_split(left, tmp_path / "splits")

    assert manifest["schema_version"] == "wfcllm-v4-negative-split/v1"
    assert manifest["heldout_access_policy"] == "forbidden_before_all_pilot_gates"
    assert manifest["secret_metadata_included"] is False
    for name in ("calibration", "heldout"):
        path = tmp_path / "splits" / f"{name}.jsonl"
        parsed = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(parsed) == 3
        assert set(parsed[0]) == {"id", "dataset", "prompt", "final_code"}
        assert len(manifest[name]["jsonl_sha256"]) == 64
        assert manifest[name]["count"] == 3
