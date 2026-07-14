#!/usr/bin/env python3
"""Freeze disjoint MBPP train/validation panels for V4 calibration and held-out."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.split_v4 import (  # noqa: E402
    NegativeSourceRow,
    freeze_negative_split,
    write_frozen_negative_split,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mbpp-dir", type=Path, required=True)
    parser.add_argument("--humaneval-arrow", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--calibration-count", type=int, default=230)
    parser.add_argument("--expected-eligible-count", type=int, default=459)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arrow_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.ipc as ipc

        with path.open("rb") as handle:
            table = ipc.open_stream(handle).read_all()
    except (OSError, ValueError) as exc:
        raise ValueError(f"failed to read Arrow source: {path}") from exc
    return table.to_pylist()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        train_path = args.mbpp_dir / "mbpp-train.arrow"
        validation_path = args.mbpp_dir / "mbpp-validation.arrow"
        test_path = args.mbpp_dir / "mbpp-test.arrow"
        train = _arrow_rows(train_path)
        validation = _arrow_rows(validation_path)
        test = _arrow_rows(test_path)
        humaneval = _arrow_rows(args.humaneval_arrow)
        source = tuple(
            NegativeSourceRow(
                sample_id=f"MBPP/{row['task_id']}",
                source_split=split,
                prompt=row["text"],
                final_code=row["code"],
            )
            for split, rows in (("train", train), ("validation", validation))
            for row in rows
        )
        frozen = freeze_negative_split(
            source,
            excluded_final_codes=tuple(row["code"] for row in test),
            humaneval_prompts=tuple(row["prompt"] for row in humaneval),
            humaneval_final_codes=tuple(
                row["prompt"] + row["canonical_solution"] for row in humaneval
            ),
            calibration_count=args.calibration_count,
            seed=args.seed,
        )
        eligible_count = frozen.audit["eligible_unique_row_count"]
        if eligible_count != args.expected_eligible_count:
            raise ValueError(
                f"eligible row count mismatch: expected "
                f"{args.expected_eligible_count}, observed {eligible_count}"
            )
        manifest = write_frozen_negative_split(frozen, args.output_dir)
        manifest["source"] = {
            "dataset": "MBPP",
            "included_splits": ["train", "validation"],
            "excluded_v3_source_split": "test",
            "train_count": len(train),
            "validation_count": len(validation),
            "test_exclusion_count": len(test),
            "arrow_sha256": {
                "train": _sha256_file(train_path),
                "validation": _sha256_file(validation_path),
                "test": _sha256_file(test_path),
            },
        }
        manifest["humaneval_overlap_reference"] = {
            "count": len(humaneval),
            "arrow_sha256": _sha256_file(args.humaneval_arrow),
        }
        manifest["leakage_scope"] = {
            "id_overlap_checked": True,
            "normalized_code_sha256_overlap_checked": True,
            "canonical_ast_overlap_checked": True,
            "humaneval_prompt_overlap_checked": True,
            "humaneval_final_code_overlap_checked": True,
            "encoder_training_overlap": "unknown_not_provably_excluded",
            "known_risk": (
                "CodeT5 pretraining and the historical semantic-encoder training "
                "corpus cannot be exhaustively reconstructed from local artifacts"
            ),
        }
        manifest["heldout_creation_access"] = {
            "created_and_hashed_during_split_freeze": True,
            "detector_accessed": False,
            "post_freeze_access_policy": "forbidden_before_all_pilot_gates",
        }
        args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
        args.manifest_output.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    print(
        f"[complete] calibration={len(frozen.calibration)} "
        f"heldout={len(frozen.heldout)}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
