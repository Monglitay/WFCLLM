#!/usr/bin/env python3
"""Fit the public, primary-key-independent V4 calibration null distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.batch_invariant_v4.calibration import (  # noqa: E402
    fit_calibration,
    save_calibration,
)
from wfcllm.batch_invariant_v4.channel import StructuralChannel  # noqa: E402
from wfcllm.batch_invariant_v4.config import load_public_config  # noqa: E402
from wfcllm.batch_invariant_v4.context import (  # noqa: E402
    ContextConfig,
    StructuralContextExtractor,
)
from wfcllm.batch_invariant_v4.detector import reconstruct_unthresholded  # noqa: E402
from wfcllm.batch_invariant_v4.keying import (  # noqa: E402
    public_calibration_reference_key,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--calibration-input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = load_public_config(args.public_config)
        rows = []
        with args.calibration_input.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != {
                    "id", "dataset", "prompt", "final_code"
                }:
                    raise ValueError(
                        f"calibration row {line_number} fields do not match schema"
                    )
                rows.append(row)
        if len(rows) != 230:
            raise ValueError(f"calibration must contain exactly 230 rows, got {len(rows)}")
        extractor = StructuralContextExtractor(
            ContextConfig(
                schema_version=config.canonical_context.schema_version,
                max_unit_bytes=config.canonical_context.max_unit_bytes,
                max_context_bytes=config.canonical_context.max_context_bytes,
                global_ordinal_keying=config.canonical_context.global_ordinal_keying,
            )
        )
        channel = StructuralChannel(
            public_calibration_reference_key(),
            bit_count=config.channel.bit_count_per_unit,
            minimum_independent_units=config.channel.minimum_independent_units,
        )
        evidence = [
            reconstruct_unthresholded(
                extractor=extractor,
                channel=channel,
                final_code=row["final_code"],
            )
            for row in rows
        ]
        source_sha256 = _sha256(args.calibration_input)
        artifact = fit_calibration(
            scores=(item.score for item in evidence),
            source_role="calibration_negative",
            source_sha256=source_sha256,
            target_fpr=config.raw["decision"]["target_fpr"],
            minimum_independent_units=config.channel.minimum_independent_units,
        )
        save_calibration(artifact, args.output)
        audit = {
            "artifact_type": "wfcllm_v4_calibration_reference_audit",
            "schema_version": "wfcllm-v4-calibration-reference-audit/v1",
            "source_role": "calibration_negative",
            "source_sha256": source_sha256,
            "row_count": len(rows),
            "eligible_count": sum(item.eligible for item in evidence),
            "independent_unit_histogram": dict(
                sorted(Counter(item.independent_units for item in evidence).items())
            ),
            "public_reference_domain": "v4-public/calibration-null-reference",
            "primary_private_key_used": False,
            "heldout_accessed": False,
            "secret_metadata_included": False,
        }
        args.audit_output.parent.mkdir(parents=True, exist_ok=True)
        args.audit_output.write_text(
            json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
