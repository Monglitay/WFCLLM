#!/usr/bin/env python
"""Apply the single preregistered V2 development repair to saved attempts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.pipeline import (  # noqa: E402
    load_jsonl_records,
    validate_final_code_detector_input_record,
)
from wfcllm.detection.signature_v2 import (  # noqa: E402
    STANDARDIZED_BIT_SUM,
    standardized_bit_sum,
)
from wfcllm.generation.selection_v2 import (  # noqa: E402
    V2_RETRY_LEDGER_SCHEMA_VERSION,
)

REPAIR_SELECTION_SCHEMA_VERSION = "wfcllm-v2-repair-selection/v1"
RETRY_ATTEMPTS = 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reselect saved V2 retry-20 attempts with standardized bit evidence.",
    )
    parser.add_argument("--base-final-code", required=True)
    parser.add_argument("--retry-ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path) -> str:
    return _sha256_bytes(Path(path).read_bytes())


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"retry ledger field {field} must be an integer")
    return value


def _load_ledger(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"retry ledger row {line_number} must be an object")
                if row.get("schema_version") != V2_RETRY_LEDGER_SCHEMA_VERSION:
                    raise ValueError(f"retry ledger row {line_number} schema mismatch")
                if row.get("audit_only") is not True:
                    raise ValueError(f"retry ledger row {line_number} must be audit_only")
                if row.get("detector_input_allowed") is not False:
                    raise ValueError(
                        f"retry ledger row {line_number} must forbid detector input"
                    )
                final_code = row.get("final_code")
                if not isinstance(final_code, str):
                    raise ValueError(f"retry ledger row {line_number} final_code invalid")
                if row.get("final_code_sha256") != _sha256_bytes(final_code.encode("utf-8")):
                    raise ValueError(f"retry ledger row {line_number} code hash mismatch")
                rows.append(row)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid retry ledger JSONL at line {exc.lineno}") from exc
    return rows


def _repair_score(row: dict[str, Any]) -> float:
    return standardized_bit_sum(
        matched_bits=_required_int(row, "matched_signature_bits"),
        total_bits=_required_int(row, "total_signature_bits"),
    )


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("quality")
    if not isinstance(value, dict):
        raise ValueError("retry ledger quality must be an object")
    return value


def _eligible_key(row: dict[str, Any]) -> tuple[float, int, int, int]:
    return (
        _repair_score(row),
        _required_int(row, "unit_count"),
        -_required_int(row, "v1_fallback_count"),
        -_required_int(row, "attempt_index"),
    )


def _fallback_key(row: dict[str, Any]) -> tuple[int, int, int]:
    quality_tier = _quality(row).get("quality_tier")
    if isinstance(quality_tier, bool) or not isinstance(quality_tier, int):
        raise ValueError("retry ledger quality_tier must be an integer")
    return (
        quality_tier,
        -_required_int(row, "v1_fallback_count"),
        -_required_int(row, "attempt_index"),
    )


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _build(args: argparse.Namespace) -> dict[str, Any]:
    base_rows = load_jsonl_records(args.base_final_code)
    base_by_id: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        validate_final_code_detector_input_record(row)
        sample_id = str(row["id"])
        if sample_id in base_by_id:
            raise ValueError(f"duplicate base id: {sample_id}")
        base_by_id[sample_id] = row

    ledger_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_ledger(args.retry_ledger):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("retry ledger row must contain an id")
        ledger_by_id[sample_id].append(row)
    if set(ledger_by_id) != set(base_by_id):
        raise ValueError("retry ledger ids must exactly match base final-code ids")

    expected_indices = set(range(RETRY_ATTEMPTS))
    output_rows: list[dict[str, str]] = []
    selections: list[dict[str, Any]] = []
    for base_row in base_rows:
        sample_id = str(base_row["id"])
        attempts = ledger_by_id[sample_id]
        indices = [_required_int(row, "attempt_index") for row in attempts]
        if len(attempts) != RETRY_ATTEMPTS or set(indices) != expected_indices:
            raise ValueError(f"{sample_id} must contain attempt indices 0..19")
        if len(set(indices)) != RETRY_ATTEMPTS:
            raise ValueError(f"{sample_id} contains duplicate attempt indices")
        eligible = [row for row in attempts if _quality(row).get("eligible") is True]
        no_embedding = not eligible
        selected = max(eligible, key=_eligible_key) if eligible else max(
            attempts,
            key=_fallback_key,
        )
        final_code = str(selected["final_code"])
        output_rows.append(
            {
                "id": sample_id,
                "dataset": str(base_row["dataset"]),
                "prompt": str(base_row["prompt"]),
                "final_code": final_code,
            }
        )
        selections.append(
            {
                "id": sample_id,
                "attempt_index": _required_int(selected, "attempt_index"),
                "seed": _required_int(selected, "seed"),
                "generation_score": _repair_score(selected),
                "unit_count": _required_int(selected, "unit_count"),
                "total_signature_bits": _required_int(
                    selected,
                    "total_signature_bits",
                ),
                "matched_signature_bits": _required_int(
                    selected,
                    "matched_signature_bits",
                ),
                "no_embedding": no_embedding,
                "same_attempt_as_original_v2": selected.get("selected") is True,
                "final_code_sha256": _sha256_bytes(final_code.encode("utf-8")),
            }
        )

    output_path = Path(args.output)
    _write_jsonl(output_path, output_rows)
    return {
        "artifact_type": "wfcllm_v2_repair_selection",
        "schema_version": REPAIR_SELECTION_SCHEMA_VERSION,
        "development_iteration": True,
        "aggregation": STANDARDIZED_BIT_SUM,
        "retry": RETRY_ATTEMPTS,
        "detector_input_is_strict_final_code": True,
        "source_ledger_is_audit_only": True,
        "sample_count": len(output_rows),
        "inputs": {
            "base_final_code": str(Path(args.base_final_code)),
            "base_final_code_sha256": _sha256_file(args.base_final_code),
            "retry_ledger": str(Path(args.retry_ledger)),
            "retry_ledger_sha256": _sha256_file(args.retry_ledger),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256_file(output_path),
        },
        "selections": selections,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = _build(args)
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] repaired V2 final code saved to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
