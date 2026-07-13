#!/usr/bin/env python
"""Select the preregistered V2-with-v1-selection detector ablation."""

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
from wfcllm.generation.selection_v2 import (  # noqa: E402
    V2_RETRY_LEDGER_SCHEMA_VERSION,
)

ABLATION_SELECTION_SCHEMA_VERSION = "wfcllm-v2-ablation-selection/v1"
RETRY_ATTEMPTS = 20


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the sole preregistered V2 detector selection ablation.",
    )
    parser.add_argument("--v2-final-code", required=True)
    parser.add_argument("--retry-ledger", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    return parser


def _load_ledger(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"retry ledger row {line_number} must be an object")
                if value.get("schema_version") != V2_RETRY_LEDGER_SCHEMA_VERSION:
                    raise ValueError(f"retry ledger row {line_number} schema mismatch")
                if value.get("audit_only") is not True:
                    raise ValueError(f"retry ledger row {line_number} must be audit_only")
                if value.get("detector_input_allowed") is not False:
                    raise ValueError(
                        f"retry ledger row {line_number} must forbid detector input"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid retry ledger JSONL at line {exc.lineno}") from exc
    return rows


def _required_int(row: dict[str, Any], field: str) -> int:
    value = row.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"retry ledger field {field} must be an integer")
    return value


def _selection_key(row: dict[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        _required_int(row, "v1_accepted_hit_count"),
        -_required_int(row, "v1_closed_without_hit_count"),
        -_required_int(row, "v1_fallback_count"),
        _required_int(row, "v1_candidate_count"),
        -_required_int(row, "attempt_index"),
    )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _build_ablation(args: argparse.Namespace) -> dict[str, Any]:
    base_rows = load_jsonl_records(args.v2_final_code)
    base_by_id: dict[str, dict[str, Any]] = {}
    for row in base_rows:
        validate_final_code_detector_input_record(row)
        sample_id = str(row["id"])
        if sample_id in base_by_id:
            raise ValueError(f"duplicate V2 final-code id: {sample_id}")
        base_by_id[sample_id] = row

    ledger_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _load_ledger(args.retry_ledger):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("retry ledger row must contain an id")
        final_code = row.get("final_code")
        if not isinstance(final_code, str):
            raise ValueError("retry ledger row final_code must be a string")
        ledger_by_id[sample_id].append(row)
    if set(ledger_by_id) != set(base_by_id):
        missing = sorted(set(base_by_id) - set(ledger_by_id))
        unexpected = sorted(set(ledger_by_id) - set(base_by_id))
        raise ValueError(
            f"retry ledger ids do not match V2 final code; "
            f"missing={missing}, unexpected={unexpected}"
        )

    output_rows: list[dict[str, str]] = []
    selections: list[dict[str, Any]] = []
    expected_indices = set(range(RETRY_ATTEMPTS))
    for base_row in base_rows:
        sample_id = str(base_row["id"])
        attempts = ledger_by_id[sample_id]
        attempt_indices = [_required_int(row, "attempt_index") for row in attempts]
        if (
            len(attempts) != RETRY_ATTEMPTS
            or len(set(attempt_indices)) != RETRY_ATTEMPTS
            or set(attempt_indices) != expected_indices
        ):
            raise ValueError(
                f"{sample_id} retry ledger must contain exactly attempt indices 0..19"
            )
        selected = max(attempts, key=_selection_key)
        selected_index = _required_int(selected, "attempt_index")
        output_rows.append(
            {
                "id": sample_id,
                "dataset": str(base_row["dataset"]),
                "prompt": str(base_row["prompt"]),
                "final_code": str(selected["final_code"]),
            }
        )
        selections.append(
            {
                "id": sample_id,
                "attempt_index": selected_index,
                "seed": _required_int(selected, "seed"),
                "selection_key": list(_selection_key(selected)),
                "same_attempt_as_v2": selected.get("selected") is True,
            }
        )

    output_path = Path(args.output)
    _write_jsonl(output_path, output_rows)
    return {
        "artifact_type": "wfcllm_v2_ablation_selection",
        "schema_version": ABLATION_SELECTION_SCHEMA_VERSION,
        "audit_only_source": True,
        "detector_input_is_strict_final_code": True,
        "selection_rule": "v1_evidence_retry_key",
        "detector": "aligned_canonical_signature_v2",
        "retry": RETRY_ATTEMPTS,
        "sample_count": len(output_rows),
        "inputs": {
            "v2_final_code": str(Path(args.v2_final_code)),
            "v2_final_code_sha256": _sha256(args.v2_final_code),
            "retry_ledger": str(Path(args.retry_ledger)),
            "retry_ledger_sha256": _sha256(args.retry_ledger),
        },
        "output": {
            "path": str(output_path),
            "sha256": _sha256(output_path),
        },
        "selections": selections,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = _build_ablation(args)
        manifest_path = Path(args.manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] V2 ablation final code saved to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
