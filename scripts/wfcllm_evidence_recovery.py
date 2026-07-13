#!/usr/bin/env python
"""Replay v1 accepted evidence against canonical windows in saved final code."""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.pipeline import load_jsonl_records  # noqa: E402
from wfcllm.detection.proxy_windows import extract_structure_contexts  # noqa: E402

RECOVERY_SCHEMA_VERSION = "wfcllm-evidence-recovery/v1"
ACCEPTED_EVENT = "accepted_generation_time_window"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare generation-time accepted hashes to final-code windows.",
    )
    parser.add_argument("--final-code", required=True)
    parser.add_argument("--candidate-sidecar", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-group-statements", type=int, default=2)
    return parser


def _load_sidecar(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"sidecar row {line_number} must be an object")
                if value.get("audit_only") is not True:
                    raise ValueError(f"sidecar row {line_number} must be audit_only")
                if value.get("detector_input_allowed") is not False:
                    raise ValueError(
                        f"sidecar row {line_number} must forbid detector input"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid sidecar JSONL at line {exc.lineno}") from exc
    return rows


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _accepted_hashes_by_id(
    rows: list[dict[str, Any]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("event") != ACCEPTED_EVENT:
            continue
        sample_id = row.get("id")
        normalized_hash = row.get("normalized_text_hash")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"accepted sidecar row {index} must contain an id")
        if (
            not isinstance(normalized_hash, str)
            or len(normalized_hash) != 64
            or any(character not in string.hexdigits for character in normalized_hash)
        ):
            raise ValueError(
                f"accepted sidecar row {index} must contain a SHA-256 hash"
            )
        grouped[sample_id].append(normalized_hash.lower())
    return dict(grouped)


def _final_window_hashes(
    final_code: str,
    *,
    max_group_statements: int,
) -> list[str]:
    contexts = extract_structure_contexts(
        final_code,
        prompt=None,
        max_group_statements=max_group_statements,
    )
    return [
        _hash_text(window.normalized_text)
        for context in contexts
        for window in context.proxy_windows
    ]


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_group_statements <= 0:
        raise ValueError("max_group_statements must be positive")
    final_rows = load_jsonl_records(args.final_code)
    sidecar_rows = _load_sidecar(args.candidate_sidecar)
    accepted_by_id = _accepted_hashes_by_id(sidecar_rows)
    final_ids = [str(row["id"]) for row in final_rows]
    if len(set(final_ids)) != len(final_ids):
        raise ValueError("final-code input contains duplicate ids")
    unexpected = sorted(set(accepted_by_id) - set(final_ids))
    if unexpected:
        raise ValueError(f"accepted sidecar ids missing from final code: {unexpected}")

    per_sample: list[dict[str, Any]] = []
    accepted_count = 0
    accepted_unique_count = 0
    recovered_accepted_count = 0
    recovered_accepted_unique_count = 0
    final_window_count = 0
    final_unique_window_count = 0
    for row in final_rows:
        sample_id = str(row["id"])
        accepted = accepted_by_id.get(sample_id, [])
        accepted_unique = set(accepted)
        recovered = _final_window_hashes(
            str(row["final_code"]),
            max_group_statements=args.max_group_statements,
        )
        recovered_unique = set(recovered)
        matched_count = sum(value in recovered_unique for value in accepted)
        matched_unique_count = len(accepted_unique & recovered_unique)
        item = {
            "id": sample_id,
            "accepted_count": len(accepted),
            "accepted_unique_count": len(accepted_unique),
            "duplicate_accepted_count": len(accepted) - len(accepted_unique),
            "recovered_accepted_count": matched_count,
            "recovered_accepted_unique_count": matched_unique_count,
            "accepted_to_recovered_ratio": (
                matched_count / len(accepted) if accepted else 0.0
            ),
            "accepted_unique_to_recovered_ratio": (
                matched_unique_count / len(accepted_unique)
                if accepted_unique
                else 0.0
            ),
            "final_proxy_window_count": len(recovered),
            "final_unique_proxy_window_count": len(recovered_unique),
        }
        per_sample.append(item)
        accepted_count += len(accepted)
        accepted_unique_count += len(accepted_unique)
        recovered_accepted_count += matched_count
        recovered_accepted_unique_count += matched_unique_count
        final_window_count += len(recovered)
        final_unique_window_count += len(recovered_unique)

    return {
        "artifact_type": "wfcllm_evidence_recovery",
        "schema_version": RECOVERY_SCHEMA_VERSION,
        "audit_only": True,
        "detector_input_allowed": False,
        "final_code_only_reextraction": True,
        "sample_count": len(final_rows),
        "accepted_count": accepted_count,
        "accepted_unique_count": accepted_unique_count,
        "duplicate_accepted_count": accepted_count - accepted_unique_count,
        "recovered_accepted_count": recovered_accepted_count,
        "recovered_accepted_unique_count": recovered_accepted_unique_count,
        "accepted_to_recovered_ratio": (
            recovered_accepted_count / accepted_count if accepted_count else 0.0
        ),
        "accepted_unique_to_recovered_ratio": (
            recovered_accepted_unique_count / accepted_unique_count
            if accepted_unique_count
            else 0.0
        ),
        "final_proxy_window_count": final_window_count,
        "final_unique_proxy_window_count": final_unique_window_count,
        "max_group_statements": args.max_group_statements,
        "inputs": {
            "final_code_path": str(Path(args.final_code)),
            "final_code_sha256": _file_sha256(args.final_code),
            "candidate_sidecar_path": str(Path(args.candidate_sidecar)),
            "candidate_sidecar_sha256": _file_sha256(args.candidate_sidecar),
        },
        "per_sample": per_sample,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = _build_report(args)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] evidence recovery report saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
