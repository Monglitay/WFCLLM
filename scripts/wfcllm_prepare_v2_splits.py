#!/usr/bin/env python
"""Materialize preregistered V2 negative panels as strict final-code rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PREREGISTRATION_ARTIFACT_TYPE = "wfcllm_v2_preregistration"
PREREGISTRATION_SCHEMA_VERSION = "wfcllm-v2-preregistration/v1"
SPLIT_MANIFEST_SCHEMA_VERSION = "wfcllm-v2-negative-splits/v1"
_SOURCE_FIELDS = {"id", "dataset", "prompt", "generated_code"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze preregistered V2 negative split JSONL artifacts.",
    )
    parser.add_argument("--preregistration", required=True)
    parser.add_argument("--negative-source", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("preregistration must be a JSON object")
    return value


def _load_source(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict) or set(value) != _SOURCE_FIELDS:
                    raise ValueError(
                        f"negative source row {line_number} must contain exactly "
                        f"{sorted(_SOURCE_FIELDS)}"
                    )
                if any(not isinstance(value[field], str) for field in _SOURCE_FIELDS):
                    raise ValueError(
                        f"negative source row {line_number} fields must be strings"
                    )
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {exc.lineno}") from exc
    return rows


def _ids(value: Any, *, field: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{field} must be a non-empty list of unique ids")
    return tuple(value)


def _validate_preregistration(
    preregistration: dict[str, Any],
) -> dict[str, tuple[str, ...]]:
    if preregistration.get("artifact_type") != PREREGISTRATION_ARTIFACT_TYPE:
        raise ValueError("preregistration artifact_type mismatch")
    if preregistration.get("schema_version") != PREREGISTRATION_SCHEMA_VERSION:
        raise ValueError("preregistration schema_version mismatch")
    try:
        negative_split = preregistration["negative_split"]
        pilot_selection = preregistration["pilot_task_selection"]
        if not isinstance(negative_split, dict) or not isinstance(
            pilot_selection, dict
        ):
            raise ValueError("preregistration split sections must be objects")
        split_ids = {
            "development": _ids(
                negative_split["pilot_development_ids"],
                field="pilot_development_ids",
            ),
            "calibration": _ids(
                negative_split["calibration_ids"],
                field="calibration_ids",
            ),
            "heldout": _ids(
                negative_split["heldout_ids"],
                field="heldout_ids",
            ),
            "pilot": _ids(pilot_selection["ids"], field="pilot_task_selection.ids"),
        }
    except KeyError as exc:
        raise ValueError(f"missing preregistration field: {exc.args[0]}") from exc

    named_partition = {
        name: set(split_ids[name])
        for name in ("development", "calibration", "heldout")
    }
    for left, right in (
        ("development", "calibration"),
        ("development", "heldout"),
        ("calibration", "heldout"),
    ):
        overlap = named_partition[left] & named_partition[right]
        if overlap:
            raise ValueError(
                f"preregistered split overlap between {left} and {right}: "
                f"{sorted(overlap)}"
            )
    if not set(split_ids["pilot"]) <= named_partition["development"]:
        raise ValueError("pilot task ids must be a subset of development ids")
    return split_ids


def _strict_row(source: dict[str, str]) -> dict[str, str]:
    return {
        "id": source["id"],
        "dataset": source["dataset"],
        "prompt": source["prompt"],
        "final_code": source["prompt"] + source["generated_code"],
    }


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _materialize(args: argparse.Namespace) -> dict[str, Any]:
    preregistration = _load_json(args.preregistration)
    split_ids = _validate_preregistration(preregistration)
    source_rows = _load_source(args.negative_source)
    by_id: dict[str, dict[str, str]] = {}
    for row in source_rows:
        sample_id = row["id"]
        if sample_id in by_id:
            raise ValueError(f"duplicate negative source id: {sample_id}")
        by_id[sample_id] = row
    partition_ids = (
        set(split_ids["development"])
        | set(split_ids["calibration"])
        | set(split_ids["heldout"])
    )
    if set(by_id) != partition_ids:
        missing = sorted(partition_ids - set(by_id))
        unexpected = sorted(set(by_id) - partition_ids)
        raise ValueError(
            f"negative source ids do not match preregistration; "
            f"missing={missing}, unexpected={unexpected}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, dict[str, Any]] = {}
    for name in ("pilot", "development", "calibration", "heldout"):
        rows = [_strict_row(by_id[sample_id]) for sample_id in split_ids[name]]
        output_path = output_dir / f"{name}.jsonl"
        _write_jsonl(output_path, rows)
        split_manifest[name] = {
            "path": str(output_path),
            "count": len(rows),
            "sha256": _file_sha256(output_path),
            "ids": list(split_ids[name]),
        }
    return {
        "artifact_type": "wfcllm_v2_negative_split_manifest",
        "schema_version": SPLIT_MANIFEST_SCHEMA_VERSION,
        "seed": preregistration.get("seed"),
        "preregistration": {
            "path": str(Path(args.preregistration)),
            "sha256": _file_sha256(args.preregistration),
        },
        "source": {
            "path": str(Path(args.negative_source)),
            "sha256": _file_sha256(args.negative_source),
            "count": len(source_rows),
        },
        "splits": split_manifest,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        manifest = _materialize(args)
        manifest_path = Path(args.output_dir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] frozen negative splits saved to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
