"""Freeze new, content-deduplicated V4 negative calibration/held-out panels."""

from __future__ import annotations

import ast
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


SPLIT_SCHEMA_VERSION = "wfcllm-v4-negative-split/v1"
_ORDER_DOMAIN = "wfcllm-v4-batch-invariant-negative-split"


def _normalized_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("negative source text fields must be strings")
    return value.strip().replace("\r\n", "\n")


def _sha256_text(value: str) -> str:
    return hashlib.sha256(_normalized_text(value).encode("utf-8")).hexdigest()


def _ast_sha256(code: str) -> str:
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, TypeError) as exc:
        raise ValueError("negative final code must parse as Python") from exc
    canonical = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class NegativeSourceRow:
    sample_id: str
    source_split: str
    prompt: str
    final_code: str

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("negative sample ID must be non-empty")
        if not isinstance(self.source_split, str) or not self.source_split:
            raise ValueError("negative source split must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("negative prompt must be non-empty")
        if not isinstance(self.final_code, str) or not self.final_code:
            raise ValueError("negative final code must be non-empty")


@dataclass(frozen=True)
class FrozenNegativeSplit:
    calibration: tuple[NegativeSourceRow, ...]
    heldout: tuple[NegativeSourceRow, ...]
    seed: int
    audit: dict[str, int]


def _domain_order(row: NegativeSourceRow, *, seed: int) -> bytes:
    return hashlib.sha256(
        (
            _ORDER_DOMAIN
            + "\0"
            + str(seed)
            + "\0"
            + row.sample_id
        ).encode("utf-8")
    ).digest()


def freeze_negative_split(
    rows: Iterable[NegativeSourceRow],
    *,
    excluded_final_codes: Sequence[str],
    humaneval_prompts: Sequence[str],
    humaneval_final_codes: Sequence[str],
    calibration_count: int,
    seed: int,
) -> FrozenNegativeSplit:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("split seed must be a non-negative integer")
    if (
        isinstance(calibration_count, bool)
        or not isinstance(calibration_count, int)
        or calibration_count <= 0
    ):
        raise ValueError("calibration_count must be a positive integer")
    source = tuple(rows)
    if not source:
        raise ValueError("negative source must not be empty")
    ids = [row.sample_id for row in source]
    if len(set(ids)) != len(ids):
        raise ValueError("negative source IDs must be unique")

    sorted_rows = sorted(source, key=lambda row: row.sample_id)
    unique_by_ast: dict[str, NegativeSourceRow] = {}
    ast_duplicate_rows_removed = 0
    for row in sorted_rows:
        ast_sha256 = _ast_sha256(row.final_code)
        if ast_sha256 in unique_by_ast:
            ast_duplicate_rows_removed += 1
            continue
        unique_by_ast[ast_sha256] = row

    excluded_code_hashes = {_sha256_text(code) for code in excluded_final_codes}
    excluded_ast_hashes = {_ast_sha256(code) for code in excluded_final_codes}
    humaneval_prompt_hashes = {_sha256_text(prompt) for prompt in humaneval_prompts}
    humaneval_code_hashes = {_sha256_text(code) for code in humaneval_final_codes}
    humaneval_ast_hashes = {_ast_sha256(code) for code in humaneval_final_codes}

    eligible: list[NegativeSourceRow] = []
    excluded_source_overlap = 0
    humaneval_prompt_overlap = 0
    humaneval_final_overlap = 0
    for ast_sha256, row in unique_by_ast.items():
        code_sha256 = _sha256_text(row.final_code)
        if code_sha256 in excluded_code_hashes or ast_sha256 in excluded_ast_hashes:
            excluded_source_overlap += 1
            continue
        if _sha256_text(row.prompt) in humaneval_prompt_hashes:
            humaneval_prompt_overlap += 1
            continue
        if code_sha256 in humaneval_code_hashes or ast_sha256 in humaneval_ast_hashes:
            humaneval_final_overlap += 1
            continue
        eligible.append(row)

    ordered = tuple(sorted(eligible, key=lambda row: _domain_order(row, seed=seed)))
    if calibration_count >= len(ordered):
        raise ValueError("negative source is insufficient for non-empty held-out split")
    audit = {
        "source_row_count": len(source),
        "source_unique_id_count": len(set(ids)),
        "ast_duplicate_rows_removed": ast_duplicate_rows_removed,
        "excluded_source_overlap_rows_removed": excluded_source_overlap,
        "humaneval_prompt_overlap_rows_removed": humaneval_prompt_overlap,
        "humaneval_final_overlap_rows_removed": humaneval_final_overlap,
        "eligible_unique_row_count": len(ordered),
    }
    return FrozenNegativeSplit(
        calibration=ordered[:calibration_count],
        heldout=ordered[calibration_count:],
        seed=seed,
        audit=audit,
    )


def _strict_row(row: NegativeSourceRow) -> dict[str, str]:
    return {
        "id": row.sample_id,
        "dataset": "mbpp",
        "prompt": row.prompt,
        "final_code": row.final_code,
    }


def _canonical_json_line(value: dict[str, str]) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"


def _write_panel(path: Path, rows: Sequence[NegativeSourceRow]) -> dict[str, Any]:
    serialized_rows = tuple(_canonical_json_line(_strict_row(row)) for row in rows)
    path.write_text("".join(serialized_rows), encoding="utf-8", newline="\n")
    return {
        "count": len(rows),
        "ids": [row.sample_id for row in rows],
        "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "record_sha256": [
            hashlib.sha256(serialized.encode("utf-8")).hexdigest()
            for serialized in serialized_rows
        ],
    }


def write_frozen_negative_split(
    frozen: FrozenNegativeSplit,
    output_dir: str | Path,
) -> dict[str, Any]:
    if not isinstance(frozen, FrozenNegativeSplit):
        raise ValueError("frozen split must be FrozenNegativeSplit")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    calibration = _write_panel(directory / "calibration.jsonl", frozen.calibration)
    heldout = _write_panel(directory / "heldout.jsonl", frozen.heldout)
    manifest: dict[str, Any] = {
        "artifact_type": "wfcllm_v4_negative_split_manifest",
        "schema_version": SPLIT_SCHEMA_VERSION,
        "seed": frozen.seed,
        "domain_separator": _ORDER_DOMAIN,
        "heldout_access_policy": "forbidden_before_all_pilot_gates",
        "secret_metadata_included": False,
        "audit": frozen.audit,
        "calibration": calibration,
        "heldout": heldout,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest
