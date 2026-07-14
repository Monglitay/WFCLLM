from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from wfcllm.batch_invariant_v4.detector import (
    CodeEvidence,
    DetectorPayload,
    V4Detector,
    exact_code_evidence_mismatches,
)
from wfcllm.batch_invariant_v4.runtime import RawCandidate


RETRY_LEDGER_SCHEMA = "wfcllm-v2-retry-ledger/v2"


def load_retry_ledgers(paths: Sequence[str | Path]) -> dict[str, tuple[dict[str, Any], ...]]:
    grouped: dict[str, dict[int, dict[str, Any]]] = {}
    for path in paths:
        source = Path(path)
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise ValueError(f"failed to read retry ledger: {source}") from exc
        for line_number, raw in enumerate(lines, start=1):
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"malformed retry ledger JSON at {source}:{line_number}") from exc
            if not isinstance(row, dict) or row.get("schema_version") != RETRY_LEDGER_SCHEMA:
                raise ValueError("retry ledger schema mismatch")
            try:
                task_id = str(row["id"])
                attempt = int(row["attempt_index"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("retry ledger identity is malformed") from exc
            attempts = grouped.setdefault(task_id, {})
            if attempt in attempts:
                raise ValueError(f"duplicate retry attempt: {task_id}/{attempt}")
            attempts[attempt] = row
    return {
        task_id: tuple(attempts[index] for index in sorted(attempts))
        for task_id, attempts in grouped.items()
    }


def rows_to_candidates(
    rows: Sequence[dict[str, Any]],
    *,
    retry: int,
) -> tuple[RawCandidate, ...]:
    if len(rows) != retry:
        raise ValueError(f"retry ledger must contain exactly {retry} rows")
    candidates: list[RawCandidate] = []
    for expected, row in enumerate(rows):
        if int(row["attempt_index"]) != expected:
            raise ValueError("retry rows must be ordered 0..retry-1")
        quality = row.get("quality")
        if not isinstance(quality, dict):
            raise ValueError("retry row quality is malformed")
        candidates.append(
            RawCandidate(
                task_id=str(row["id"]),
                attempt_index=expected,
                final_code=str(row["final_code"]),
                final_code_sha256=str(row["final_code_sha256"]),
                quality_tier=int(quality["quality_tier"]),
                valid=bool(quality["eligible"]),
                fallback_count=int(row.get("v1_fallback_count", 0)),
            )
        )
    return tuple(candidates)


def evidence_dict(evidence: CodeEvidence) -> dict[str, Any]:
    return asdict(evidence)


def evidence_sha256(evidence: CodeEvidence) -> str:
    payload = json.dumps(
        evidence_dict(evidence),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def scheduled_indices(
    candidates: Sequence[RawCandidate],
    *,
    batch_size: int,
    order: str,
) -> tuple[tuple[int, ...], ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = list(range(len(candidates)))
    if order == "reverse":
        indices.reverse()
    elif order == "permutation":
        random.Random(20260714).shuffle(indices)
    elif order == "short_first":
        indices.sort(key=lambda index: (len(candidates[index].final_code), index))
    elif order == "long_first":
        indices.sort(key=lambda index: (-len(candidates[index].final_code), index))
    elif order != "forward":
        raise ValueError(f"unknown schedule order: {order}")
    return tuple(
        tuple(indices[offset : offset + batch_size])
        for offset in range(0, len(indices), batch_size)
    )


def replay_schedule(
    detector: V4Detector,
    candidates: Sequence[RawCandidate],
    schedule: Iterable[Iterable[int]],
) -> dict[int, CodeEvidence]:
    output: dict[int, CodeEvidence] = {}
    for group in schedule:
        for index in group:
            candidate = candidates[index]
            output[candidate.attempt_index] = detector.detect(
                DetectorPayload(final_code=candidate.final_code)
            )
    if set(output) != {candidate.attempt_index for candidate in candidates}:
        raise ValueError("schedule did not cover every candidate exactly once")
    return output


def compare_evidence_maps(
    reference: dict[int, CodeEvidence],
    candidate: dict[int, CodeEvidence],
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if set(reference) != set(candidate):
        return ("attempt_keys",)
    for attempt in sorted(reference):
        for field in exact_code_evidence_mismatches(reference[attempt], candidate[attempt]):
            mismatches.append(f"attempt[{attempt}].{field}")
    return tuple(mismatches)
