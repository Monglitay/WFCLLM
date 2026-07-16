from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from wfcllm.method.artifacts import FinalCodeRecord, write_jsonl_rows

FINAL_CODE_FIELDS = frozenset({"id", "dataset", "prompt", "final_code"})


def write_final_code_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> str:
    final_rows: list[dict[str, str]] = []
    for row in rows:
        if set(row) != FINAL_CODE_FIELDS:
            raise ValueError("final-code rows must contain exactly id,dataset,prompt,final_code")
        final_rows.append(
            FinalCodeRecord(
                id=row["id"],
                dataset=row["dataset"],
                prompt=row["prompt"],
                final_code=row["final_code"],
            ).to_dict()
        )
    return write_jsonl_rows(path, final_rows)


def write_generation_sidecar_rows(
    path: str | Path,
    rows: Iterable[dict[str, Any]],
) -> str:
    """Write generation-only rows while making detector exclusion explicit."""

    sidecars: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["audit_only"] = True
        value["not_detector_input"] = True
        if set(value) == FINAL_CODE_FIELDS:
            raise ValueError("generation sidecars cannot use the detector input schema")
        sidecars.append(value)
    return write_jsonl_rows(path, sidecars)


def write_generation_manifest(path: str | Path, manifest: dict[str, Any]) -> str:
    """Write the public generation binding as canonical, secret-free JSON."""

    if "secret" in manifest:
        raise ValueError("generation manifest cannot contain secret material")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = __import__("json").dumps(
            manifest,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("generation manifest must be JSON-safe") from exc
    output.write_text(payload + "\n", encoding="utf-8")
    return str(output)
