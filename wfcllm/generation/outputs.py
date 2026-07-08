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
