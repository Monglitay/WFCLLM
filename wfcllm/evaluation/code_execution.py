from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"evaluation input does not exist: {source}")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid evaluation JSONL at {source}:{line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(
                    f"evaluation row at {source}:{line_number} must be an object"
                )
            records.append(value)
    if not records:
        raise ValueError(f"evaluation input is empty: {source}")
    return records


def compute_pass_at_1(records: list[dict[str, Any]]) -> float:
    """Compute task-macro Pass@1 from posthoc correctness results."""

    grouped: dict[str, list[bool]] = {}
    for record in records:
        task_id = record.get("task_id", record.get("id"))
        if not isinstance(task_id, str) or not task_id:
            raise ValueError("each evaluation row must provide task_id or id")
        if type(record.get("is_correct")) is not bool:
            raise ValueError("each evaluation row must provide boolean is_correct")
        grouped.setdefault(task_id, []).append(record["is_correct"])
    if not grouped:
        raise ValueError("posthoc correctness rows must not be empty")
    task_rates = (sum(values) / len(values) for values in grouped.values())
    return sum(task_rates) / len(grouped)
