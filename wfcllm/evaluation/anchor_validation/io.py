"""UTF-8 JSONL helpers for anchor validation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from wfcllm.evaluation.anchor_validation.schema import (
    CandidateBlock,
    CandidateContext,
    dataclass_to_jsonable,
)

_REQUIRED_CONTEXT_FIELDS = (
    "context_id",
    "dataset",
    "task_id",
    "prompt",
    "function_signature",
    "ast_path",
    "node_type",
    "parent_node_type",
    "block_ordinal",
    "context_hash",
    "temperature",
    "candidates",
)

_REQUIRED_CANDIDATE_FIELDS = ("candidate_id", "block_text", "rank")


def write_jsonl(path: Path, rows: Iterable[object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            payload = row if isinstance(row, dict) else dataclass_to_jsonable(row)
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_candidate_contexts(path: Path, contexts: Iterable[CandidateContext]) -> Path:
    return write_jsonl(path, contexts)


def load_candidate_contexts(path: Path) -> list[CandidateContext]:
    contexts: list[CandidateContext] = []
    for payload in read_jsonl(path):
        _require_fields(payload, _REQUIRED_CONTEXT_FIELDS, "candidate context")
        candidates_payload = payload["candidates"]
        if not isinstance(candidates_payload, list):
            raise ValueError("candidate context candidates must be a list")
        candidates = tuple(
            _load_candidate_block(item, idx)
            for idx, item in enumerate(candidates_payload)
        )
        contexts.append(
            CandidateContext(
                context_id=str(payload["context_id"]),
                dataset=str(payload["dataset"]),
                task_id=str(payload["task_id"]),
                prompt=str(payload["prompt"]),
                function_signature=str(payload["function_signature"]),
                ast_path=tuple(str(part) for part in payload.get("ast_path", [])),
                node_type=str(payload["node_type"]),
                parent_node_type=str(payload["parent_node_type"]),
                block_ordinal=int(payload["block_ordinal"]),
                context_hash=str(payload["context_hash"]),
                context_before=str(payload.get("context_before", "")),
                context_after=str(payload.get("context_after", "")),
                masked_parent_context=str(payload.get("masked_parent_context", "")),
                import_and_helper_signatures=tuple(
                    str(part)
                    for part in payload.get("import_and_helper_signatures", [])
                ),
                temperature=(
                    float(payload["temperature"])
                    if payload.get("temperature") is not None
                    else None
                ),
                candidates=candidates,
            )
        )
    return contexts


def _require_fields(
    payload: dict,
    required_fields: tuple[str, ...],
    label: str,
) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{label} missing required fields: {missing}")


def _load_candidate_block(payload: object, idx: int) -> CandidateBlock:
    if not isinstance(payload, dict):
        raise ValueError(f"candidate {idx} must be an object")
    _require_fields(payload, _REQUIRED_CANDIDATE_FIELDS, f"candidate {idx}")
    return CandidateBlock(
        candidate_id=str(payload["candidate_id"]),
        block_text=str(payload["block_text"]),
        rank=_load_int(payload["rank"], f"candidate {idx} rank"),
        syntax_valid=_load_bool(payload.get("syntax_valid", True), f"candidate {idx} syntax_valid"),
        parse_valid=_load_bool(payload.get("parse_valid", True), f"candidate {idx} parse_valid"),
        quality=_load_quality(payload.get("quality", {}), idx),
    )


def _load_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _load_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _load_quality(value: object, idx: int) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"candidate {idx} quality must be an object")
    return dict(value)
