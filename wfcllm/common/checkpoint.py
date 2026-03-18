"""Utilities for sample-level checkpoint recovery."""
from __future__ import annotations

import json
import logging
from pathlib import Path


def find_latest_jsonl(directory: Path, pattern: str = "*.jsonl") -> Path | None:
    """Return the most recently modified JSONL file matching pattern."""
    if not directory.exists():
        return None

    matches = [path for path in directory.glob(pattern) if path.is_file()]
    if not matches:
        return None

    latest = max(matches, key=lambda path: path.stat().st_mtime)
    if latest.stat().st_size == 0:
        logging.warning("Resume file is empty: %s", latest)
    return latest


def load_processed_ids(jsonl_path: Path) -> set[str]:
    """Load processed sample IDs from a JSONL file."""
    processed_ids: set[str] = set()
    non_empty_lines = 0
    malformed_lines = 0

    with open(jsonl_path, encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            non_empty_lines += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines += 1
                logging.warning(
                    "Skip malformed JSONL line %s in %s", line_no, jsonl_path
                )
                continue

            sample_id = payload.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                malformed_lines += 1
                logging.warning(
                    "Skip JSONL line without valid id at %s:%s",
                    jsonl_path,
                    line_no,
                )
                continue

            processed_ids.add(sample_id)

    if non_empty_lines and malformed_lines / non_empty_lines > 0.1:
        raise ValueError(f"Too many malformed lines in {jsonl_path}")

    return processed_ids


def resolve_resume_path(
    resume: str | None,
    output_dir: Path,
    default_pattern: str = "*.jsonl",
) -> tuple[Path | None, bool]:
    """Resolve resume configuration into a path and resume-mode flag."""
    if resume is None:
        return None, False

    if resume == "latest":
        latest = find_latest_jsonl(output_dir, default_pattern)
        if latest is None:
            logging.warning(
                "No resume file found in %s for pattern %s",
                output_dir,
                default_pattern,
            )
            return None, False
        return latest, True

    path = Path(resume).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Resume file not found: {path}")
    return path, True
