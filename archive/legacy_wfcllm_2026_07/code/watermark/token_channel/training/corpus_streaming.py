"""Streaming loader for token-channel training corpus to avoid OOM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from wfcllm.watermark.token_channel.training.corpus import TRAINING_CACHE_SCHEMA_VERSION


def stream_training_cache(path: str | Path) -> Iterator[dict[str, object]]:
    """Stream training rows from cache file without loading entire file into memory.

    This function parses the JSON file incrementally, yielding one row at a time.
    It expects the cache file to have the format:
    {
      "schema_version": "...",
      "rows": [
        {...},
        {...},
        ...
      ]
    }

    Args:
        path: Path to the training cache JSON file

    Yields:
        Individual training row dicts

    Raises:
        ValueError: If schema version is incompatible or file format is invalid
    """
    cache_path = Path(path)

    with open(cache_path, 'r', encoding='utf-8') as f:
        # Read opening brace and schema_version
        line = f.readline().strip()
        if line != '{':
            raise ValueError("training cache must start with '{'")

        # Read schema_version line
        line = f.readline().strip()
        if not line.startswith('"schema_version":'):
            raise ValueError("training cache must have schema_version as first field")

        # Extract and validate schema version
        try:
            # Parse: "schema_version": "token-channel-training-corpus/v1",
            schema_line = line.rstrip(',')
            schema_obj = json.loads('{' + schema_line + '}')
            schema_version = schema_obj.get('schema_version')
        except (json.JSONDecodeError, KeyError) as exc:
            raise ValueError("training cache schema_version is malformed") from exc

        if schema_version != TRAINING_CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"training cache schema_version is incompatible: "
                f"expected {TRAINING_CACHE_SCHEMA_VERSION}, got {schema_version}"
            )

        # Read "rows": [
        line = f.readline().strip()
        if line != '"rows": [':
            raise ValueError("training cache must have 'rows' array after schema_version")

        # Stream rows one by one
        row_buffer = ""
        brace_depth = 0
        in_row = False

        for line in f:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Check for end of rows array
            if stripped == ']' and brace_depth == 0:
                # End of rows array
                break

            # Accumulate row JSON
            for char in stripped:
                if char == '{':
                    if brace_depth == 0:
                        in_row = True
                        row_buffer = char
                    else:
                        row_buffer += char
                    brace_depth += 1
                elif char == '}':
                    brace_depth -= 1
                    row_buffer += char
                    if brace_depth == 0 and in_row:
                        # Complete row accumulated
                        try:
                            row = json.loads(row_buffer)
                            yield row
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"malformed row JSON: {row_buffer[:100]}...") from exc
                        row_buffer = ""
                        in_row = False
                elif in_row:
                    row_buffer += char


def count_training_cache_rows(path: str | Path) -> int:
    """Count total rows in training cache without loading into memory.

    Args:
        path: Path to the training cache JSON file

    Returns:
        Total number of rows in the cache
    """
    count = 0
    for _ in stream_training_cache(path):
        count += 1
    return count


def split_training_cache_streaming(
    path: str | Path,
    *,
    split_ratio: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Determine train/validation split indices without loading full cache.

    This function:
    1. Counts total rows
    2. Generates shuffled indices
    3. Splits indices into train/val sets

    The actual row loading happens later during batch construction.

    Args:
        path: Path to the training cache JSON file
        split_ratio: Fraction of data to use for training (0 < split_ratio < 1)
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_indices, validation_indices)
    """
    import random

    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between 0 and 1")

    # Count total rows
    total_rows = count_training_cache_rows(path)

    if total_rows < 2:
        raise ValueError("training cache must contain at least 2 rows")

    # Generate shuffled indices
    indices = list(range(total_rows))
    random.Random(seed).shuffle(indices)

    # Split indices
    split_index = min(total_rows - 1, max(1, int(total_rows * split_ratio)))
    train_indices = indices[:split_index]
    validation_indices = indices[split_index:]

    return train_indices, validation_indices


def load_rows_by_indices(
    path: str | Path,
    indices: list[int],
) -> Iterator[dict[str, object]]:
    """Stream only the rows at specified indices from cache file.

    Args:
        path: Path to the training cache JSON file
        indices: Row indices to load, yielded in the same order as requested

    Yields:
        Training rows at the specified indices
    """
    if not indices:
        return

    indices_set = set(indices)
    max_index = max(indices_set)
    rows_by_index: dict[int, dict[str, object]] = {}
    current_index = 0

    for row in stream_training_cache(path):
        if current_index in indices_set:
            rows_by_index[current_index] = row
            if len(rows_by_index) == len(indices_set):
                break
        current_index += 1

        # Early exit if we've loaded all requested indices
        if current_index > max_index:
            break

    missing_indices = [index for index in indices if index not in rows_by_index]
    if missing_indices:
        raise ValueError(f"training cache is missing requested row indices: {missing_indices}")

    for index in indices:
        yield rows_by_index[index]
