"""Sweep specification: YAML → Cartesian-product expansion."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


def short_hash(values: dict[str, Any]) -> str:
    """Return a deterministic 8-char SHA-256 prefix of the JSON-serialised values."""
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class ResolvedConfig:
    """One concrete combination from a sweep: full config dict + the axis values that produced it."""
    axes_values: dict[str, Any]
    config: dict[str, Any]
    run_id: str

    def short_hash(self) -> str:
        return short_hash(self.axes_values)


@dataclass(frozen=True)
class SweepSpec:
    """Parsed YAML sweep specification."""
    name: str
    base_config: Path
    phases: tuple[str, ...]
    axes: tuple[tuple[str, tuple[Any, ...]], ...]   # ordered axis pairs (path, values)
    output_dir: Path
    metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepSpec":
        raise NotImplementedError  # filled in next step

    def expand(self) -> Iterator[ResolvedConfig]:
        raise NotImplementedError  # filled in Task 2
