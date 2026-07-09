"""Sweep specification: YAML → Cartesian-product expansion."""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import yaml


def short_hash(values: dict[str, Any]) -> str:
    """Return a deterministic 8-char SHA-256 prefix of the JSON-serialised values."""
    payload = json.dumps(values, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


@dataclass(frozen=True)
class ResolvedConfig:
    axes_values: dict[str, Any]
    config: dict[str, Any]
    run_id: str

    def short_hash(self) -> str:
        return short_hash(self.axes_values)


@dataclass(frozen=True)
class SweepSpec:
    name: str
    base_config: Path
    phases: tuple[str, ...]
    axes: tuple[tuple[str, tuple[Any, ...]], ...]
    output_dir: Path
    metrics: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SweepSpec":
        yaml_path = Path(path).resolve()
        if not yaml_path.exists():
            raise FileNotFoundError(f"sweep yaml not found: {yaml_path}")
        with open(yaml_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"sweep yaml must be a mapping, got {type(raw).__name__}")

        name = raw.get("name")
        if not name or not isinstance(name, str):
            raise ValueError("sweep yaml: 'name' must be a non-empty string")

        base_config_raw = raw.get("base_config")
        if not base_config_raw:
            raise ValueError("sweep yaml: 'base_config' is required")
        base_config = (yaml_path.parent / base_config_raw).resolve()

        phases_raw = raw.get("phases")
        if not phases_raw or not isinstance(phases_raw, list):
            raise ValueError("sweep yaml: 'phases' must be a non-empty list")
        phases = tuple(str(p) for p in phases_raw)

        axes_raw = raw.get("axes")
        if not isinstance(axes_raw, dict) or not axes_raw:
            raise ValueError("sweep yaml: 'axes' must be a non-empty mapping")
        axes_pairs: list[tuple[str, tuple[Any, ...]]] = []
        for axis_path, values in axes_raw.items():
            if not isinstance(values, list) or not values:
                raise ValueError(
                    f"sweep yaml: axes['{axis_path}'] must be a non-empty list, got {values!r}"
                )
            axes_pairs.append((str(axis_path), tuple(values)))

        output_dir_raw = raw.get("output_dir")
        if not output_dir_raw:
            raise ValueError("sweep yaml: 'output_dir' is required")
        output_dir = (yaml_path.parent / output_dir_raw).resolve()

        metrics_raw = raw.get("metrics")
        if not metrics_raw or not isinstance(metrics_raw, list):
            raise ValueError("sweep yaml: 'metrics' must be a non-empty list")
        metrics = tuple(str(m) for m in metrics_raw)

        return cls(
            name=name,
            base_config=base_config,
            phases=phases,
            axes=tuple(axes_pairs),
            output_dir=output_dir,
            metrics=metrics,
        )

    def load_base_config(self) -> dict[str, Any]:
        with open(self.base_config, encoding="utf-8") as f:
            return json.load(f)

    def expand(self) -> Iterator[ResolvedConfig]:
        base = self.load_base_config()
        axis_paths = [p for p, _ in self.axes]
        axis_value_lists = [v for _, v in self.axes]
        for index, combo in enumerate(itertools.product(*axis_value_lists)):
            axes_values = dict(zip(axis_paths, combo))
            cfg = copy.deepcopy(base)
            for path, value in axes_values.items():
                _apply_dotted_override(cfg, path, value)
            run_id = f"{index:04d}_{short_hash(axes_values)}"
            yield ResolvedConfig(
                axes_values=axes_values,
                config=cfg,
                run_id=run_id,
            )


def _apply_dotted_override(cfg: dict[str, Any], dotted_path: str, value: Any) -> None:
    """Walk dotted_path, creating intermediate dicts if missing, set leaf."""
    parts = dotted_path.split(".")
    cursor: Any = cfg
    for part in parts[:-1]:
        if not isinstance(cursor, dict):
            raise ValueError(
                f"axes path '{dotted_path}': cannot descend into non-dict at '{part}'"
            )
        if part not in cursor or not isinstance(cursor.get(part), dict):
            cursor[part] = {}
        cursor = cursor[part]
    if not isinstance(cursor, dict):
        raise ValueError(f"axes path '{dotted_path}': leaf parent is not a dict")
    cursor[parts[-1]] = value
