"""Offline entropy calibration profile for adaptive watermark scheduling."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


_REQUIRED_QUANTILES: tuple[str, ...] = ("p10", "p50", "p75", "p90", "p95")


@dataclass(frozen=True)
class EntropyProfile:
    """Canonical quantile profile scoped to language + model family."""

    language: str
    model_family: str
    quantiles_units_map: dict[str, int] = field(default_factory=dict)
    strategy: str = "piecewise_quantile"

    def __post_init__(self) -> None:
        if not isinstance(self.language, str) or not self.language:
            raise ValueError("language must be a non-empty string")
        if not isinstance(self.model_family, str) or not self.model_family:
            raise ValueError("model_family must be a non-empty string")
        if not isinstance(self.quantiles_units_map, dict):
            raise ValueError("quantiles_units must be an object")

        validated: dict[str, int] = {}
        for quantile in _REQUIRED_QUANTILES:
            if quantile not in self.quantiles_units_map:
                raise ValueError(f"quantiles_units missing required key: {quantile}")
            value = self.quantiles_units_map[quantile]
            if type(value) is not int:
                raise ValueError(f"quantiles_units[{quantile}] must be an integer")
            if value < 0:
                raise ValueError(f"quantiles_units[{quantile}] must be >= 0")
            validated[quantile] = value

        ordered = [validated[name] for name in _REQUIRED_QUANTILES]
        if any(ordered[index] > ordered[index + 1] for index in range(len(ordered) - 1)):
            raise ValueError("quantiles_units must be monotonic: p10 <= p50 <= p75 <= p90 <= p95")

        object.__setattr__(self, "quantiles_units_map", validated)

    @classmethod
    def load(cls, path: str | Path) -> "EntropyProfile":
        """Load and validate an entropy profile JSON file."""
        profile_path = Path(path)
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("profile JSON root must be an object")

        return cls(
            language=payload.get("language"),
            model_family=payload.get("model_family"),
            quantiles_units_map=payload.get("quantiles_units"),
            strategy=payload.get("strategy", "piecewise_quantile"),
        )

    def quantile_units(self, quantile: str) -> int:
        """Return entropy quantile in canonical integer units."""
        try:
            return self.quantiles_units_map[quantile]
        except KeyError as exc:
            raise ValueError(f"unknown quantile: {quantile}") from exc
