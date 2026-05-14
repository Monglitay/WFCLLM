"""Build an entropy profile JSON from a watermark debug log.

Logic moved from scripts/calibrate.py:_build_entropy_profile (Phase 3 refactor).
The CLI shell that used to wrap this lives in scripts/build_entropy_profile.py.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

_ENTROPY_PATTERN = re.compile(r"entropy=(?P<entropy>-?\d+(?:\.\d+)?)")
_ENTROPY_SCALE = 10000
_QUANTILES: tuple[tuple[str, float], ...] = (
    ("p10", 0.10),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
)


def _nearest_rank_quantile(sorted_values: list[int], probability: float) -> int:
    index = max(1, math.ceil(probability * len(sorted_values))) - 1
    return sorted_values[index]


def build_entropy_profile_from_log(
    *,
    input_log: str | Path,
    output: str | Path,
    language: str,
    model_family: str,
    strategy: str = "piecewise_quantile",
    profile_id: str | None = None,
) -> Path:
    """Parse `entropy=<float>` lines from `input_log`, compute quantiles, write JSON.

    Returns the resolved output path. Raises ``ValueError`` when the log
    contains no parseable entropy values.
    """
    entropy_units: list[int] = []
    with open(input_log, encoding="utf-8") as handle:
        for line in handle:
            match = _ENTROPY_PATTERN.search(line)
            if match is None:
                continue
            entropy_value = float(match.group("entropy"))
            entropy_units.append(max(0, int(round(entropy_value * _ENTROPY_SCALE))))

    if not entropy_units:
        raise ValueError("No entropy=<float> entries found in input log")

    entropy_units.sort()
    payload: dict = {
        "language": language,
        "model_family": model_family,
        "strategy": strategy,
        "sample_count": len(entropy_units),
        "quantiles_units": {
            name: _nearest_rank_quantile(entropy_units, probability)
            for name, probability in _QUANTILES
        },
    }
    if profile_id is not None:
        payload["profile_id"] = profile_id

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
