from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WFCLLMMethodPreset:
    method: dict[str, Any]
    generation: dict[str, Any] = field(default_factory=dict)
    semantic_lsh: dict[str, Any] = field(default_factory=dict)
    detector: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method.get("name") != "evidence_retry_seed7x3":
            raise ValueError("method.name must be evidence_retry_seed7x3")
        if self.method.get("strict_no_quality_gate") is not True:
            raise ValueError("method.strict_no_quality_gate must be true")
        if self.method.get("strict_code_only_detector") is not True:
            raise ValueError("method.strict_code_only_detector must be true")
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
