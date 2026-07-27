from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from wfcllm.gate.input import GATE_INPUT_CONTRACT_VERSION
from wfcllm.windowing.contracts import is_supported_window_contract

GATED_DETECTOR_MODE = "wfcllm-gated-semantic-window/v1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GatedDetectionConfig:
    """Immutable bindings required by final-code-only gated detection."""

    gate_bundle_path: Path
    gate_bundle_sha256: str
    window_contract_version: str
    gate_input_contract_version: str
    semantic_encoder_hash: str
    lsh_config_hash: str
    minimum_reliable_windows: int
    calibration_artifact_path: Path
    detector_mode: str = GATED_DETECTOR_MODE

    def __post_init__(self) -> None:
        for name in ("gate_bundle_path", "calibration_artifact_path"):
            value = getattr(self, name)
            if not isinstance(value, Path) or str(value) in {"", "."}:
                raise ValueError(f"{name} must be a non-empty pathlib.Path")
        for name in (
            "gate_bundle_sha256",
            "semantic_encoder_hash",
            "lsh_config_hash",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not is_supported_window_contract(self.window_contract_version):
            raise ValueError(
                "window_contract_version must be a supported language contract"
            )
        if self.gate_input_contract_version != GATE_INPUT_CONTRACT_VERSION:
            raise ValueError(
                "gate_input_contract_version must be "
                f"{GATE_INPUT_CONTRACT_VERSION!r}"
            )
        if (
            isinstance(self.minimum_reliable_windows, bool)
            or not isinstance(self.minimum_reliable_windows, int)
            or self.minimum_reliable_windows <= 0
        ):
            raise ValueError("minimum_reliable_windows must be a positive integer")
        if self.detector_mode != GATED_DETECTOR_MODE:
            raise ValueError(f"detector_mode must be {GATED_DETECTOR_MODE!r}")
