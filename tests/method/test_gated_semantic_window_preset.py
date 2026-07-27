from __future__ import annotations

from dataclasses import replace

import pytest

from wfcllm.method.config import WFCLLMMethodPreset
from wfcllm.method.presets import (
    GATED_SEMANTIC_WINDOW_V1_NAME,
    load_method_preset,
)
from wfcllm.orchestration.state import PHASES


def test_gate_preset_is_the_only_method_and_uses_full_chain() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    assert preset.method["name"] == GATED_SEMANTIC_WINDOW_V1_NAME
    assert list(preset.runtime["default_phases"]) == PHASES
    assert "bundle_path" not in preset.method["gate"]
    assert "bundle_sha256" not in preset.method["gate"]
    assert preset.calibration["target_negative_count"] == 100
    assert preset.semantic_lsh["lsh_d"] == 12
    assert preset.semantic_lsh["lsh_gamma"] == pytest.approx(0.45)

    with pytest.raises(ValueError, match="unknown WFCLLM method preset"):
        load_method_preset("removed_method")


def test_gate_preset_is_deeply_immutable_and_copyable() -> None:
    first = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(TypeError):
        first.method["gate"]["max_input_tokens"] = 128
    payload = first.to_dict()
    payload["runtime"]["default_phases"].clear()
    second = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    assert list(second.runtime["default_phases"]) == PHASES


def test_gate_preset_rejects_short_or_compatibility_phase_modes() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="full reproduction chain"):
        replace(
            preset,
            runtime={
                "default_phases": [
                    "generate",
                    "calibrate",
                    "detect",
                    "report",
                ]
            },
        )


def test_gate_preset_rejects_unknown_gate_fields() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    gate = dict(preset.method["gate"])
    gate["removed_path"] = "unsupported"
    with pytest.raises(ValueError, match="method.gate has unknown fields"):
        replace(preset, method={**preset.method, "gate": gate})


def test_method_dataclass_rejects_non_gate_identity() -> None:
    with pytest.raises(ValueError, match="unsupported method.name"):
        WFCLLMMethodPreset(method={"name": "unsupported"})
