from __future__ import annotations

import json

from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset


def test_evidence_retry_seed7x3_preset_matches_spec() -> None:
    preset = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME)

    assert preset.method["name"] == "evidence_retry_seed7x3"
    assert preset.method["strict_no_quality_gate"] is True
    assert preset.method["strict_code_only_detector"] is True
    assert preset.generation["seed"] == 7
    assert preset.generation["evidence_retry_attempts"] == 3
    assert preset.generation["evidence_retry_seed_stride"] == 101
    assert preset.generation["retry_repetition_penalty"] == 4.0
    assert preset.semantic_lsh["rule_name"] == "semantic_lsh"
    assert preset.semantic_lsh["lsh_d"] == 4
    assert preset.semantic_lsh["lsh_gamma"] == 0.25
    assert preset.detector["statistic"] == "calibrated_context_mean_proxy_penalized"
    assert preset.detector["proxy_penalty_alpha"] == 0.4
    assert preset.runtime["default_phases"] == [
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
    ]
    assert preset.gate_data == {}
    assert preset.gate_train == {}
    assert preset.gate_validate == {}


def test_evidence_retry_preset_is_deep_copied_between_loads() -> None:
    first = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME)
    first.generation["seed"] = -1
    first.runtime["default_phases"].clear()

    second = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME)
    assert second.generation["seed"] == 7
    assert second.runtime["default_phases"] == [
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
    ]


def test_default_preset_has_no_legacy_sections() -> None:
    payload = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    forbidden = [
        "token_channel",
        "dual-channel",
        "adaptive_gamma",
        "watermark",
        "extract",
        "pretrain",
        "token_channel_train",
        "build_entropy_profile",
    ]
    for field in forbidden:
        assert field not in serialized
