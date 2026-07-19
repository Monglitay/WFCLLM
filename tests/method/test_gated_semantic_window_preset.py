from __future__ import annotations

import json
import copy
import pickle
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from wfcllm.method.config import WFCLLMMethodPreset
from wfcllm.method.presets import (
    GATED_SEMANTIC_WINDOW_V1_NAME,
    load_method_preset,
)


SEVEN_PHASES = [
    "gate-data",
    "gate-train",
    "gate-validate",
    "generate",
    "calibrate",
    "detect",
    "report",
]
FAST_PHASES = [
    "gate-data",
    "gate-train",
    "generate",
    "calibrate",
    "detect",
    "report",
]
FIVE_PHASES = ["generate", "calibrate", "detect", "report", "audit"]
FOUR_PHASES = ["generate", "calibrate", "detect", "report"]


def _replace_section(
    preset: WFCLLMMethodPreset,
    section_name: str,
    mutate,
) -> WFCLLMMethodPreset:
    section = deepcopy(preset.to_dict()[section_name])
    mutate(section)
    return replace(preset, **{section_name: section})


def test_gated_preset_has_full_phase_sequence_and_no_secret() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)

    assert preset.method["name"] == GATED_SEMANTIC_WINDOW_V1_NAME
    assert preset.runtime["default_phases"] == FAST_PHASES
    assert preset.runtime["external_validated_bundle_phases"] == FOUR_PHASES
    assert preset.method["windowing"]["max_units"] == 3
    assert preset.method["gate"]["max_input_tokens"] == 256
    assert preset.method["gate"]["require_validated"] is False
    assert preset.gate_data["training_key_count"] == 32
    assert preset.gate_validate["holdout_key_count"] == 8
    assert "secret" not in json.dumps(preset.to_dict(), sort_keys=True).lower()


def test_gated_preset_contains_frozen_contracts_and_thresholds() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)

    assert preset.method["windowing"]["contract_version"] == "python-statement-window/v1"
    assert preset.method["semantic"]["parent_descriptor_version"] == "python-statement-window/v1"
    assert preset.method["gate"]["input_contract_version"] == "wfcllm-gate-input/v1"
    assert preset.method["gate"]["bundle_contract_version"] == "wfcllm-gate-bundle/v1"
    assert preset.method["rewrite"] == {
        "candidate_zero": "original_window",
        "max_attempts": 3,
        "experiment_budgets": [1, 3],
        "key_blind": True,
        "temperature": 0.2,
        "top_p": 0.95,
        "max_new_tokens": 16,
        "generation_attempts": 9,
        "candidate_selection": "unique-key-blind-structural-fallback/v1",
    }
    assert preset.gate_data["label_thresholds"] == {
        "reliable_success_rate_r3_min": 0.60,
        "structurally_valid_rewrite_rate_r3_min": pytest.approx(2 / 3),
        "unstable_candidate_rate_r3_max": 0.10,
    }
    assert preset.gate_validate["acceptance_thresholds"] == {
        "decision_agreement_min": 0.999,
        "float_quantized_accepted_set_agreement_min": 0.999,
        "formal_accepted_span_consensus_min": 1.0,
        "suitable_false_positive_rate_max": 0.05,
    }
    assert preset.detector["mode"] == "wfcllm-gated-semantic-window/v1"
    assert preset.gate_data["feasibility_contract_version"] == "gate-data-feasibility/v1"
    assert len(preset.gate_data["feasibility_thresholds"]) == 15
    assert preset.detector["target_fpr"] == 0.05


def test_gated_preset_is_deep_copied_between_loads() -> None:
    first = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(TypeError):
        first.method["windowing"] = {}
    with pytest.raises(TypeError):
        first.method["windowing"]["excluded_statement_types"] += ("made_up",)

    payload = first.to_dict()
    payload["method"]["windowing"]["excluded_statement_types"].append("made_up")
    payload["runtime"]["default_phases"].clear()

    second = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    assert first.method is not second.method
    assert first.method["windowing"] is not second.method["windowing"]
    assert "made_up" not in second.method["windowing"]["excluded_statement_types"]
    assert second.runtime["default_phases"] == FAST_PHASES
    assert isinstance(payload["method"], dict)
    assert isinstance(payload["runtime"]["default_phases"], list)


def test_gated_preset_supports_deepcopy_and_pickle_as_frozen_graphs() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)

    copied = copy.deepcopy(preset)
    restored = pickle.loads(pickle.dumps(preset))

    for candidate in (copied, restored):
        assert candidate == preset
        assert candidate is not preset
        assert candidate.method is not preset.method
        assert candidate.to_dict() == preset.to_dict()
        with pytest.raises(TypeError):
            candidate.method["gate"] = {}
        with pytest.raises((AttributeError, TypeError)):
            candidate.runtime["default_phases"].append("made_up")


def test_gated_preset_rejects_invalid_method_specific_contract() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)

    with pytest.raises(ValueError, match="max_units"):
        replace(
            preset,
            method={
                **preset.method,
                "windowing": {**preset.method["windowing"], "max_units": 4},
            },
        )


def test_gated_preset_supports_external_validated_bundle_four_phase_mode() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    external = replace(
        preset,
        method={
            **preset.method,
            "gate": {
                **preset.method["gate"],
                "bundle_path": "local/bundles/gate-v1",
                "bundle_sha256": "a" * 64,
            },
        },
        runtime={
            **preset.runtime,
            "default_phases": FOUR_PHASES,
        },
    )

    assert external.runtime["default_phases"] == FOUR_PHASES
    assert external.method["gate"]["require_validated"] is False


def test_gated_preset_allows_local_paths_containing_secrets_word() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    external = replace(
        preset,
        method={
            **preset.method,
            "gate": {
                **preset.method["gate"],
                "bundle_path": "local/secrets/gate-v1",
                "bundle_sha256": "a" * 64,
            },
        },
        artifacts={"run_root": "local/secrets/runs"},
        gate_train={
            **preset.gate_train,
            "base_encoder_id": "data/models/secrets/apiKey=value",
        },
        runtime={**preset.runtime, "default_phases": FOUR_PHASES},
    )
    assert external.method["gate"]["bundle_path"] == "local/secrets/gate-v1"
    assert json.loads(json.dumps(external.to_dict())) == external.to_dict()


def test_gated_preset_rejects_four_phase_mode_without_bundle() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="configured local phases"):
        replace(
            preset,
            runtime={**preset.runtime, "default_phases": FOUR_PHASES},
        )


def test_gated_preset_rejects_path_traversal_bundle() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="local path"):
        replace(
            preset,
            method={
                **preset.method,
                "gate": {
                    **preset.method["gate"],
                    "bundle_path": "../outside",
                    "bundle_sha256": "a" * 64,
                },
            },
            runtime={**preset.runtime, "default_phases": FOUR_PHASES},
        )


def test_gated_preset_rejects_unknown_method_schema_field() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="unknown fields"):
        replace(preset, method={**preset.method, "unexpected": True})


def test_gated_preset_requires_experimental_true_and_typed_rewrite_budgets() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="experimental"):
        replace(preset, method={**preset.method, "experimental": False})

    with pytest.raises(ValueError, match="experiment_budgets"):
        replace(
            preset,
            method={
                **preset.method,
                "rewrite": {
                    **preset.method["rewrite"],
                    "experiment_budgets": [True, 3],
                },
            },
        )

    with pytest.raises(ValueError, match="max_attempts"):
        replace(
            preset,
            method={
                **preset.method,
                "rewrite": {
                    **preset.method["rewrite"],
                    "max_attempts": True,
                },
            },
        )

    with pytest.raises(ValueError, match="rewrite_budgets"):
        _replace_section(
            preset,
            "gate_data",
            lambda value: value.__setitem__("rewrite_budgets", [True, 3]),
        )

    with pytest.raises(ValueError, match="rewrite_count"):
        _replace_section(
            preset,
            "gate_data",
            lambda value: value.__setitem__("rewrite_count", True),
        )

    with pytest.raises(ValueError, match="bundle"):
        replace(
            preset,
            method={
                **preset.method,
                "gate": {
                    **preset.method["gate"],
                    "bundle_path": "bundle",
                    "bundle_sha256": None,
                },
            },
        )


def test_unknown_preset_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown WFCLLM method preset"):
        load_method_preset("unknown")


def test_committed_gated_config_matches_preset_and_has_no_absolute_or_secret_value() -> None:
    config_path = (
        Path(__file__).parents[2]
        / "configs"
        / "wfcllm"
        / "gated_semantic_window_v1.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    assert payload == load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    serialized = json.dumps(payload, sort_keys=True).lower()
    assert "secret" not in serialized
    assert "/home/" not in serialized


def test_preset_dataclass_defaults_gate_sections_only_for_legacy_method() -> None:
    legacy = WFCLLMMethodPreset(
        method={
            "name": "evidence_retry_seed7x3",
            "strict_no_quality_gate": True,
            "strict_code_only_detector": True,
        }
    )
    assert legacy.gate_data == {}
    assert legacy.gate_train == {}
    assert legacy.gate_validate == {}


def test_preset_preserves_all_seven_legacy_positional_arguments() -> None:
    method = {
        "name": "evidence_retry_seed7x3",
        "strict_no_quality_gate": True,
        "strict_code_only_detector": True,
    }
    generation = {"generation": 1}
    semantic_lsh = {"semantic_lsh": 2}
    detector = {"detector": 3}
    calibration = {"calibration": 4}
    artifacts = {"run_root": "custom"}
    runtime = {"default_phases": FIVE_PHASES}

    preset = WFCLLMMethodPreset(
        method,
        generation,
        semantic_lsh,
        detector,
        calibration,
        artifacts,
        runtime,
    )

    assert preset.generation is generation
    assert preset.semantic_lsh is semantic_lsh
    assert preset.detector is detector
    assert preset.calibration is calibration
    assert preset.artifacts is artifacts
    assert preset.runtime is runtime
    assert preset.gate_data == {}
    assert preset.gate_train == {}
    assert preset.gate_validate == {}


def test_gate_train_records_exact_formal_loss_weights() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    assert preset.gate_train["loss_weights"] == {
        "close_bce": 1.0,
        "suitable_bce": 1.0,
        "close_positive": 1.0,
        "suitable_positive": 1.0,
        "suitable_false_positive": 4.0,
    }


@pytest.mark.parametrize(
    ("section_name", "mutation", "match"),
    [
        ("gate_data", lambda value: value.pop("schema_version"), "missing fields"),
        ("gate_data", lambda value: value.__setitem__("junk", True), "unknown fields"),
        ("gate_data", lambda value: value.__setitem__("training_key_count", 31), "training_key_count"),
        ("gate_data", lambda value: value.__setitem__("rewrite_budgets", [1]), "rewrite_budgets"),
        ("gate_data", lambda value: value["label_thresholds"].__setitem__("unstable_candidate_rate_r3_max", 0.2), "label_thresholds"),
        ("gate_data", lambda value: value["feasibility_thresholds"].__setitem__("pilot_suitable_positive_min", 9), "feasibility_thresholds"),
        ("gate_train", lambda value: value.pop("optimizer"), "missing fields"),
        ("gate_train", lambda value: value.__setitem__("junk", True), "unknown fields"),
        ("gate_train", lambda value: value["losses"].append("made_up"), "losses"),
        ("gate_train", lambda value: value["loss_weights"].__setitem__("made_up", 0.2), "loss_weights"),
        ("gate_train", lambda value: value["loss_weights"].__setitem__("close_bce", True), "loss_weights"),
        ("gate_validate", lambda value: value.pop("batch_sizes"), "missing fields"),
        ("gate_validate", lambda value: value.__setitem__("junk", True), "unknown fields"),
        ("gate_validate", lambda value: value.__setitem__("batch_sizes", [1, 8]), "batch_sizes"),
        ("gate_validate", lambda value: value.__setitem__("orders", ["reverse"]), "orders"),
        ("gate_validate", lambda value: value.__setitem__("independent_reloads", 2), "independent_reloads"),
        ("gate_validate", lambda value: value["acceptance_thresholds"].__setitem__("decision_agreement_min", 0.9), "acceptance_thresholds"),
        ("gate_validate", lambda value: value["acceptance_thresholds"].__setitem__("formal_accepted_span_consensus_min", True), "acceptance_thresholds"),
    ],
)
def test_gated_low_level_sections_reject_missing_junk_or_tampered_values(
    section_name: str,
    mutation,
    match: str,
) -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match=match):
        _replace_section(preset, section_name, mutation)


@pytest.mark.parametrize(
    ("section_name", "mutation"),
    [
        ("generation", lambda value: value.__setitem__("deployment_key", "0101")),
        ("detector", lambda value: value.__setitem__("api_key", "value")),
        ("gate_data", lambda value: value.__setitem__("raw_training_key", "0101")),
        ("gate_train", lambda value: value.__setitem__("key_material", "0101")),
        ("gate_validate", lambda value: value.__setitem__("private_key", "value")),
    ],
)
def test_gated_public_config_rejects_secret_carriers_recursively(
    section_name: str,
    mutation,
) -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="public config|unknown fields"):
        _replace_section(preset, section_name, mutation)


@pytest.mark.parametrize(
    ("section_name", "mutation"),
    [
        ("generation", lambda value: value.__setitem__("deploymentKey", "0101")),
        ("gate_data", lambda value: value.__setitem__("rawTrainingKey", "0101")),
        ("detector", lambda value: value.__setitem__("apiKey", "value")),
        ("gate_train", lambda value: value.__setitem__("keyMaterial", "0101")),
    ],
)
def test_gated_public_config_rejects_camel_case_secret_carriers(
    section_name: str,
    mutation,
) -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match="public config"):
        _replace_section(preset, section_name, mutation)


def test_gated_public_config_allows_irreversible_key_metadata() -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    gate_data = preset.to_dict()["gate_data"]
    gate_data.update(
        {
            "training_key_bank_manifest_sha256": "a" * 64,
            "holdout_key_bank_manifest_sha256": "b" * 64,
            "training_key_bank_id": "training-key-bank/v1:sha256:" + "c" * 64,
            "holdout_key_bank_id": "holdout-key-bank/v1:sha256:" + "d" * 64,
        }
    )
    resolved = replace(preset, gate_data=gate_data)
    assert resolved.gate_data["training_key_count"] == 32
    assert resolved.gate_data["training_key_bank_manifest_sha256"] == "a" * 64


@pytest.mark.parametrize(
    ("section_name", "mutation", "match"),
    [
        ("generation", lambda value: value.__setitem__("device", "0101"), "generation.device"),
        ("semantic_lsh", lambda value: value.__setitem__("lsh_d", "0101"), "semantic_lsh.lsh_d"),
        ("gate_train", lambda value: value.__setitem__("base_encoder_id", "0101"), "base_encoder_id"),
    ],
)
def test_gated_public_config_rejects_raw_values_in_non_secret_carrier_fields(
    section_name: str,
    mutation,
    match: str,
) -> None:
    preset = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME)
    with pytest.raises(ValueError, match=match):
        _replace_section(preset, section_name, mutation)
