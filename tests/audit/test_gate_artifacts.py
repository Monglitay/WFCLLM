from __future__ import annotations

import math

import pytest

from wfcllm.audit.artifact_integrity import (
    audit_gate_artifact,
    reject_secret_key_leak,
)


@pytest.mark.parametrize(
    "field",
    [
        "raw_training_key",
        "rawTrainingKey",
        "raw_training_keys",
        "rawTrainingKeys",
        "rawtrainingkey",
        "rawSecrets",
        "deployment-key",
        "deploymentKey",
        "deploymentKeys",
        "key_material",
        "keyMaterial",
        "target_lsh_region",
        "targetLSHRegion",
        "targetLSHRegions",
        "secret_key",
        "rawSecretKey",
    ],
)
def test_gate_artifacts_reject_secret_fields_with_nested_path(field: str) -> None:
    with pytest.raises(ValueError, match=rf"manifest\.entries\[0\]\.{field}"):
        audit_gate_artifact({"manifest": {"entries": [{field: "secret"}]}})


@pytest.mark.parametrize(
    "field",
    [
        "pass",
        "test_result",
        "testResult",
        "test_results",
        "testResults",
        "testresults",
        "correctness_score",
        "correctnessScore",
        "syntax_valid",
        "syntaxValid",
        "syntaxValids",
        "syntaxvalid",
        "quality_rank",
        "qualityRank",
        "qualityRanks",
        "signaturesCompatible",
        "signaturecompatible",
        "unit_test_outcome",
        "oracle_reward",
    ],
)
def test_formal_gate_data_rejects_quality_proxy_fields_and_aliases(
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"groups\[0\]\.{field}"):
        audit_gate_artifact({"groups": [{field: True}]})


def test_formal_gate_data_allows_semantic_preservation_evidence() -> None:
    assert (
        audit_gate_artifact(
            {
                "candidate_observation": {
                    "semantic_reference_cosine": 0.97,
                    "semantic_preservation_passed": True,
                }
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "markers",
    [
        {"diagnostic_test_backend": True, "formal_eligible": False},
        {"diagnostic_only": True, "not_official_method": True},
    ],
)
def test_explicitly_nonformal_diagnostic_artifact_may_report_quality_results(
    markers: dict[str, bool],
) -> None:
    assert (
        audit_gate_artifact(
            {**markers, "results": [{"correctnessScore": 0.5, "test_result": "fail"}]}
        )
        is None
    )


def test_allowlisted_diagnostic_artifact_type_may_report_quality_results() -> None:
    assert (
        audit_gate_artifact(
            {
                "diagnostic_only": True,
                "not_official_method": True,
                "artifact_type": "diagnostic-report",
                "schema_version": "diagnostic-report/v1",
                "correctness_score": 0.5,
            }
        )
        is None
    )


def test_unknown_diagnostic_artifact_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="diagnostic.*artifact_type"):
        audit_gate_artifact(
            {
                "diagnostic_only": True,
                "not_official_method": True,
                "artifact_type": "unregistered-report",
                "correctness_score": 0.5,
            }
        )


@pytest.mark.parametrize(
    "incomplete_markers",
    [
        {"diagnostic_test_backend": True},
        {"formal_eligible": False},
        {"diagnostic_only": True},
        {"not_official_method": True},
    ],
)
def test_incomplete_diagnostic_marker_cannot_bypass_formal_quality_audit(
    incomplete_markers: dict[str, bool],
) -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        audit_gate_artifact(
            {**incomplete_markers, "results": {"correctnessScore": 0.5}}
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"diagnostic_test_backend": True, "formal_eligible": True},
        {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "validated": True,
        },
        {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "artifact_type": "formal-bundle",
        },
        {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "formal_bundle": True,
        },
        {
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "diagnostic_only": True,
            "not_official_method": True,
        },
        {"diagnostic_only": True, "not_official_method": False},
    ],
)
def test_diagnostic_quality_exemption_rejects_conflicting_identity(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="diagnostic"):
        audit_gate_artifact({**payload, "correctness_score": 0.5})


@pytest.mark.parametrize(
    "artifact_type",
    [
        "gate-data-jsonl",
        "training-metrics",
        "checkpoint-metadata",
        "bundle-manifest",
        "validation-summary",
        "generation-window-audit",
        "gated-calibration",
        "gated-detection-details",
        "wfcllm_detection_calibration",
    ],
)
def test_diagnostic_marker_cannot_relabel_known_formal_artifact_type(
    artifact_type: str,
) -> None:
    with pytest.raises(ValueError, match="diagnostic.*formal"):
        audit_gate_artifact(
            {
                "diagnostic_test_backend": True,
                "formal_eligible": False,
                "artifact_type": artifact_type,
                "correctness_score": 0.5,
            }
        )


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("contract_version", "wfcllm-gate-training-metrics/v1"),
        ("contract_version", "wfcllm-gate-training-checkpoint/v1"),
        ("contract_version", "wfcllm-gate-development-summary/v1"),
        ("contract_version", "wfcllm-gate-bundle/v1"),
        ("contract_version", "wfcllm-gate-model-state/v1"),
        ("contract_version", "wfcllm-gate-validation/v1"),
        ("contract_version", "wfcllm-gate-input/v1"),
        ("contract_version", "python-statement-window/v1"),
        ("contract_version", "gate-data-feasibility/v1"),
        ("schema_version", "wfcllm-gate-data/v1"),
        ("schema_version", "wfcllm-gate-data-manifest/v1"),
        ("schema_version", "wfcllm-gate-source-manifest/v1"),
        ("schema_version", "wfcllm-gate-split/v1"),
        ("schema_version", "wfcllm-training-key-bank-manifest/v1"),
        ("schema_version", "wfcllm-gate-train-candidate/v1"),
        ("schema_version", "wfcllm-gate-validate-publication/v1"),
        ("schema_version", "wfcllm-gate-candidate-attempts/v2"),
        ("schema_version", "wfcllm-gate-label/v1"),
        ("schema_version", "wfcllm-production-gate-adapter/v1"),
        ("schema_version", "wfcllm-detect-calibration/v1"),
    ],
)
def test_diagnostic_marker_cannot_relabel_task9_to_task12_formal_version(
    identity_field: str,
    identity_value: str,
) -> None:
    with pytest.raises(ValueError, match="diagnostic.*formal"):
        audit_gate_artifact(
            {
                "diagnostic_only": True,
                "not_official_method": True,
                identity_field: identity_value,
                "correctness_score": 0.5,
            }
        )


@pytest.mark.parametrize(
    ("identity_field", "identity_value"),
    [
        ("artifact_type", "training-metrics-v2"),
        ("artifact_type", "wfcllm-future-formal-publication"),
        ("schema_version", "wfcllm-gate-future-artifact/v99"),
        ("contract_version", "wfcllm-detect-future-contract/v2"),
    ],
)
def test_diagnostic_marker_fails_closed_for_unknown_formal_like_identity(
    identity_field: str,
    identity_value: str,
) -> None:
    with pytest.raises(ValueError, match="diagnostic.*formal"):
        audit_gate_artifact(
            {
                "diagnostic_test_backend": True,
                "formal_eligible": False,
                identity_field: identity_value,
                "correctness_score": 0.5,
            }
        )


def test_diagnostic_marker_never_exempts_secret_material() -> None:
    with pytest.raises(ValueError, match="rawTrainingKey"):
        audit_gate_artifact(
            {
                "diagnostic_test_backend": True,
                "formal_eligible": False,
                "rawTrainingKey": "secret",
            }
        )


def test_irreversible_secret_provenance_metadata_is_not_secret_material() -> None:
    assert (
        audit_gate_artifact(
            {
                "secret_source_type": "environment",
                "deployment_key_source_type": "environment",
                "key_material_present": True,
                "training_key_bank_id": "training-key-bank/v1:sha256:" + "a" * 64,
                "holdout_key_bank_sha256": "b" * 64,
            }
        )
        is None
    )


def test_gate_artifact_accepts_only_structural_lsh_gate_and_provenance_facts() -> None:
    payload = {
        "schema_version": "wfcllm-gate-data/v1",
        "window": {
            "parse_status": "ok",
            "same_parent_scope": True,
            "unit_count": 3,
            "boundary_span": [4, 12],
            "lsh_hit": True,
            "lsh_margin": 0.25,
            "stable_across_precision_modes": True,
            "stable_across_batch_modes": True,
            "close_probability": 0.93,
            "suitable_probability": 0.84,
            "close_high_threshold": 0.8,
            "sha256": "a" * 64,
            "retry_budget": 6,
        },
        "split_counts": {"train": 20_000, "validation": 1_000, "test": 1_000},
        "checkpoint_metadata": {
            "epoch": 2,
            "optimizer_steps": 300,
            "best": True,
            "status": "completed",
        },
        "generation_window_audit": {
            "gate_decision": "close",
            "candidate_trajectory": [0, 1, 2],
            "retry_count": 2,
        },
    }

    assert audit_gate_artifact(payload) is None


def test_parse_status_is_not_misread_as_a_pass_proxy() -> None:
    assert (
        audit_gate_artifact(
            {
                "parse_status": "parse_error",
                "same_parent_scope": False,
                "unit_count": 0,
            }
        )
        is None
    )


@pytest.mark.parametrize(
    "field",
    ["parse_status", "analysis", "status", "stability", "class", "loss"],
)
def test_plural_normalization_does_not_strip_s_from_safe_words(field: str) -> None:
    assert audit_gate_artifact({field: "safe structural metadata"}) is None


@pytest.mark.parametrize(
    "field",
    [
        "ｒａｗ＿ｔｒａｉｎｉｎｇ＿ｋｅｙ",
        "paſs",
    ],
)
def test_nfkc_normalized_forbidden_keys_cannot_bypass_audit(field: str) -> None:
    with pytest.raises(ValueError):
        audit_gate_artifact({field: "hidden"})


@pytest.mark.parametrize("field", ["π", "pa\u200bss", "bad\nfield", "\x7f"])
def test_formal_artifact_rejects_residual_non_ascii_or_control_keys(
    field: str,
) -> None:
    with pytest.raises(ValueError, match="field name"):
        audit_gate_artifact({field: "safe value"})


def test_unicode_is_allowed_in_values_but_not_interpreted_as_a_key() -> None:
    assert audit_gate_artifact({"note": "安全值：π 与 ſ 都只是值"}) is None


@pytest.mark.parametrize(
    "artifact",
    [
        {
            "artifact_type": "gate-data-jsonl",
            "schema_version": "wfcllm-gate-data/v1",
            "group_id": "group-1",
            "split": "train",
            "lsh_by_key_id": {"train-key-001": {"hit": True, "margin": 0.2}},
        },
        {
            "artifact_type": "training-metrics",
            "epoch": 1,
            "total_loss": 0.5,
            "validation": {"coverage": 0.9, "decision_consistency": 1.0},
        },
        {
            "artifact_type": "checkpoint-metadata",
            "epoch": 1,
            "config_hash": "a" * 64,
            "dataset_manifest_hash": "b" * 64,
            "state_dict_sha256": "c" * 64,
            "tensors": [
                {
                    "name": "classifier.weight",
                    "shape": [2, 768],
                    "dtype": "float32",
                }
            ],
        },
        {
            "artifact_type": "bundle-manifest",
            "validated": True,
            "float_model_sha256": "c" * 64,
            "quantized_model_sha256": "d" * 64,
            "training_key_bank_id": "training-key-bank/v1:sha256:" + "e" * 64,
        },
        {
            "artifact_type": "validation-summary",
            "decision_agreement": 1.0,
            "formal_accepted_span_consensus": 1.0,
            "thresholds": {"close_low": 0.2, "close_high": 0.8},
        },
        {
            "artifact_type": "generation-window-audit",
            "audit_only": True,
            "gate_decision": "close",
            "retry_budget": 6,
            "selected_candidate_index": 2,
        },
        {
            "artifact_type": "gated-calibration",
            "reference_negative_count": 200,
            "threshold_at_fpr": 0.42,
            "target_fpr": 0.05,
        },
        {
            "artifact_type": "gated-detection-details",
            "id": "HumanEval/0",
            "detector_score": 1.2,
            "p_value": 0.03,
            "window_count": 4,
        },
    ],
    ids=[
        "gate-data",
        "training-metrics",
        "checkpoint-metadata",
        "bundle-manifest",
        "validation-summary",
        "generation-window-audit",
        "gated-calibration",
        "gated-detection-details",
    ],
)
def test_gate_artifact_audit_covers_every_formal_artifact_family(
    artifact: dict[str, object],
) -> None:
    assert audit_gate_artifact(artifact) is None


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_gate_artifact_rejects_non_finite_numbers_with_path(value: float) -> None:
    with pytest.raises(ValueError, match=r"metrics\.loss"):
        audit_gate_artifact({"metrics": {"loss": value}})


def test_gate_artifact_rejects_cycles_without_recursing_forever() -> None:
    payload: dict[str, object] = {}
    payload["cycle"] = payload

    with pytest.raises(ValueError, match=r"cycle.*cyclic"):
        audit_gate_artifact(payload)


def test_gate_artifact_rejects_excessive_nesting() -> None:
    payload: dict[str, object] = {}
    cursor = payload
    for index in range(80):
        nested: dict[str, object] = {}
        cursor[f"level_{index}"] = nested
        cursor = nested

    with pytest.raises(ValueError, match="nesting depth"):
        audit_gate_artifact(payload)


def test_gate_artifact_rejects_oversized_scalar_before_serialization() -> None:
    with pytest.raises(ValueError, match=r"metadata\.note.*size limit"):
        audit_gate_artifact({"metadata": {"note": "x" * (1024 * 1024 + 1)}})


def test_gate_artifact_rejects_non_json_checkpoint_objects_without_loading_them() -> None:
    class PickleOnlyCheckpoint:
        def __reduce__(self):  # pragma: no cover - must never be invoked
            raise AssertionError("unsafe pickle protocol was invoked")

    with pytest.raises(ValueError, match=r"checkpoint\.model"):
        audit_gate_artifact({"checkpoint": {"model": PickleOnlyCheckpoint()}})


@pytest.mark.parametrize(
    "field",
    [
        "state_dict",
        "stateDict",
        "model_state",
        "optimizerState",
        "scheduler_state",
        "rng-state",
        "pickle",
    ],
)
def test_checkpoint_audit_accepts_metadata_but_not_serialized_runtime_state(
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"checkpoint\.{field}.*metadata only"):
        audit_gate_artifact({"checkpoint": {field: [0.1, 0.2]}})


@pytest.mark.parametrize(
    "context_field",
    ["checkpoint", "checkpoint_metadata", "checkpointMetadata", "checkpointmetadata"],
)
@pytest.mark.parametrize(
    "payload_field",
    [
        "weights",
        "parameters",
        "params",
        "optimizer",
        "optim",
        "scheduler",
        "model",
        "state",
        "payload",
        "modelPayload",
        "modelWeights",
        "parameterValues",
        "optimizerPayload",
        "tensorPayload",
        "trainingState",
    ],
)
def test_checkpoint_context_rejects_runtime_payload_aliases(
    context_field: str,
    payload_field: str,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{context_field}\.{payload_field}.*metadata only",
    ):
        audit_gate_artifact({context_field: {payload_field: {"opaque": 1}}})


@pytest.mark.parametrize(
    "payload_field",
    ["weights_metadata", "model_state_metadata", "optimizerMetadata"],
)
def test_runtime_tokens_cannot_hide_behind_metadata_suffix(
    payload_field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"{payload_field}.*metadata only"):
        audit_gate_artifact(
            {"checkpoint_metadata": {payload_field: {"sha256": "a" * 64}}}
        )


def test_each_nested_dict_self_identifies_checkpoint_contract() -> None:
    with pytest.raises(ValueError, match=r"envelope\.model_state"):
        audit_gate_artifact(
            {
                "envelope": {
                    "contract_version": "wfcllm-gate-training-checkpoint/v1",
                    "model_state": {"opaque": 1},
                }
            }
        )


def test_checkpoint_manifest_parent_enters_metadata_only_mode() -> None:
    with pytest.raises(ValueError, match=r"checkpoint_manifest\.weights"):
        audit_gate_artifact(
            {"checkpoint_manifest": {"weights": {"opaque": 1}}}
        )


def test_checkpoint_metadata_rejects_arbitrary_numeric_mapping_fields() -> None:
    with pytest.raises(ValueError, match=r"checkpoint_metadata\.made_up_metric"):
        audit_gate_artifact(
            {"checkpoint_metadata": {"made_up_metric": 0.125}}
        )


@pytest.mark.parametrize(
    "field",
    ["values", "coefficients", "moments", "buffer"],
)
def test_checkpoint_metadata_rejects_numeric_arrays_outside_shape(
    field: str,
) -> None:
    with pytest.raises(ValueError, match=rf"checkpoint_metadata\.{field}"):
        audit_gate_artifact({"checkpoint_metadata": {field: [0.1, 0.2]}})


@pytest.mark.parametrize(
    "tensors",
    [
        {},
        [0.1, 0.2],
        [{}],
        [{"name": "weight", "shape": [2], "dtype": "float32", "values": [1.0]}],
        [{"name": "", "shape": [2], "dtype": "float32"}],
        [{"name": "weight", "shape": [-1], "dtype": "float32"}],
        [{"name": "weight", "shape": [True], "dtype": "float32"}],
        [{"name": "weight", "shape": [2], "dtype": "object"}],
        [
            {
                "name": "weight",
                "shape": [2],
                "dtype": "float32",
                "sha256": "not-a-digest",
            }
        ],
        [
            {
                "name": "weight",
                "shape": [2, 3],
                "dtype": "float32",
                "numel": 5,
            }
        ],
    ],
)
def test_checkpoint_tensor_metadata_has_an_exact_non_payload_schema(
    tensors: object,
) -> None:
    with pytest.raises(ValueError, match=r"checkpoint_metadata\.tensors"):
        audit_gate_artifact({"checkpoint_metadata": {"tensors": tensors}})


def test_tensor_shape_rejects_total_numel_over_hard_limit_when_numel_omitted() -> None:
    with pytest.raises(ValueError, match=r"tensors\[0\]\.shape.*numel"):
        audit_gate_artifact(
            {
                "checkpoint_metadata": {
                    "tensors": [
                        {
                            "name": "too-large",
                            "shape": [65_536, 65_536],
                            "dtype": "float32",
                        }
                    ]
                }
            }
        )


@pytest.mark.parametrize(
    ("shape", "numel"),
    [([], 1), ([0, 2**31 - 1], 0)],
)
def test_tensor_shape_scalar_and_zero_dimension_numel_contract(
    shape: list[int],
    numel: int,
) -> None:
    assert (
        audit_gate_artifact(
            {
                "checkpoint_metadata": {
                    "tensors": [
                        {
                            "name": "bounded",
                            "shape": shape,
                            "dtype": "float32",
                            "numel": numel,
                        }
                    ]
                }
            }
        )
        is None
    )


def test_checkpoint_tensor_metadata_accepts_only_shape_dtype_and_digests() -> None:
    assert (
        audit_gate_artifact(
            {
                "checkpoint_metadata": {
                    "epoch": 2,
                    "checkpoint_sha256": "a" * 64,
                    "model_sha256": "c" * 64,
                    "model_format_version": "stable-gate/v1",
                    "optimizer_steps": 30,
                    "parameter_count": 1536,
                    "tensor_count": 1,
                    "tensors": [
                        {
                            "name": "classifier.weight",
                            "shape": [2, 768],
                            "dtype": "float32",
                            "sha256": "b" * 64,
                            "numel": 1536,
                        }
                    ],
                }
            }
        )
        is None
    )


def test_task9_checkpoint_contract_enters_metadata_only_mode() -> None:
    with pytest.raises(ValueError, match=r"model_state.*metadata only"):
        audit_gate_artifact(
            {
                "contract_version": "wfcllm-gate-training-checkpoint/v1",
                "config_hash": "a" * 64,
                "dataset_manifest_hash": "b" * 64,
                "epoch": 1,
                "model_state": {"classifier.weight": [0.1, 0.2]},
            }
        )


def test_task9_checkpoint_safe_metadata_names_remain_accepted() -> None:
    assert (
        audit_gate_artifact(
            {
                "contract_version": "wfcllm-gate-training-checkpoint-metadata/v1",
                "config_hash": "a" * 64,
                "dataset_manifest_hash": "b" * 64,
                "epoch": 1,
                "validation": {
                    "coverage": 0.9,
                    "decision_consistency": 1.0,
                },
                "model_state_sha256": "c" * 64,
                "optimizer_state_sha256": "d" * 64,
                "scheduler_state_sha256": "e" * 64,
                "tensors": [
                    {
                        "name": "classifier.weight",
                        "shape": [2],
                        "dtype": "torch.float32",
                        "numel": 2,
                    }
                ],
            }
        )
        is None
    )


def test_diagnostic_checkpoint_metadata_cannot_carry_runtime_state() -> None:
    with pytest.raises(ValueError, match=r"checkpoint_metadata\.weights"):
        audit_gate_artifact(
            {
                "diagnostic_test_backend": True,
                "formal_eligible": False,
                "checkpoint_metadata": {"weights": [0.1, 0.2]},
            }
        )


def test_gate_artifact_rejects_oversized_field_name_without_echoing_values() -> None:
    secret_value = "never-echo-this-secret-value"
    with pytest.raises(ValueError, match="field name") as exc_info:
        audit_gate_artifact({"x" * 1025: secret_value})

    assert secret_value not in str(exc_info.value)


def test_reject_secret_key_leak_uses_cycle_safe_common_walker() -> None:
    payload: dict[str, object] = {}
    payload["cycle"] = payload

    with pytest.raises(ValueError, match="cyclic"):
        reject_secret_key_leak(payload)


def test_reject_secret_key_leak_uses_same_plural_rules_and_depth_budget() -> None:
    with pytest.raises(ValueError, match="rawTrainingKeys"):
        reject_secret_key_leak({"nested": [{"rawTrainingKeys": "hidden"}]})

    payload: dict[str, object] = {}
    cursor = payload
    for index in range(80):
        child: dict[str, object] = {}
        cursor[f"level_{index}"] = child
        cursor = child
    with pytest.raises(ValueError, match="nesting depth"):
        reject_secret_key_leak(payload)


def test_gate_artifact_rejects_non_string_mapping_keys() -> None:
    with pytest.raises(ValueError, match=r"metadata.*string field names"):
        audit_gate_artifact({"metadata": {1: "not-a-public-JSON-field"}})
