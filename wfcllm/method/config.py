from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

from wfcllm.datasets.constants import SUPPORTED_DATASETS


_ALLOWED_TORCH_DTYPES = ("auto", "fp32", "fp16", "bf16")
_ALLOWED_PROMPT_MODES = ("completion", "chat")
_ALLOWED_RULE_NAMES = ("hash", "semantic_lsh")
DEFAULT_HUMANEVAL_STOP_SEQUENCES = ("\nclass", "\ndef", "\n#", "\nif", "\nprint")
_EVIDENCE_RETRY_METHOD = "evidence_retry_seed7x3"
_GATED_METHOD = "gated_semantic_window_v1"
_GATED_PHASES = (
    "gate-data",
    "gate-train",
    "gate-validate",
    "generate",
    "calibrate",
    "detect",
    "report",
)
_GATED_FAST_PHASES = (
    "gate-data",
    "gate-train",
    "generate",
    "calibrate",
    "detect",
    "report",
)
_MAIN_PHASES = ("generate", "calibrate", "detect", "report", "audit")
_GATED_MAIN_PHASES = ("generate", "calibrate", "detect", "report")
_SENSITIVE_PUBLIC_TOKENS = (
    "secret",
    "deployment_key",
    "raw_training_key",
    "raw_holdout_key",
    "raw_key",
    "key_material",
    "key_value",
    "key_bytes",
    "api_key",
    "private_key",
    "access_key",
)
_FAST_LOSSES = [
    "close_bce",
    "suitable_bce",
    "dangerous_negative_fp",
]
_FAST_LOSS_WEIGHTS = {
    "close_bce": 1.0,
    "suitable_bce": 1.0,
    "close_positive": 1.0,
    "suitable_positive": 1.0,
    "suitable_false_positive": 4.0,
}
_LABEL_THRESHOLDS = {
    "reliable_success_rate_r3_min": 0.60,
    "structurally_valid_rewrite_rate_r3_min": 2 / 3,
    "unstable_candidate_rate_r3_max": 0.10,
}
_FEASIBILITY_THRESHOLDS = {
    "pilot_independent_group_min": 100,
    "pilot_independent_group_max": 300,
    "full_independent_group_min": 300,
    "full_independent_group_max": 300,
    "pilot_suitable_positive_min": 10,
    "pilot_suitable_negative_min": 25,
    "full_suitable_positive_min": 30,
    "full_suitable_negative_min": 75,
    "window_length_group_min": 50,
    "major_statement_family_count_min": 4,
    "major_statement_family_group_min": 25,
    "r3_minus_r1_bootstrap_lower_95_exclusive_min": 0.0,
    "holdout_key_absolute_decline_max": 0.10,
    "validation_test_suitable_positive_min": 3,
    "validation_test_suitable_negative_min": 8,
}
_ACCEPTANCE_THRESHOLDS = {
    "decision_agreement_min": 0.999,
    "float_quantized_accepted_set_agreement_min": 0.999,
    "formal_accepted_span_consensus_min": 1.0,
    "suitable_false_positive_rate_max": 0.05,
}
_HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class _FrozenSequence(tuple[Any, ...]):
    """Tuple-backed config sequence with list-compatible equality."""

    def __eq__(self, other: object) -> bool:
        if isinstance(other, (list, tuple)):
            return tuple(self) == tuple(other)
        return False

    def __ne__(self, other: object) -> bool:
        return not self == other

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenSequence:
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (_rebuild_frozen_sequence, (_thaw_config(self),))

    __hash__ = tuple.__hash__


class _FrozenDict(Mapping[str, Any]):
    """Read-only mapping used to make validated gated configs immutable."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(value)))

    def __setattr__(self, name: str, value: Any) -> None:
        raise TypeError("frozen config mappings cannot be modified")

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Mapping) and dict(self.items()) == dict(other.items())

    def __repr__(self) -> str:
        return f"_FrozenDict({self._data!r})"

    def __deepcopy__(self, memo: dict[int, Any]) -> _FrozenDict:
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (_rebuild_frozen_dict, (_thaw_config(self),))


def _freeze_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict(
            {key: _freeze_config(nested) for key, nested in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return _FrozenSequence(_freeze_config(nested) for nested in value)
    return value


def _thaw_config(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_config(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_config(nested) for nested in value]
    return value


def _rebuild_frozen_sequence(value: list[Any]) -> _FrozenSequence:
    frozen = _freeze_config(value)
    if not isinstance(frozen, _FrozenSequence):
        raise ValueError("invalid frozen sequence payload")
    return frozen


def _rebuild_frozen_dict(value: dict[str, Any]) -> _FrozenDict:
    frozen = _freeze_config(value)
    if not isinstance(frozen, _FrozenDict):
        raise ValueError("invalid frozen mapping payload")
    return frozen


@dataclass(frozen=True)
class WFCLLMMethodPreset:
    method: Mapping[str, Any]
    generation: Mapping[str, Any] = field(default_factory=dict)
    semantic_lsh: Mapping[str, Any] = field(default_factory=dict)
    detector: Mapping[str, Any] = field(default_factory=dict)
    calibration: Mapping[str, Any] = field(default_factory=dict)
    artifacts: Mapping[str, Any] = field(default_factory=dict)
    runtime: Mapping[str, Any] = field(default_factory=dict)
    gate_data: Mapping[str, Any] = field(default_factory=dict)
    gate_train: Mapping[str, Any] = field(default_factory=dict)
    gate_validate: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        section_names = (
            "method",
            "generation",
            "semantic_lsh",
            "detector",
            "calibration",
            "gate_data",
            "gate_train",
            "gate_validate",
            "artifacts",
            "runtime",
        )
        for section_name in section_names:
            section = getattr(self, section_name)
            if not isinstance(section, Mapping):
                raise ValueError(f"{section_name} must be a dict")

        method_name = self.method.get("name")
        if method_name not in {_EVIDENCE_RETRY_METHOD, _GATED_METHOD}:
            raise ValueError(f"unsupported method.name: {method_name!r}")
        if self.method.get("strict_no_quality_gate") is not True:
            raise ValueError("method.strict_no_quality_gate must be true")
        if self.method.get("strict_code_only_detector") is not True:
            raise ValueError("method.strict_code_only_detector must be true")
        if method_name == _GATED_METHOD:
            for section_name in section_names:
                object.__setattr__(
                    self,
                    section_name,
                    _freeze_config(getattr(self, section_name)),
                )
            self._reject_public_secret_material(self.to_dict())
            self._validate_gated_method()
        else:
            self._require_exact_keys(
                self.method,
                {
                    "name",
                    "strict_no_quality_gate",
                    "strict_code_only_detector",
                },
                "method",
            )

        try:
            json.dumps(self.to_dict(), allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("method preset must be JSON-safe") from exc

    def __deepcopy__(self, memo: dict[int, Any]) -> WFCLLMMethodPreset:
        existing = memo.get(id(self))
        if existing is not None:
            return existing
        copied = _rebuild_method_preset(self.to_dict())
        memo[id(self)] = copied
        return copied

    def __reduce__(self):
        return (_rebuild_method_preset, (self.to_dict(),))

    def _validate_gated_method(self) -> None:
        self._require_exact_keys(
            self.method,
            {
                "name",
                "strict_no_quality_gate",
                "strict_code_only_detector",
                "experimental",
                "windowing",
                "gate",
                "rewrite",
                "semantic",
            },
            "method",
        )
        windowing = self._required_mapping(self.method, "windowing")
        gate = self._required_mapping(self.method, "gate")
        rewrite = self._required_mapping(self.method, "rewrite")
        semantic = self._required_mapping(self.method, "semantic")
        self._require_exact_keys(
            self.generation,
            {
                "dataset",
                "max_new_tokens",
                "temperature",
                "top_p",
                "top_k",
                "torch_dtype",
                "device",
                "seed",
                "load_in_4bit",
                "prompt_mode",
                "program_finalizer",
                "max_total_sampled_tokens",
            },
            "generation",
        )
        self._require_exact_keys(
            self.semantic_lsh,
            {
                "rule_name",
                "lsh_d",
                "lsh_gamma",
                "semantic_margin",
                "use_ordinal_keying",
            },
            "semantic_lsh",
        )
        self._require_exact_keys(
            self.detector,
            {
                "mode",
                "target_fpr",
                "minimum_reliable_windows",
                "statistic",
                "abstain_policy",
            },
            "detector",
        )
        self._require_exact_keys(
            self.calibration,
            {
                "method",
                "group_by",
                "target_fpr",
                "posthoc_pass_at_1_noninferiority_absolute_drop_max",
            },
            "calibration",
        )
        self._require_exact_keys(self.artifacts, {"run_root"}, "artifacts")
        self._require_exact_keys(
            self.runtime,
            {"default_phases", "external_validated_bundle_phases"},
            "runtime",
        )
        self._require_exact_keys(
            windowing,
            {
                "enabled",
                "contract_version",
                "max_units",
                "max_preceding_units",
                "excluded_statement_types",
                "compound_header_singleton",
            },
            "method.windowing",
        )
        self._require_exact_keys(
            gate,
            {
                "input_contract_version",
                "bundle_contract_version",
                "bundle_path",
                "bundle_sha256",
                "require_validated",
                "uncertain_boundary_policy",
                "max_input_tokens",
            },
            "method.gate",
        )
        self._require_exact_keys(
            rewrite,
            {
                "candidate_zero",
                "max_attempts",
                "experiment_budgets",
                "key_blind",
                "temperature",
                "top_p",
                "max_new_tokens",
                "generation_attempts",
                "candidate_selection",
            },
            "method.rewrite",
        )
        self._require_exact_keys(
            semantic,
            {"parent_descriptor_version", "encoder_id", "lsh"},
            "method.semantic",
        )
        semantic_lsh = self._required_mapping(semantic, "lsh")
        self._require_exact_keys(
            semantic_lsh,
            {"d", "gamma", "margin", "key_derivation_version"},
            "method.semantic.lsh",
        )

        if self.method.get("experimental") is not True:
            raise ValueError("method.experimental must be true")

        if windowing.get("enabled") is not True:
            raise ValueError("method.windowing.enabled must be true")
        if windowing.get("contract_version") != "python-statement-window/v1":
            raise ValueError("method.windowing.contract_version is incompatible")
        if windowing.get("max_units") != 3:
            raise ValueError("method.windowing.max_units must equal 3")
        if windowing.get("max_preceding_units") != 3:
            raise ValueError("method.windowing.max_preceding_units must equal 3")
        if windowing.get("compound_header_singleton") is not True:
            raise ValueError("method.windowing.compound_header_singleton must be true")
        excluded = windowing.get("excluded_statement_types")
        required_excluded = [
            "pass",
            "break",
            "continue",
            "raise",
            "import",
            "import_from",
            "global",
            "nonlocal",
            "delete",
            "assert",
            "function_definition_header",
            "class_definition_header",
            "parser_recovery",
        ]
        if excluded != required_excluded:
            raise ValueError(
                "method.windowing.excluded_statement_types must match the v1 contract"
            )

        if gate.get("input_contract_version") != "wfcllm-gate-input/v1":
            raise ValueError("method.gate.input_contract_version is incompatible")
        if gate.get("bundle_contract_version") != "wfcllm-gate-bundle/v1":
            raise ValueError("method.gate.bundle_contract_version is incompatible")
        if type(gate.get("require_validated")) is not bool:
            raise ValueError("method.gate.require_validated must be a bool")
        if gate.get("uncertain_boundary_policy") != "close_and_skip":
            raise ValueError(
                "method.gate.uncertain_boundary_policy must be close_and_skip"
            )
        if gate.get("max_input_tokens") != 256:
            raise ValueError("method.gate.max_input_tokens must equal 256")
        bundle_path = gate.get("bundle_path")
        bundle_sha256 = gate.get("bundle_sha256")
        if (bundle_path is None) != (bundle_sha256 is None):
            raise ValueError("method.gate bundle path and sha256 must be set together")
        if bundle_path is not None:
            if not isinstance(bundle_path, str) or not bundle_path:
                raise ValueError("method.gate.bundle_path must be a non-empty string")
            bundle_parts = Path(bundle_path).parts
            if "://" in bundle_path or "\x00" in bundle_path or ".." in bundle_parts:
                raise ValueError("method.gate.bundle_path must identify a local path")
            if (
                not isinstance(bundle_sha256, str)
                or len(bundle_sha256) != 64
                or any(character not in "0123456789abcdef" for character in bundle_sha256)
            ):
                raise ValueError("method.gate.bundle_sha256 must be lowercase SHA-256")

        self._require_fixed_values(
            rewrite,
            {
                "candidate_zero": "original_window",
                "max_attempts": 3,
                "experiment_budgets": [1, 3],
                "key_blind": True,
                "candidate_selection": "unique-key-blind-structural-fallback/v1",
            },
            "method.rewrite",
        )
        rewrite_temperature = rewrite.get("temperature")
        if (
            isinstance(rewrite_temperature, bool)
            or not isinstance(rewrite_temperature, (int, float))
            or rewrite_temperature <= 0
        ):
            raise ValueError("method.rewrite.temperature must be positive")
        rewrite_top_p = rewrite.get("top_p")
        if (
            isinstance(rewrite_top_p, bool)
            or not isinstance(rewrite_top_p, (int, float))
            or not 0 < rewrite_top_p <= 1
        ):
            raise ValueError("method.rewrite.top_p must be in (0, 1]")
        rewrite_max_new_tokens = rewrite.get("max_new_tokens")
        if type(rewrite_max_new_tokens) is not int or rewrite_max_new_tokens <= 0:
            raise ValueError("method.rewrite.max_new_tokens must be a positive integer")
        rewrite_generation_attempts = rewrite.get("generation_attempts")
        if (
            type(rewrite_generation_attempts) is not int
            or rewrite_generation_attempts < 3
        ):
            raise ValueError(
                "method.rewrite.generation_attempts must be an integer of at least 3"
            )

        if semantic.get("parent_descriptor_version") != "python-statement-window/v1":
            raise ValueError("method.semantic.parent_descriptor_version is incompatible")
        if semantic.get("encoder_id") != "semantic-encoder-local-v1":
            raise ValueError("method.semantic.encoder_id must match the v1 contract")
        if not isinstance(semantic.get("lsh"), Mapping):
            raise ValueError("method.semantic.lsh must be a dict")
        formal_semantic_lsh = gate.get("require_validated") is True
        self._require_fixed_values(
            semantic_lsh,
            {
                "d": 4 if formal_semantic_lsh else 1,
                "gamma": 0.25 if formal_semantic_lsh else 0.5,
                "margin": 0.0,
                "key_derivation_version": "wfcllm-parent-key/v1",
            },
            "method.semantic.lsh",
        )

        self._validate_reusable_sections()
        self._validate_gate_data()
        self._validate_gate_train()
        self._validate_gate_validate()

        expected_default_phases = (
            list(_GATED_MAIN_PHASES)
            if bundle_path is not None
            else list(_GATED_PHASES)
            if gate.get("require_validated") is True
            else list(_GATED_FAST_PHASES)
        )
        if self.runtime.get("default_phases") != expected_default_phases:
            phase_description = (
                "configured local phases" if bundle_path is None else "four main phases"
            )
            raise ValueError(
                f"gated runtime.default_phases must use the {phase_description}"
            )
        if self.runtime.get("external_validated_bundle_phases") != list(_GATED_MAIN_PHASES):
            raise ValueError(
                "gated runtime.external_validated_bundle_phases must use the four main phases"
            )

    def _validate_reusable_sections(self) -> None:
        self._require_fixed_values(
            self.generation,
            {
                "dataset": "humaneval",
                "max_new_tokens": 512,
                "temperature": 0.0,
                "top_p": 0.95,
                "top_k": 0,
                "torch_dtype": "bf16",
                "device": "cuda",
                "seed": 7,
                "load_in_4bit": True,
                "prompt_mode": "completion",
                "program_finalizer": "humaneval_target_function_v1",
                "max_total_sampled_tokens": 32768,
            },
            "generation",
        )
        formal_semantic_lsh = (
            isinstance(self.method.get("gate"), Mapping)
            and self.method["gate"].get("require_validated") is True
        )
        self._require_fixed_values(
            self.semantic_lsh,
            {
                "rule_name": (
                    "semantic_lsh" if formal_semantic_lsh else "keyed_text_region"
                ),
                "lsh_d": 4 if formal_semantic_lsh else 1,
                "lsh_gamma": 0.25 if formal_semantic_lsh else 0.5,
                "semantic_margin": 0.0,
                "use_ordinal_keying": False,
            },
            "semantic_lsh",
        )
        self._require_fixed_values(
            self.detector,
            {
                "mode": "wfcllm-gated-semantic-window/v1",
                "target_fpr": 0.05,
                "minimum_reliable_windows": 2,
                "statistic": "reliable_window_hit_rate",
                "abstain_policy": "exclude_from_denominator",
            },
            "detector",
        )
        self._require_fixed_values(
            self.calibration,
            {
                "method": "pooled_negative_binomial_right_tail",
                "group_by": "pooled_binomial_tail",
                "target_fpr": 0.05,
                "posthoc_pass_at_1_noninferiority_absolute_drop_max": 0.02,
            },
            "calibration",
        )
        run_root = self.artifacts.get("run_root")
        if (
            not isinstance(run_root, str)
            or not run_root
            or "://" in run_root
            or "\x00" in run_root
            or ".." in Path(run_root).parts
        ):
            raise ValueError("artifacts.run_root must identify a local directory")

    def _validate_gate_data(self) -> None:
        expected_keys = {
            "schema_version",
            "source_manifest_version",
            "split_contract_version",
            "sources",
            "human_eval_included",
            "scale",
            "pilot_independent_group_min",
            "pilot_independent_group_max",
            "full_independent_group_min",
            "full_independent_group_max",
            "learning_curve_group_counts",
            "window_lengths",
            "candidate_zero",
            "rewrite_count",
            "rewrite_budgets",
            "training_key_count",
            "training_key_bank_file_parameter",
            "training_key_bank_manifest_sha256",
            "training_key_bank_id",
            "holdout_key_count",
            "holdout_key_bank_file_parameter",
            "holdout_key_bank_manifest_sha256",
            "holdout_key_bank_id",
            "label_contract_version",
            "label_thresholds",
            "feasibility_contract_version",
            "feasibility_thresholds",
        }
        self._require_exact_keys(self.gate_data, expected_keys, "gate_data")
        fixed = {
            "schema_version": "wfcllm-gate-data/v1",
            "source_manifest_version": "wfcllm-gate-source-manifest/v1",
            "split_contract_version": "wfcllm-gate-split/v1",
            "sources": [
                "main_generation",
                "mbpp_train",
                "mbpp_validation",
                "oss_python",
                "parser_boundary",
            ],
            "human_eval_included": False,
            "pilot_independent_group_min": 100,
            "pilot_independent_group_max": 300,
            "full_independent_group_min": 300,
            "full_independent_group_max": 300,
            "learning_curve_group_counts": ["full"],
            "window_lengths": [1, 2, 3],
            "candidate_zero": "original_window",
            "rewrite_count": 3,
            "rewrite_budgets": [1, 3],
            "training_key_count": 32,
            "training_key_bank_file_parameter": "training_key_bank_file",
            "holdout_key_count": 8,
            "holdout_key_bank_file_parameter": "holdout_key_bank_file",
            "label_contract_version": "wfcllm-gate-label/v1",
            "label_thresholds": _LABEL_THRESHOLDS,
            "feasibility_contract_version": "gate-data-feasibility/v1",
            "feasibility_thresholds": _FEASIBILITY_THRESHOLDS,
        }
        self._require_fixed_values(self.gate_data, fixed, "gate_data")
        if self.gate_data.get("scale") not in {"pilot", "full"}:
            raise ValueError("gate_data.scale must be pilot or full")
        self._validate_digest_or_none(
            self.gate_data["training_key_bank_manifest_sha256"],
            "gate_data.training_key_bank_manifest_sha256",
        )
        self._validate_digest_or_none(
            self.gate_data["holdout_key_bank_manifest_sha256"],
            "gate_data.holdout_key_bank_manifest_sha256",
        )
        self._validate_bank_id_or_none(
            self.gate_data["training_key_bank_id"],
            prefix="training-key-bank/v1:sha256:",
            field_name="gate_data.training_key_bank_id",
        )
        self._validate_bank_id_or_none(
            self.gate_data["holdout_key_bank_id"],
            prefix="holdout-key-bank/v1:sha256:",
            field_name="gate_data.holdout_key_bank_id",
        )

    def _validate_gate_train(self) -> None:
        expected_keys = {
            "model_contract_version",
            "base_encoder_id",
            "parameter_count_min",
            "parameter_count_max",
            "max_tokens",
            "optimizer",
            "learning_rate",
            "max_epochs",
            "early_stopping_patience",
            "losses",
            "loss_weights",
        }
        self._require_exact_keys(self.gate_train, expected_keys, "gate_train")
        fixed = {
            "model_contract_version": "wfcllm-gate-model-state/v1",
            "parameter_count_min": 30000000,
            "parameter_count_max": 80000000,
            "max_tokens": 256,
            "optimizer": "adamw",
            "learning_rate": 0.00002,
            "max_epochs": 4,
            "early_stopping_patience": 1,
            "losses": _FAST_LOSSES,
            "loss_weights": _FAST_LOSS_WEIGHTS,
        }
        self._require_fixed_values(self.gate_train, fixed, "gate_train")
        base_encoder_id = self.gate_train.get("base_encoder_id")
        base_encoder_path = (
            Path(base_encoder_id)
            if isinstance(base_encoder_id, str) and base_encoder_id
            else None
        )
        if (
            not isinstance(base_encoder_id, str)
            or not base_encoder_id
            or "://" in base_encoder_id
            or "\x00" in base_encoder_id
            or ".." in Path(base_encoder_id).parts
            or base_encoder_path is None
            or base_encoder_path.parts[:2] != ("data", "models")
            or len(base_encoder_path.parts) < 3
        ):
            raise ValueError("gate_train.base_encoder_id must identify a local model")

    def _validate_gate_validate(self) -> None:
        expected_keys = {
            "contract_version",
            "holdout_key_count",
            "holdout_key_bank_file_parameter",
            "threshold_fit_grouped",
            "agreement_subset_disjoint",
            "batch_sizes",
            "orders",
            "cpu_precisions",
            "gpu_float_if_available",
            "independent_reloads",
            "formal_quantization",
            "max_input_tokens",
            "acceptance_thresholds",
        }
        self._require_exact_keys(self.gate_validate, expected_keys, "gate_validate")
        fixed = {
            "contract_version": "wfcllm-gate-validation/v1",
            "holdout_key_count": 8,
            "holdout_key_bank_file_parameter": "holdout_key_bank_file",
            "threshold_fit_grouped": True,
            "agreement_subset_disjoint": True,
            "batch_sizes": [1],
            "orders": ["original"],
            "cpu_precisions": ["float", "dynamic_qint8"],
            "gpu_float_if_available": False,
            "independent_reloads": 1,
            "formal_quantization": "torch-dynamic-qint8-linear",
            "max_input_tokens": 512,
            "acceptance_thresholds": _ACCEPTANCE_THRESHOLDS,
        }
        self._require_fixed_values(self.gate_validate, fixed, "gate_validate")

    @staticmethod
    def _required_mapping(
        parent: Mapping[str, Any], name: str
    ) -> Mapping[str, Any]:
        value = parent.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"method.{name} must be a dict")
        return value

    @staticmethod
    def _require_exact_keys(
        value: Mapping[str, Any],
        expected: set[str],
        section_name: str,
    ) -> None:
        actual = set(value)
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        if missing:
            raise ValueError(f"{section_name} is missing fields: {missing}")
        if unknown:
            raise ValueError(f"{section_name} has unknown fields: {unknown}")

    @staticmethod
    def _require_fixed_values(
        value: Mapping[str, Any],
        expected: dict[str, Any],
        section_name: str,
    ) -> None:
        for field_name, required in expected.items():
            actual = value.get(field_name)
            if not WFCLLMMethodPreset._typed_equal(actual, required):
                raise ValueError(
                    f"{section_name}.{field_name} must equal {required!r}"
                )

    @staticmethod
    def _typed_equal(actual: object, required: object) -> bool:
        if isinstance(required, dict):
            if not isinstance(actual, Mapping):
                return False
            if set(actual) != set(required):
                return False
            return all(
                WFCLLMMethodPreset._typed_equal(actual[key], expected_value)
                for key, expected_value in required.items()
            )
        if isinstance(required, list):
            if not isinstance(actual, tuple):
                return False
            return len(actual) == len(required) and all(
                WFCLLMMethodPreset._typed_equal(candidate, expected_value)
                for candidate, expected_value in zip(actual, required)
            )
        return type(actual) is type(required) and actual == required

    @staticmethod
    def _validate_digest_or_none(value: object, field_name: str) -> None:
        if value is not None and (
            not isinstance(value, str) or _HEX_DIGEST_PATTERN.fullmatch(value) is None
        ):
            raise ValueError(f"{field_name} must be null or lowercase SHA-256")

    @staticmethod
    def _validate_bank_id_or_none(
        value: object,
        *,
        prefix: str,
        field_name: str,
    ) -> None:
        if value is None:
            return
        if (
            not isinstance(value, str)
            or not value.startswith(prefix)
            or _HEX_DIGEST_PATTERN.fullmatch(value[len(prefix) :]) is None
        ):
            raise ValueError(f"{field_name} must be an irreversible bank ID")

    @classmethod
    def _reject_public_secret_material(cls, value: object) -> None:
        def visit(candidate: object) -> None:
            if isinstance(candidate, Mapping):
                for key, nested in candidate.items():
                    if not isinstance(key, str):
                        raise ValueError("gated public config keys must be strings")
                    cls._reject_sensitive_text(key)
                    visit(nested)
            elif isinstance(candidate, (list, tuple)):
                for nested in candidate:
                    visit(nested)

        visit(value)

    @staticmethod
    def _reject_sensitive_text(value: str) -> None:
        camel_split = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", value)
        camel_split = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", camel_split)
        normalized = re.sub(
            r"[^a-z0-9]+", "_", camel_split.lower()
        ).strip("_")
        compact = re.sub(r"[^a-z0-9]+", "", normalized)
        compact_tokens = tuple(token.replace("_", "") for token in _SENSITIVE_PUBLIC_TOKENS)
        if normalized in {"key", "keys"} or compact in {"key", "keys"} or any(
            token in normalized or compact_token in compact
            for token, compact_token in zip(
                _SENSITIVE_PUBLIC_TOKENS,
                compact_tokens,
            )
        ):
            raise ValueError("gated public config must not contain secret material")

    def to_dict(self) -> dict[str, Any]:
        if self.method.get("name") == _EVIDENCE_RETRY_METHOD:
            payload = asdict(self)
            for section_name in ("gate_data", "gate_train", "gate_validate"):
                payload.pop(section_name)
            return payload
        payload = {
            "method": _thaw_config(self.method),
            "generation": _thaw_config(self.generation),
            "semantic_lsh": _thaw_config(self.semantic_lsh),
            "detector": _thaw_config(self.detector),
            "calibration": _thaw_config(self.calibration),
            "artifacts": _thaw_config(self.artifacts),
            "runtime": _thaw_config(self.runtime),
            "gate_data": _thaw_config(self.gate_data),
            "gate_train": _thaw_config(self.gate_train),
            "gate_validate": _thaw_config(self.gate_validate),
        }
        return payload


def _rebuild_method_preset(payload: dict[str, Any]) -> WFCLLMMethodPreset:
    if not isinstance(payload, dict):
        raise ValueError("method preset pickle payload must be a dict")
    return WFCLLMMethodPreset(**payload)


@dataclass(frozen=True)
class SawrGenerationConfig:
    """Local causal-LM generation settings for the SAWR smoke runner."""

    model_path: str
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    retry_repetition_penalty: float = 1.0
    torch_dtype: str = "auto"
    device: str = "cuda"
    seed: int = 0
    load_in_4bit: bool = False
    eos_token_id: int | None = None
    prompt_mode: str = "completion"
    stop_sequences: tuple[str, ...] = DEFAULT_HUMANEVAL_STOP_SEQUENCES

    def __post_init__(self) -> None:
        if not Path(self.model_path).exists():
            raise ValueError(f"model_path does not exist: {self.model_path}")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if (
            isinstance(self.retry_repetition_penalty, bool)
            or not isinstance(self.retry_repetition_penalty, (int, float))
            or not math.isfinite(float(self.retry_repetition_penalty))
        ):
            raise ValueError("retry_repetition_penalty must be a finite number")
        if self.retry_repetition_penalty < 1.0:
            raise ValueError("retry_repetition_penalty must be >= 1.0")
        if self.torch_dtype not in _ALLOWED_TORCH_DTYPES:
            raise ValueError(
                f"torch_dtype must be one of {_ALLOWED_TORCH_DTYPES}, got {self.torch_dtype!r}"
            )
        object.__setattr__(
            self,
            "retry_repetition_penalty",
            float(self.retry_repetition_penalty),
        )
        if self.prompt_mode not in _ALLOWED_PROMPT_MODES:
            raise ValueError(
                f"prompt_mode must be one of {_ALLOWED_PROMPT_MODES}, got {self.prompt_mode!r}"
            )
        if isinstance(self.stop_sequences, str) or not isinstance(
            self.stop_sequences,
            (tuple, list),
        ):
            raise ValueError("stop_sequences must be a sequence of strings")
        if any(
            not isinstance(stop_sequence, str) or not stop_sequence
            for stop_sequence in self.stop_sequences
        ):
            raise ValueError("stop_sequences entries must be non-empty strings")
        object.__setattr__(self, "stop_sequences", tuple(self.stop_sequences))


@dataclass(frozen=True)
class SawrRuleConfig:
    """Embedding rule settings for deterministic SAWR smoke decisions."""

    rule_name: str = "hash"
    target_accept_rate: float = 0.5
    parameters: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rule_name not in _ALLOWED_RULE_NAMES:
            raise ValueError(
                f"rule_name must be one of {_ALLOWED_RULE_NAMES}, got {self.rule_name!r}"
            )
        if not 0 <= self.target_accept_rate <= 1:
            raise ValueError("target_accept_rate must be in [0, 1]")
        if not isinstance(self.parameters, dict):
            raise ValueError("parameters must be a dict")
        try:
            parameters_json = json.dumps(self.parameters, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("parameters must be JSON-serializable") from exc
        object.__setattr__(self, "parameters", json.loads(parameters_json))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SawrPipelineConfig:
    """Top-level SAWR smoke pipeline settings."""

    dataset: str
    dataset_path: str
    output_dir: str
    generation: SawrGenerationConfig
    rule: SawrRuleConfig = field(default_factory=SawrRuleConfig)
    sample_limit: int | None = None
    sample_offset: int | None = None
    max_group_statements: int = 2
    retry_budget: int = 1
    statement_retry_budget: int | None = None
    window_retry_budget: int | None = None
    compound_retry_budget: int | None = None
    global_rollback_budget: int | None = None
    max_total_sampled_tokens: int | None = None
    evidence_retry_attempts: int = 1
    evidence_retry_seed_stride: int = 1009
    resume: str | None = None
    candidate_sidecar_output: str | None = None
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.generation, SawrGenerationConfig):
            raise ValueError("generation must be SawrGenerationConfig")
        if not isinstance(self.rule, SawrRuleConfig):
            raise ValueError("rule must be SawrRuleConfig")
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )
        if self.sample_limit is not None and self.sample_limit < 0:
            raise ValueError("sample_limit must be non-negative")
        if self.sample_offset is not None and self.sample_offset < 0:
            raise ValueError("sample_offset must be non-negative")
        if self.max_group_statements <= 0:
            raise ValueError("max_group_statements must be positive")
        if self.retry_budget < 0:
            raise ValueError("retry_budget must be non-negative")
        if self.statement_retry_budget is not None and self.statement_retry_budget < 0:
            raise ValueError("statement_retry_budget must be non-negative")
        if self.window_retry_budget is not None and self.window_retry_budget < 0:
            raise ValueError("window_retry_budget must be non-negative")
        if self.compound_retry_budget is not None and self.compound_retry_budget < 0:
            raise ValueError("compound_retry_budget must be non-negative")
        retry_budget_for_limits = sum(
            (
                0
                if self.statement_retry_budget is None
                else self.statement_retry_budget,
                0 if self.window_retry_budget is None else self.window_retry_budget,
                (
                    self.retry_budget
                    if self.compound_retry_budget is None
                    else self.compound_retry_budget
                ),
            )
        )
        if self.global_rollback_budget is None:
            object.__setattr__(self, "global_rollback_budget", retry_budget_for_limits)
        elif self.global_rollback_budget < 0:
            raise ValueError("global_rollback_budget must be non-negative")
        if self.max_total_sampled_tokens is None:
            derived_budget = self.generation.max_new_tokens * max(
                2,
                int(self.global_rollback_budget) + 2,
            )
            object.__setattr__(self, "max_total_sampled_tokens", derived_budget)
        elif self.max_total_sampled_tokens <= 0:
            raise ValueError("max_total_sampled_tokens must be positive")
        if self.evidence_retry_attempts <= 0:
            raise ValueError("evidence_retry_attempts must be positive")
        if self.evidence_retry_seed_stride <= 0:
            raise ValueError("evidence_retry_seed_stride must be positive")
        if self.resume is not None and self.resume != "latest":
            raise ValueError("resume must be None or 'latest'")
        if self.candidate_sidecar_output is not None and not isinstance(
            self.candidate_sidecar_output,
            str,
        ):
            raise ValueError("candidate_sidecar_output must be a string or None")
        if self.run_id is not None and (
            not isinstance(self.run_id, str) or not self.run_id
        ):
            raise ValueError("run_id must be a non-empty string or None")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


WFCLLMGenerationConfig = SawrGenerationConfig
WFCLLMRuleConfig = SawrRuleConfig
WFCLLMPipelineConfig = SawrPipelineConfig
SawrGenerationConfig = WFCLLMGenerationConfig
SawrRuleConfig = WFCLLMRuleConfig
SawrPipelineConfig = WFCLLMPipelineConfig
