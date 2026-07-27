from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

_GATED_METHOD = "gated_semantic_window_v1"
_GATED_PHASES = (
    "encoder",
    "gate-data",
    "gate-train",
    "generate",
    "calibrate",
    "detect",
    "report",
    "audit",
)
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
_GATE_LOSSES = [
    "close_bce",
    "suitable_bce",
    "dangerous_negative_fp",
]
_GATE_LOSS_WEIGHTS = {
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
_HEX_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class _FrozenSequence(tuple[Any, ...]):
    """Tuple-backed config sequence equal to the same list or tuple values."""

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

    def __post_init__(self) -> None:
        section_names = (
            "method",
            "generation",
            "semantic_lsh",
            "detector",
            "calibration",
            "gate_data",
            "gate_train",
            "artifacts",
            "runtime",
        )
        for section_name in section_names:
            section = getattr(self, section_name)
            if not isinstance(section, Mapping):
                raise ValueError(f"{section_name} must be a dict")

        method_name = self.method.get("name")
        if method_name != _GATED_METHOD:
            raise ValueError(f"unsupported method.name: {method_name!r}")
        if self.method.get("strict_no_quality_gate") is not True:
            raise ValueError("method.strict_no_quality_gate must be true")
        if self.method.get("strict_code_only_detector") is not True:
            raise ValueError("method.strict_code_only_detector must be true")
        for section_name in section_names:
            object.__setattr__(
                self,
                section_name,
                _freeze_config(getattr(self, section_name)),
            )
        self._reject_public_secret_material(self.to_dict())
        self._validate_gated_method()

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
                "evidence_channels",
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
        calibration_required_keys = {
            "method",
            "group_by",
            "target_fpr",
        }
        calibration_optional_keys = {
            "target_negative_count",
            "supplement",
        }
        self._require_exact_keys(
            self.calibration,
            calibration_required_keys
            | (calibration_optional_keys & set(self.calibration)),
            "calibration",
        )
        self._require_exact_keys(self.artifacts, {"run_root"}, "artifacts")
        self._require_exact_keys(
            self.runtime,
            {"default_phases"},
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
                "candidate_contract_version",
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
        formal_semantic_lsh = self._formal_semantic_lsh()
        semantic_keys = {"parent_descriptor_version", "encoder_id", "lsh"}
        if formal_semantic_lsh:
            semantic_keys.add("preservation")
        self._require_exact_keys(semantic, semantic_keys, "method.semantic")
        semantic_lsh = self._required_mapping(semantic, "lsh")
        self._require_exact_keys(
            semantic_lsh,
            {"d", "gamma", "margin", "key_derivation_version"},
            "method.semantic.lsh",
        )

        if windowing.get("enabled") is not True:
            raise ValueError("method.windowing.enabled must be true")
        if windowing.get("contract_version") != "python-statement-window/v1":
            raise ValueError("method.windowing.contract_version is incompatible")
        max_units = windowing.get("max_units")
        if (
            type(max_units) is not int
            or not 1 <= max_units <= 3
        ):
            raise ValueError("method.windowing.max_units must be an integer from 1 through 3")
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
        if (
            gate.get("candidate_contract_version")
            != "wfcllm-gate-train-candidate/v1"
        ):
            raise ValueError(
                "method.gate.candidate_contract_version is incompatible"
            )
        if gate.get("uncertain_boundary_policy") != "close_and_skip":
            raise ValueError(
                "method.gate.uncertain_boundary_policy must be close_and_skip"
            )
        if gate.get("max_input_tokens") != 256:
            raise ValueError("method.gate.max_input_tokens must equal 256")
        self._require_fixed_values(
            rewrite,
            {
                "candidate_zero": "original_window",
                "experiment_budgets": [1, 3],
                "key_blind": True,
                "candidate_selection": "fixed-key-blind-abc-trajectory/v1",
            },
            "method.rewrite",
        )
        max_attempts = rewrite.get("max_attempts")
        if (
            type(max_attempts) is not int
            or not 3 <= max_attempts <= 48
        ):
            raise ValueError(
                "method.rewrite.max_attempts must be an integer from 3 through 48"
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
        if type(rewrite_generation_attempts) is not int or rewrite_generation_attempts != 3:
            raise ValueError(
                "method.rewrite.generation_attempts must equal 3"
            )

        if semantic.get("parent_descriptor_version") != "python-statement-window/v1":
            raise ValueError("method.semantic.parent_descriptor_version is incompatible")
        if semantic.get("encoder_id") != "semantic-encoder-local-v1":
            raise ValueError("method.semantic.encoder_id must match the v1 contract")
        if not isinstance(semantic.get("lsh"), Mapping):
            raise ValueError("method.semantic.lsh must be a dict")
        self._require_fixed_values(
            semantic_lsh,
            {
                "d": 12 if formal_semantic_lsh else 1,
                "gamma": 0.45 if formal_semantic_lsh else 0.5,
                "margin": 0.0,
                "key_derivation_version": "wfcllm-parent-key/v1",
            },
            "method.semantic.lsh",
        )
        if formal_semantic_lsh:
            preservation = self._required_mapping(semantic, "preservation")
            self._require_exact_keys(
                preservation,
                {"rule", "threshold"},
                "method.semantic.preservation",
            )
            self._require_fixed_values(
                preservation,
                {
                    "rule": "codet5-cosine-to-original/v1",
                    "threshold": 0.9,
                },
                "method.semantic.preservation",
            )

        self._validate_reusable_sections()
        self._validate_gate_data()
        self._validate_gate_train()

        if self.runtime.get("default_phases") != list(_GATED_PHASES):
            raise ValueError(
                "gated runtime.default_phases must use the full reproduction chain"
            )

    def _formal_semantic_lsh(self) -> bool:
        """Whether the config declares the formal semantic-LSH evidence rule."""

        rule_name = self.semantic_lsh.get("rule_name")
        if rule_name != "semantic_lsh":
            raise ValueError("semantic_lsh.rule_name must be semantic_lsh")
        return True

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
        formal_semantic_lsh = self._formal_semantic_lsh()
        self._require_fixed_values(
            self.semantic_lsh,
            {
                "rule_name": "semantic_lsh",
                "lsh_d": 12 if formal_semantic_lsh else 1,
                "lsh_gamma": 0.45 if formal_semantic_lsh else 0.5,
                "semantic_margin": 0.0,
                "use_ordinal_keying": False,
            },
            "semantic_lsh",
        )
        evidence_channels = self.semantic_lsh.get("evidence_channels")
        if (
            isinstance(evidence_channels, bool)
            or not isinstance(evidence_channels, int)
            or not 1 <= evidence_channels <= 4
        ):
            raise ValueError(
                "semantic_lsh.evidence_channels must be an integer in [1, 4]"
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
                "target_fpr": 0.05,
            },
            "calibration",
        )
        calibration_contract = (
            self.calibration.get("method"),
            self.calibration.get("group_by"),
        )
        if calibration_contract not in {
            (
                "pooled_negative_binomial_right_tail",
                "pooled_binomial_tail",
            ),
            (
                "pooled_negative_empirical_right_tail",
                "pooled_reliable_hit_rate",
            ),
            (
                "pooled_negative_empirical_binomial_surprisal",
                "pooled_empirical_binomial_surprisal",
            ),
            (
                "pooled_negative_empirical_standardized_hit_surplus",
                "pooled_empirical_standardized_hit_surplus",
            ),
        }:
            raise ValueError(
                "calibration method/group_by must select one supported "
                "predeclared 5% FPR statistic"
            )
        self._validate_calibration_negative_supplement()
        run_root = self.artifacts.get("run_root")
        if (
            not isinstance(run_root, str)
            or not run_root
            or "://" in run_root
            or "\x00" in run_root
            or ".." in Path(run_root).parts
        ):
            raise ValueError("artifacts.run_root must identify a local directory")

    def _validate_calibration_negative_supplement(self) -> None:
        """Validate the optional autonomous negative-supplement keys (ADR 0008)."""

        target_negative_count = self.calibration.get("target_negative_count")
        if target_negative_count is not None and (
            type(target_negative_count) is not int or target_negative_count < 1
        ):
            raise ValueError(
                "calibration.target_negative_count must be a positive integer"
            )
        supplement = self.calibration.get("supplement")
        if supplement is None:
            return
        if not isinstance(supplement, Mapping):
            raise ValueError("calibration.supplement must be a dict")
        allowed_keys = {"max_new_tokens", "temperature", "top_p", "seed"}
        unknown = sorted(set(supplement) - allowed_keys)
        if unknown:
            raise ValueError(
                f"calibration.supplement has unknown fields: {unknown}"
            )
        max_new_tokens = supplement.get("max_new_tokens")
        if max_new_tokens is not None and (
            type(max_new_tokens) is not int or max_new_tokens < 1
        ):
            raise ValueError(
                "calibration.supplement.max_new_tokens must be a positive integer"
            )
        temperature = supplement.get("temperature")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or temperature < 0
        ):
            raise ValueError(
                "calibration.supplement.temperature must be a non-negative number"
            )
        top_p = supplement.get("top_p")
        if top_p is not None and (
            isinstance(top_p, bool)
            or not isinstance(top_p, (int, float))
            or not 0 < top_p <= 1
        ):
            raise ValueError("calibration.supplement.top_p must be in (0, 1]")
        seed = supplement.get("seed")
        if seed is not None and type(seed) is not int:
            raise ValueError("calibration.supplement.seed must be an integer")

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
            "losses": _GATE_LOSSES,
            "loss_weights": _GATE_LOSS_WEIGHTS,
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
        return {
            "method": _thaw_config(self.method),
            "generation": _thaw_config(self.generation),
            "semantic_lsh": _thaw_config(self.semantic_lsh),
            "detector": _thaw_config(self.detector),
            "calibration": _thaw_config(self.calibration),
            "artifacts": _thaw_config(self.artifacts),
            "runtime": _thaw_config(self.runtime),
            "gate_data": _thaw_config(self.gate_data),
            "gate_train": _thaw_config(self.gate_train),
        }


def _rebuild_method_preset(payload: dict[str, Any]) -> WFCLLMMethodPreset:
    if not isinstance(payload, dict):
        raise ValueError("method preset pickle payload must be a dict")
    return WFCLLMMethodPreset(**payload)
