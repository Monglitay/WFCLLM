"""Strict Supplementary Ablation specification and resolved identity."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any


SUPPLEMENTARY_ABLATION_SPEC_VERSION = "wfcllm-supplementary-ablation-spec/v1"
SUPPLEMENTARY_ABLATION_IDENTITY_VERSION = (
    "wfcllm-supplementary-ablation-identity/v1"
)
SUPPLEMENTARY_ABLATION_STUDY_KIND = "supplementary_ablation"

_FAMILY_ID = re.compile(r"[a-z0-9](?:[a-z0-9._-]{0,126}[a-z0-9])?\Z")
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_MAX_SPEC_BYTES = 64 * 1024
_PAPER_COVERED_FACTORS = frozenset(
    {
        "B",
        "gamma",
        "generator_scale",
        "model_scale",
        "DIPPER",
        "variable_renaming",
        "renaming",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "schema_version",
        "study_kind",
        "family_id",
        "factor",
        "level",
        "default_level",
        "canonical_baseline_config_hash",
        "language",
        "dataset",
        "profile",
    }
)

SUPPLEMENTARY_ABLATION_LEVELS: Mapping[str, tuple[int | float, ...]] = (
    MappingProxyType(
        {
            "d": (8, 12, 14),
            "delta": (0.0, 0.005, 0.01, 0.02),
            "tau_suit": (0.3, 0.5, 0.7),
            "tau_close": (0.4, 0.5, 0.6),
            "n_min": (1, 2, 3, 4),
            "max_units": (1, 2, 3),
        }
    )
)
SUPPLEMENTARY_ABLATION_DEFAULTS: Mapping[str, int | float] = MappingProxyType(
    {
        "d": 12,
        "delta": 0.0,
        "tau_suit": 0.5,
        "tau_close": 0.5,
        "n_min": 2,
        "max_units": 3,
    }
)


@dataclass(frozen=True)
class SupplementaryAblationSpec:
    """One immutable, one-factor-at-a-time study request."""

    family_id: str
    factor: str
    level: int | float
    default_level: int | float
    canonical_baseline_config_hash: str
    language: str = "python"
    dataset: str = "humaneval"
    profile: str = "full"
    schema_version: str = SUPPLEMENTARY_ABLATION_SPEC_VERSION
    study_kind: str = SUPPLEMENTARY_ABLATION_STUDY_KIND

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object]
    ) -> SupplementaryAblationSpec:
        if not isinstance(value, Mapping):
            raise ValueError("Supplementary Ablation Spec must be an object")
        if set(value) != _SPEC_FIELDS:
            unknown = sorted(set(value) - _SPEC_FIELDS)
            missing = sorted(_SPEC_FIELDS - set(value))
            raise ValueError(
                "Supplementary Ablation Spec schema mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("Supplementary Ablation Spec fields are invalid") from exc

    def __post_init__(self) -> None:
        if self.schema_version != SUPPLEMENTARY_ABLATION_SPEC_VERSION:
            raise ValueError(
                "Supplementary Ablation Spec schema_version must be "
                f"{SUPPLEMENTARY_ABLATION_SPEC_VERSION}"
            )
        if self.study_kind != SUPPLEMENTARY_ABLATION_STUDY_KIND:
            raise ValueError(
                "Supplementary Ablation Spec study_kind must be "
                "supplementary_ablation"
            )
        if not isinstance(self.family_id, str) or _FAMILY_ID.fullmatch(
            self.family_id
        ) is None:
            raise ValueError(
                "Supplementary Ablation family_id must be a lowercase public slug"
            )
        if isinstance(self.factor, str) and self.factor in _PAPER_COVERED_FACTORS:
            raise ValueError(
                f"Supplementary Ablation factor {self.factor!r} is "
                "Paper-Covered Analysis and must not be repeated"
            )
        if not isinstance(self.factor, str) or self.factor not in (
            SUPPLEMENTARY_ABLATION_LEVELS
        ):
            raise ValueError(
                f"Supplementary Ablation factor {self.factor!r} is not allowlisted"
            )
        expected_default = SUPPLEMENTARY_ABLATION_DEFAULTS[self.factor]
        canonical_level = _canonical_level(self.factor, self.level, "level")
        canonical_default = _canonical_level(
            self.factor, self.default_level, "default_level"
        )
        if canonical_level not in SUPPLEMENTARY_ABLATION_LEVELS[self.factor]:
            raise ValueError(
                f"Supplementary Ablation factor={self.factor!r} level="
                f"{self.level!r} is not allowlisted"
            )
        if canonical_default != expected_default:
            raise ValueError(
                f"Supplementary Ablation factor={self.factor!r} default_level "
                f"must equal {expected_default!r}"
            )
        if (
            not isinstance(self.canonical_baseline_config_hash, str)
            or _DIGEST.fullmatch(self.canonical_baseline_config_hash) is None
        ):
            raise ValueError(
                "Supplementary Ablation canonical_baseline_config_hash must be "
                "a lowercase SHA-256 digest"
            )
        if (self.language, self.dataset, self.profile) != (
            "python",
            "humaneval",
            "full",
        ):
            raise ValueError(
                "Supplementary Ablation only supports python/humaneval/full"
            )
        object.__setattr__(self, "level", canonical_level)
        object.__setattr__(self, "default_level", canonical_default)


def load_supplementary_ablation_spec(
    path: Path,
) -> SupplementaryAblationSpec:
    """Load one bounded, duplicate-free public spec file."""

    if not isinstance(path, Path):
        raise ValueError("Supplementary Ablation Spec path must be a pathlib.Path")
    absolute = path if path.is_absolute() else Path.cwd() / path
    if any(candidate.is_symlink() for candidate in (absolute, *absolute.parents)):
        raise ValueError("Supplementary Ablation Spec path must not traverse symlinks")
    if not path.is_file():
        raise ValueError(f"Supplementary Ablation Spec file is missing: {path}")
    if path.stat().st_size > _MAX_SPEC_BYTES:
        raise ValueError("Supplementary Ablation Spec exceeds the size limit")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"Supplementary Ablation Spec contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def no_constants(value: str) -> None:
        raise ValueError(
            f"Supplementary Ablation Spec contains non-finite number {value}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constants,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Supplementary Ablation Spec is invalid JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("Supplementary Ablation Spec must contain a JSON object")
    return SupplementaryAblationSpec.from_mapping(value)


def canonical_baseline_config_hash(
    canonical_config: Mapping[str, object],
) -> str:
    """Hash the canonical resolved Python/HumanEval Full configuration."""

    from wfcllm.cli.config_resolver import resolve_method_config
    from wfcllm.gate.production import experiment_contract_hash

    resolved = resolve_method_config(canonical_config)
    generation = resolved.get("generation")
    experiment = resolved.get("experiment")
    if (
        not isinstance(generation, Mapping)
        or generation.get("language") != "python"
        or generation.get("dataset") != "humaneval"
        or not isinstance(experiment, Mapping)
        or experiment.get("profile") != "full"
    ):
        raise ValueError(
            "Supplementary Ablation baseline must be canonical "
            "python/humaneval/full"
        )
    return experiment_contract_hash(resolved)


def resolve_supplementary_ablation(
    baseline: Mapping[str, object],
    spec_value: Mapping[str, object] | SupplementaryAblationSpec,
) -> dict[str, Any]:
    """Derive one closed variant from a validated canonical resolved config."""

    spec = (
        spec_value
        if isinstance(spec_value, SupplementaryAblationSpec)
        else SupplementaryAblationSpec.from_mapping(spec_value)
    )
    from wfcllm.gate.production import experiment_contract_hash

    actual_baseline_hash = experiment_contract_hash(baseline)
    if spec.canonical_baseline_config_hash != actual_baseline_hash:
        raise ValueError(
            f"Supplementary Ablation factor={spec.factor!r} level={spec.level!r} "
            "canonical_baseline_config_hash does not match the current canonical "
            "Python/HumanEval Full configuration"
        )

    is_baseline = spec.level == SUPPLEMENTARY_ABLATION_DEFAULTS[spec.factor]
    effective_factor = "d" if is_baseline else spec.factor
    effective_level: int | float = 12 if is_baseline else spec.level
    effective_default: int | float = (
        12 if is_baseline else spec.default_level
    )

    resolved: dict[str, Any] = deepcopy(dict(baseline))
    semantic_lsh = _mutable_mapping(resolved, "semantic_lsh")
    method = _mutable_mapping(resolved, "method")
    semantic = _mutable_mapping(method, "semantic", prefix="method")
    method_lsh = _mutable_mapping(semantic, "lsh", prefix="method.semantic")
    windowing = _mutable_mapping(method, "windowing", prefix="method")
    detector = _mutable_mapping(resolved, "detector")
    gate_data = _mutable_mapping(resolved, "gate_data")

    runtime = {
        "d": int(semantic_lsh["lsh_d"]),
        "delta": float(semantic_lsh.get("semantic_margin", 0.0)),
        "tau_suit": 0.5,
        "tau_close": 0.5,
        "closure_band_width": 0.1,
        "closure_low": 0.45,
        "closure_high": 0.55,
        "n_min": int(detector["minimum_reliable_windows"]),
        "max_units": int(windowing["max_units"]),
        "nominal_gamma": float(semantic_lsh["lsh_gamma"]),
    }
    runtime[effective_factor] = effective_level
    if effective_factor == "d":
        semantic_lsh["lsh_d"] = int(effective_level)
        method_lsh["d"] = int(effective_level)
    elif effective_factor == "delta":
        semantic_lsh["semantic_margin"] = float(effective_level)
        method_lsh["margin"] = float(effective_level)
    elif effective_factor == "n_min":
        detector["minimum_reliable_windows"] = int(effective_level)
    elif effective_factor == "max_units":
        max_units = int(effective_level)
        windowing["max_units"] = max_units
        gate_data["window_lengths"] = list(range(1, max_units + 1))

    dimension = int(runtime["d"])
    gamma = float(runtime["nominal_gamma"])
    region_count = max(1, round(gamma * (2**dimension)))
    runtime["realized_region_count"] = region_count
    runtime["realized_region_ratio"] = region_count / (2**dimension)
    close_center = float(runtime["tau_close"])
    runtime["closure_low"] = round(close_center - 0.05, 10)
    runtime["closure_high"] = round(close_center + 0.05, 10)

    identity: dict[str, Any] = {
        "schema_version": SUPPLEMENTARY_ABLATION_IDENTITY_VERSION,
        "study_kind": SUPPLEMENTARY_ABLATION_STUDY_KIND,
        "family_id": spec.family_id,
        "factor": effective_factor,
        "canonical_level": effective_level,
        "default_level": effective_default,
        "canonical_baseline_config_hash": actual_baseline_hash,
        "language": spec.language,
        "dataset": spec.dataset,
        "profile": spec.profile,
        "one_factor_at_a_time": True,
        "single_fixed_seed": True,
        "multi_seed_significance_claim": False,
        "replaces_canonical_result": False,
        "formal_eligible": True,
        "diagnostic_only": False,
        "not_official_method": False,
        "runtime": runtime,
    }
    resolved["supplementary_ablation"] = identity
    identity["resolved_config_hash"] = _canonical_hash(resolved)
    return resolved


def build_supplementary_artifact_binding(
    config: Mapping[str, object],
    *,
    diagnostic_test_backend: bool = False,
) -> dict[str, object]:
    """Return the conditional public binding for one study artifact or state.

    Canonical profiles deliberately receive no additional fields.  Diagnostic
    fixture artifacts retain the same study/config identity but cannot inherit
    the production configuration's Formal Eligible markers.
    """

    if not isinstance(config, Mapping):
        raise ValueError("resolved config must be an object")
    if type(diagnostic_test_backend) is not bool:
        raise ValueError("diagnostic_test_backend must be boolean")
    value = config.get("supplementary_ablation")
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Supplementary Ablation identity must be an object")
    identity = deepcopy(dict(value))
    if identity.get("schema_version") != SUPPLEMENTARY_ABLATION_IDENTITY_VERSION:
        raise ValueError("Supplementary Ablation identity schema is invalid")
    if identity.get("study_kind") != SUPPLEMENTARY_ABLATION_STUDY_KIND:
        raise ValueError("Supplementary Ablation study kind is invalid")
    if _DIGEST.fullmatch(str(identity.get("resolved_config_hash", ""))) is None:
        raise ValueError("Supplementary Ablation resolved config hash is invalid")

    formal = not diagnostic_test_backend
    identity.update(
        {
            "formal": formal,
            "formal_eligible": formal,
            "diagnostic_test_backend": diagnostic_test_backend,
            "diagnostic_only": diagnostic_test_backend,
            "not_official_method": diagnostic_test_backend,
        }
    )
    from wfcllm.gate.production import experiment_contract_hash

    return {
        "resolved_config_sha256": experiment_contract_hash(config),
        "supplementary_ablation": identity,
    }


def _canonical_level(
    factor: str, value: object, field_name: str
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Supplementary Ablation factor={factor!r} {field_name} must be numeric"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"Supplementary Ablation factor={factor!r} {field_name} must be finite"
        )
    if factor in {"d", "n_min", "max_units"}:
        if type(value) is not int:
            raise ValueError(
                f"Supplementary Ablation factor={factor!r} {field_name} "
                "must be an integer"
            )
        return value
    return float(value)


def _mutable_mapping(
    parent: dict[str, Any], key: str, *, prefix: str = ""
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        label = f"{prefix}.{key}" if prefix else key
        raise ValueError(f"canonical baseline {label} must be an object")
    return value


def _canonical_hash(value: object) -> str:
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ValueError("Supplementary Ablation config must be canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "SUPPLEMENTARY_ABLATION_DEFAULTS",
    "SUPPLEMENTARY_ABLATION_IDENTITY_VERSION",
    "SUPPLEMENTARY_ABLATION_LEVELS",
    "SUPPLEMENTARY_ABLATION_SPEC_VERSION",
    "SUPPLEMENTARY_ABLATION_STUDY_KIND",
    "SupplementaryAblationSpec",
    "build_supplementary_artifact_binding",
    "canonical_baseline_config_hash",
    "load_supplementary_ablation_spec",
    "resolve_supplementary_ablation",
]
