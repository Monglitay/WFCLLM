"""Current-method supplementary experiment programs.

These helpers reproduce the experiment families from the server-side study
layout against the current Gate-only artifacts.  They deliberately avoid
legacy run readers, frozen server paths, private keys, model paths, and
checkpoint references.
"""

from __future__ import annotations

import ast
import builtins
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import keyword
import math
from pathlib import Path
import random
import string
from typing import Any

from wfcllm.detection.code_only import validate_final_code_record_exact
from wfcllm.detection.gated_pipeline import load_jsonl_records


PLAN_SCHEMA_VERSION = "wfcllm-supplementary-experiment-plan/v1"
SPEC_SCHEMA_VERSION = "wfcllm-supplementary-experiment-spec/v1"
IDENTITY_SCHEMA_VERSION = "wfcllm-supplementary-experiment-identity/v1"
STUDY_KIND = "supplementary_experiment"
RENAME_MANIFEST_SCHEMA_VERSION = "wfcllm-rename-intensity-manifest/v1"
CONDITION_METRICS_SCHEMA_VERSION = "wfcllm-robustness-condition-metrics/v1"
PASSK_SCHEMA_VERSION = "wfcllm-passk-summary/v1"

DEFAULT_RENAME_CONDITIONS: Mapping[str, float] = {
    "rename25": 0.25,
    "rename50": 0.50,
    "rename75": 0.75,
    "rename100": 1.00,
}
SUPPLEMENTARY_EXPERIMENT_LEVELS: Mapping[str, tuple[int | float, ...]] = {
    "B": (0, 6, 12, 18, 24),
    "gamma": (0.25, 0.375, 0.5),
    "d": (6, 12, 24),
    "deployment_key_replicate": (1, 2, 3, 4, 5),
}
SUPPLEMENTARY_EXPERIMENT_DEFAULTS: Mapping[str, int | float] = {
    "B": 18,
    "gamma": 0.25,
    "d": 12,
    "deployment_key_replicate": 1,
}
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


@dataclass(frozen=True)
class SupplementaryExperimentFamily:
    family_id: str
    title: str
    execution_kind: str
    measured_factors: tuple[str, ...]
    levels: Mapping[str, tuple[int | float | str, ...]]
    current_method_rule: str
    required_inputs: tuple[str, ...]
    output_artifacts: tuple[str, ...]
    server_reference: str

    def to_dict(self) -> dict[str, object]:
        return {
            "family_id": self.family_id,
            "title": self.title,
            "execution_kind": self.execution_kind,
            "measured_factors": list(self.measured_factors),
            "levels": {
                key: list(value) for key, value in sorted(self.levels.items())
            },
            "current_method_rule": self.current_method_rule,
            "required_inputs": list(self.required_inputs),
            "output_artifacts": list(self.output_artifacts),
            "server_reference": self.server_reference,
        }


@dataclass(frozen=True)
class SupplementaryExperimentSpec:
    """One controlled paper-covered or post-paper current-method study point."""

    family_id: str
    factor: str
    level: int | float
    default_level: int | float
    canonical_baseline_config_hash: str
    language: str = "python"
    dataset: str = "humaneval"
    profile: str = "full"
    schema_version: str = SPEC_SCHEMA_VERSION
    study_kind: str = STUDY_KIND

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> SupplementaryExperimentSpec:
        if not isinstance(value, Mapping):
            raise ValueError("Supplementary Experiment Spec must be an object")
        if set(value) != _SPEC_FIELDS:
            unknown = sorted(set(value) - _SPEC_FIELDS)
            missing = sorted(_SPEC_FIELDS - set(value))
            raise ValueError(
                "Supplementary Experiment Spec schema mismatch: "
                f"missing={missing}, unknown={unknown}"
            )
        try:
            return cls(**dict(value))  # type: ignore[arg-type]
        except TypeError as exc:
            raise ValueError("Supplementary Experiment Spec fields are invalid") from exc

    def __post_init__(self) -> None:
        if self.schema_version != SPEC_SCHEMA_VERSION:
            raise ValueError(
                f"Supplementary Experiment Spec schema_version must be {SPEC_SCHEMA_VERSION}"
            )
        if self.study_kind != STUDY_KIND:
            raise ValueError(
                f"Supplementary Experiment Spec study_kind must be {STUDY_KIND}"
            )
        if (
            not isinstance(self.family_id, str)
            or not self.family_id
            or any(ch.isspace() for ch in self.family_id)
        ):
            raise ValueError("Supplementary Experiment family_id must be a slug")
        if self.factor not in SUPPLEMENTARY_EXPERIMENT_LEVELS:
            raise ValueError(
                f"Supplementary Experiment factor {self.factor!r} is not allowlisted"
            )
        level = _canonical_spec_level(self.factor, self.level, "level")
        default = _canonical_spec_level(
            self.factor, self.default_level, "default_level"
        )
        if level not in SUPPLEMENTARY_EXPERIMENT_LEVELS[self.factor]:
            raise ValueError(
                f"Supplementary Experiment factor={self.factor!r} level={self.level!r} "
                "is not allowlisted"
            )
        if default != SUPPLEMENTARY_EXPERIMENT_DEFAULTS[self.factor]:
            raise ValueError(
                f"Supplementary Experiment factor={self.factor!r} default_level "
                f"must equal {SUPPLEMENTARY_EXPERIMENT_DEFAULTS[self.factor]!r}"
            )
        if (
            not isinstance(self.canonical_baseline_config_hash, str)
            or len(self.canonical_baseline_config_hash) != 64
            or any(ch not in "0123456789abcdef" for ch in self.canonical_baseline_config_hash)
        ):
            raise ValueError(
                "Supplementary Experiment canonical_baseline_config_hash must be "
                "a lowercase SHA-256 digest"
            )
        if self.profile != "full":
            raise ValueError("Supplementary Experiment only supports full profile")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "default_level", default)


def load_supplementary_experiment_spec(path: Path) -> SupplementaryExperimentSpec:
    """Load one strict current-method supplementary experiment spec."""

    if not isinstance(path, Path):
        raise ValueError("Supplementary Experiment Spec path must be a pathlib.Path")
    if not path.is_file():
        raise ValueError(f"Supplementary Experiment Spec file is missing: {path}")
    if path.stat().st_size > 64 * 1024:
        raise ValueError("Supplementary Experiment Spec exceeds the size limit")

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(
                    f"Supplementary Experiment Spec contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    def no_constants(value: str) -> None:
        raise ValueError(
            f"Supplementary Experiment Spec contains non-finite number {value}"
        )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=no_duplicates,
            parse_constant=no_constants,
        )
    except json.JSONDecodeError as exc:
        raise ValueError("Supplementary Experiment Spec is invalid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Supplementary Experiment Spec must contain a JSON object")
    return SupplementaryExperimentSpec.from_mapping(payload)


def resolve_supplementary_experiment(
    baseline: Mapping[str, object],
    spec_value: Mapping[str, object] | SupplementaryExperimentSpec,
) -> dict[str, Any]:
    """Derive a controlled current-method study config from a canonical full config."""

    spec = (
        spec_value
        if isinstance(spec_value, SupplementaryExperimentSpec)
        else SupplementaryExperimentSpec.from_mapping(spec_value)
    )
    from wfcllm.gate.production import experiment_contract_hash

    actual_baseline_hash = experiment_contract_hash(baseline)
    if spec.canonical_baseline_config_hash != actual_baseline_hash:
        raise ValueError(
            f"Supplementary Experiment factor={spec.factor!r} level={spec.level!r} "
            "canonical_baseline_config_hash does not match the current canonical "
            "Full configuration"
        )
    generation = _mapping(baseline, "generation")
    if (
        generation.get("language") != spec.language
        or generation.get("dataset") != spec.dataset
    ):
        raise ValueError(
            "Supplementary Experiment language/dataset must match the baseline config"
        )
    resolved: dict[str, Any] = deepcopy(dict(baseline))
    method = _mutable_mapping(resolved, "method")
    rewrite = _mutable_mapping(method, "rewrite", prefix="method")
    semantic = _mutable_mapping(method, "semantic", prefix="method")
    method_lsh = _mutable_mapping(semantic, "lsh", prefix="method.semantic")
    semantic_lsh = _mutable_mapping(resolved, "semantic_lsh")
    gate_data = _mutable_mapping(resolved, "gate_data")

    runtime = {
        "B": int(rewrite["max_attempts"]),
        "gamma": float(semantic_lsh["lsh_gamma"]),
        "d": int(semantic_lsh["lsh_d"]),
        "deployment_key_replicate": 1,
    }
    runtime[spec.factor] = spec.level
    if spec.factor == "B":
        rewrite["max_attempts"] = int(spec.level)
        level = int(spec.level)
        gate_data["rewrite_count"] = level
        if level == 0:
            gate_data["rewrite_budgets"] = [0]
        else:
            gate_data["rewrite_budgets"] = sorted(
                {budget for budget in (1, 3, 6, 12, 18, level) if budget <= level}
            )
    elif spec.factor == "gamma":
        semantic_lsh["lsh_gamma"] = float(spec.level)
        method_lsh["gamma"] = float(spec.level)
    elif spec.factor == "d":
        semantic_lsh["lsh_d"] = int(spec.level)
        method_lsh["d"] = int(spec.level)
    elif spec.factor == "deployment_key_replicate":
        pass
    else:  # pragma: no cover - protected by spec validation.
        raise ValueError(f"unsupported Supplementary Experiment factor: {spec.factor}")
    runtime["realized_region_count"] = max(
        1, round(float(runtime["gamma"]) * (2 ** int(runtime["d"])))
    )
    runtime["realized_region_ratio"] = runtime["realized_region_count"] / (
        2 ** int(runtime["d"])
    )
    if spec.factor in {"d", "gamma"}:
        gate_data["lsh_study_point"] = {
            "factor": spec.factor,
            "level": spec.level,
            "not_official_method": True,
        }

    identity: dict[str, Any] = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "study_kind": STUDY_KIND,
        "family_id": spec.family_id,
        "factor": spec.factor,
        "canonical_level": spec.level,
        "default_level": spec.default_level,
        "canonical_baseline_config_hash": actual_baseline_hash,
        "language": spec.language,
        "dataset": spec.dataset,
        "profile": spec.profile,
        "paper_covered_or_posthoc": True,
        "replaces_canonical_result": False,
        "formal": False,
        "formal_eligible": False,
        "diagnostic_only": False,
        "not_official_method": True,
        "runtime": runtime,
    }
    resolved["supplementary_experiment"] = identity
    identity["resolved_config_hash"] = _canonical_hash(resolved)
    return resolved


def build_all_experiment_plan() -> dict[str, object]:
    """Return the local plan covering the two servers' experiment families."""

    from wfcllm.method.ablation import (
        SUPPLEMENTARY_ABLATION_DEFAULTS,
        SUPPLEMENTARY_ABLATION_LEVELS,
    )

    families = [
        SupplementaryExperimentFamily(
            family_id="multilingual-full-matrix",
            title="Five current Full Reproduction Profiles",
            execution_kind="fresh_current_full_runs",
            measured_factors=("language", "dataset"),
            levels={
                "profile": ("full",),
                "language_dataset": (
                    "python/humaneval",
                    "python/mbpp",
                    "cpp/humanevalpack",
                    "java/humanevalpack",
                    "js/humanevalpack",
                ),
            },
            current_method_rule=(
                "Run the canonical eight-phase chain for each existing full "
                "profile without supplementary identity."
            ),
            required_inputs=(
                "local models",
                "datasets",
                "pilot and full source catalogs",
                "training key bank",
                "holdout key bank",
                "deployment key",
                "calibration negative input",
            ),
            output_artifacts=(
                "inputs/final_code.jsonl",
                "detection/positive_details.jsonl",
                "reports/reference_report.json",
                "audit/artifact_integrity.json",
            ),
            server_reference="server14576:multilingual-current-snapshot",
        ),
        SupplementaryExperimentFamily(
            family_id="formal-mechanism-ablation",
            title="Current Supplementary Ablation full-run matrix",
            execution_kind="fresh_current_ablation_runs",
            measured_factors=tuple(SUPPLEMENTARY_ABLATION_LEVELS),
            levels={
                factor: tuple(levels)
                for factor, levels in SUPPLEMENTARY_ABLATION_LEVELS.items()
            },
            current_method_rule=(
                "Use --ablation-spec against python/humaneval/full; every point "
                "runs encoder through audit from a fresh root."
            ),
            required_inputs=(
                "canonical python_humaneval_full.json",
                "ablation spec JSON",
                "family-shared private resources",
                "independent calibration and test negatives",
            ),
            output_artifacts=(
                "supplementary_ablation identity in every formal artifact",
                "ablation_detection",
                "mechanism_funnel",
                "latency",
            ),
            server_reference="server21019:parameter-sensitivity-he-20260723-132453",
        ),
        SupplementaryExperimentFamily(
            family_id="proposal-budget",
            title="Proposal budget study",
            execution_kind="controlled_current_method_study",
            measured_factors=("B",),
            levels={"B": (0, 6, 12, 18, 24)},
            current_method_rule=(
                "Treat B as a paper-covered supplementary study: run current "
                "generation with fixed Gate, fixed detector, and explicit "
                "not_official_method study metadata; do not replace canonical "
                "full-profile results."
            ),
            required_inputs=(
                "current generated positive population or current model prompts",
                "current Gate Bundle from the same run family",
                "independent negative detector input",
            ),
            output_artifacts=(
                "proposal_budget_metrics.json",
                "proposal_budget_table.md",
            ),
            server_reference="server21019:proposal-budget-b0-8-16-18-24-20260724-004424",
        ),
        SupplementaryExperimentFamily(
            family_id="gamma-d-key-confirmation",
            title="Gamma, dimension, and deployment-key confirmation",
            execution_kind="controlled_current_method_study",
            measured_factors=("gamma", "d", "deployment_key_replicate"),
            levels={
                "gamma": (0.25, 0.375, 0.50),
                "d": (6, 12, 24),
                "deployment_key_replicate": (1, 2, 3, 4, 5),
            },
            current_method_rule=(
                "Regenerate current-method keyed regions and detector artifacts "
                "for each declared factor point; never reuse or publish raw keys."
            ),
            required_inputs=(
                "current baseline config",
                "family-shared source catalogs",
                "private key-bank resources",
                "strict positive and negative detector inputs",
            ),
            output_artifacts=("metrics.json", "FINAL_RESULTS.md"),
            server_reference="server21019:gamma-d-key-confirmation-he-20260724-174800",
        ),
        SupplementaryExperimentFamily(
            family_id="rename-intensity",
            title="Deterministic identifier rename robustness",
            execution_kind="post_run_attack_and_redetect",
            measured_factors=("rename_fraction",),
            levels={"rename_fraction": tuple(DEFAULT_RENAME_CONDITIONS.values())},
            current_method_rule=(
                "Transform strict current final_code JSONL into attacked strict "
                "detector inputs, re-run the current detector, then aggregate "
                "full-sample and reliable-window-conditioned metrics."
            ),
            required_inputs=(
                "strict positive final_code.jsonl",
                "strict negative final_code.jsonl",
                "entry_points.jsonl",
                "current calibration artifact",
            ),
            output_artifacts=(
                "rename_intensity_manifest.json",
                "detection/<condition>/{positive,negative}_details.jsonl",
                "metrics/rename_intensity_metrics.json",
            ),
            server_reference="server21019:rename-intensity-he-20260722-2050",
        ),
        SupplementaryExperimentFamily(
            family_id="dipper-rename-robustness",
            title="DIPPER and rename robustness comparison",
            execution_kind="post_run_attack_and_redetect",
            measured_factors=("attack",),
            levels={"attack": ("original", "dipper", "rename")},
            current_method_rule=(
                "Use externally produced attacked code only as strict detector "
                "input; sidecars, quality results, and attack metadata stay out "
                "of detector positives."
            ),
            required_inputs=(
                "strict original positive and negative inputs",
                "strict attacked positive and negative inputs",
                "current calibration artifact",
                "optional posthoc functionality summaries",
            ),
            output_artifacts=("metrics/robustness_metrics.json",),
            server_reference=(
                "server21019:robustness-gamma25-he-20260722-1348 and "
                "dipper-code-adapter-v1-20260722-1615"
            ),
        ),
        SupplementaryExperimentFamily(
            family_id="passk",
            title="Posthoc Pass@k sampled-pool study",
            execution_kind="posthoc_generation_pool_evaluation",
            measured_factors=("k", "sample_count"),
            levels={"k": (1, 10), "sample_count": (10,)},
            current_method_rule=(
                "Compute Pass@k only after generation and evaluation; never feed "
                "correctness back into generation, calibration, detection, or "
                "candidate selection."
            ),
            required_inputs=(
                "generation/all_samples.jsonl",
                "evaluation/sample_<index>/results.jsonl",
            ),
            output_artifacts=("results/passk_summary.json", "results/per_task_passk.jsonl"),
            server_reference="server21019:pass10-he-gated-7b-20260723-184500",
        ),
    ]
    return {
        "schema_version": PLAN_SCHEMA_VERSION,
        "method": "gated_semantic_window_v1",
        "authority": "current-local-architecture",
        "formal_ablation_defaults": dict(SUPPLEMENTARY_ABLATION_DEFAULTS),
        "families": [family.to_dict() for family in families],
    }


def write_all_experiment_plan(path: Path) -> dict[str, object]:
    plan = build_all_experiment_plan()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return plan


def write_current_study_specs(
    *,
    canonical_config_path: Path,
    output_root: Path,
    family_id: str = "current-study",
) -> dict[str, object]:
    """Write paper-covered current-study specs and a shell command manifest."""

    from wfcllm.cli.config_resolver import load_config, resolve_method_config
    from wfcllm.gate.production import experiment_contract_hash
    from wfcllm.method.ablation import (
        SUPPLEMENTARY_ABLATION_DEFAULTS,
        SUPPLEMENTARY_ABLATION_LEVELS,
        canonical_baseline_config_hash,
    )

    canonical = load_config(canonical_config_path)
    baseline = resolve_method_config(canonical)
    generation = _mapping(baseline, "generation")
    experiment = _mapping(baseline, "experiment")
    language = str(generation["language"])
    dataset = str(generation["dataset"])
    profile = str(experiment["profile"])
    baseline_hash = experiment_contract_hash(baseline)
    ablation_hash = canonical_baseline_config_hash(canonical)
    ablation_dir = output_root / "formal_ablation_specs"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    ablation_specs: list[dict[str, object]] = []
    wrote_formal_baseline = False
    for factor, levels in SUPPLEMENTARY_ABLATION_LEVELS.items():
        for level in levels:
            default = SUPPLEMENTARY_ABLATION_DEFAULTS[factor]
            if level == default:
                if wrote_formal_baseline:
                    continue
                wrote_formal_baseline = True
            spec = {
                "schema_version": "wfcllm-supplementary-ablation-spec/v1",
                "study_kind": "supplementary_ablation",
                "family_id": f"{family_id}-formal-ablation",
                "factor": factor,
                "level": level,
                "default_level": default,
                "canonical_baseline_config_hash": ablation_hash,
                "language": language,
                "dataset": dataset,
                "profile": profile,
            }
            stem = f"{factor}_{str(level).replace('.', 'p')}"
            path = ablation_dir / f"{stem}.json"
            path.write_text(
                json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            ablation_specs.append(
                {"factor": factor, "level": level, "path": str(path)}
            )
    spec_dir = output_root / "supplementary_experiment_specs"
    spec_dir.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, object]] = []
    for factor, levels in SUPPLEMENTARY_EXPERIMENT_LEVELS.items():
        for level in levels:
            if level == SUPPLEMENTARY_EXPERIMENT_DEFAULTS[factor]:
                continue
            spec = {
                "schema_version": SPEC_SCHEMA_VERSION,
                "study_kind": STUDY_KIND,
                "family_id": family_id,
                "factor": factor,
                "level": level,
                "default_level": SUPPLEMENTARY_EXPERIMENT_DEFAULTS[factor],
                "canonical_baseline_config_hash": baseline_hash,
                "language": language,
                "dataset": dataset,
                "profile": profile,
            }
            resolved = resolve_method_config(
                canonical,
                supplementary_experiment=SupplementaryExperimentSpec.from_mapping(spec),
            )
            identity = resolved["supplementary_experiment"]
            assert isinstance(identity, Mapping)
            stem = f"{factor}_{str(level).replace('.', 'p')}"
            path = spec_dir / f"{stem}.json"
            path.write_text(
                json.dumps(spec, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            specs.append(
                {
                    "factor": factor,
                    "level": level,
                    "path": str(path),
                    "resolved_config_hash": identity["resolved_config_hash"],
                }
            )
    commands_path = output_root / "run_commands.sh"
    commands_path.write_text(
        _current_study_commands(specs, ablation_specs, canonical_config_path),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "wfcllm-current-study-spec-manifest/v1",
        "canonical_config_path": str(canonical_config_path),
        "canonical_baseline_config_hash": baseline_hash,
        "family_id": family_id,
        "formal_ablation_spec_count": len(ablation_specs),
        "formal_ablation_specs": ablation_specs,
        "current_study_spec_count": len(specs),
        "current_study_specs": specs,
        "run_commands_path": str(commands_path),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def prepare_rename_intensity_inputs(
    *,
    positive_input: Path,
    negative_input: Path,
    entry_points: Path,
    output_root: Path,
    conditions: Mapping[str, float] | None = None,
    name_seed: int = 7,
    selection_seed: int = 20260722,
) -> dict[str, object]:
    """Create strict renamed detector inputs for the rename-intensity study."""

    condition_map = dict(conditions or DEFAULT_RENAME_CONDITIONS)
    _validate_conditions(condition_map)
    entry_index = _load_entry_points(entry_points)
    positive_rows = load_jsonl_records(positive_input)
    negative_rows = load_jsonl_records(negative_input)
    output_root.mkdir(parents=True, exist_ok=True)

    positive_original = output_root / "inputs" / "original_positive.jsonl"
    negative_original = output_root / "inputs" / "original_negative.jsonl"
    _write_jsonl(positive_original, positive_rows)
    _write_jsonl(negative_original, negative_rows)

    manifest: dict[str, object] = {
        "schema_version": RENAME_MANIFEST_SCHEMA_VERSION,
        "protocol": (
            "deterministic-python-identifier-renaming/v1; public entry names, "
            "formal parameters, top-level function names, imports, builtins, "
            "self/cls, and dunder names are protected"
        ),
        "positive_input_sha256": _sha256_file(positive_input),
        "negative_input_sha256": _sha256_file(negative_input),
        "entry_points_sha256": _sha256_file(entry_points),
        "name_seed": name_seed,
        "selection_seed": selection_seed,
        "conditions": {},
    }
    for condition_id, fraction in sorted(condition_map.items()):
        positive_out, positive_summary = _rename_rows(
            rows=positive_rows,
            entry_index=entry_index,
            output_path=output_root / "inputs" / condition_id / "positive.jsonl",
            condition_id=condition_id,
            fraction=fraction,
            name_seed=name_seed,
            selection_seed=selection_seed,
        )
        negative_out, negative_summary = _rename_rows(
            rows=negative_rows,
            entry_index=entry_index,
            output_path=output_root / "inputs" / condition_id / "negative.jsonl",
            condition_id=condition_id,
            fraction=fraction,
            name_seed=name_seed,
            selection_seed=selection_seed,
        )
        manifest["conditions"][condition_id] = {
            "fraction": fraction,
            "positive": positive_summary,
            "negative": negative_summary,
            "positive_path": str(positive_out),
            "negative_path": str(negative_out),
            "positive_sha256": _sha256_file(positive_out),
            "negative_sha256": _sha256_file(negative_out),
        }
    manifest_path = output_root / "rename_intensity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def aggregate_detection_conditions(
    *,
    root: Path,
    conditions: Sequence[str],
    output_name: str = "robustness_metrics",
    minimum_reliable_windows: int = 3,
) -> dict[str, object]:
    """Aggregate current detector details for robustness-style conditions."""

    if not conditions:
        raise ValueError("at least one condition is required")
    result: dict[str, object] = {
        "schema_version": CONDITION_METRICS_SCHEMA_VERSION,
        "score": (
            "hit_rate for rows with reliable windows; zero-window rows score 0"
        ),
        "minimum_reliable_windows": minimum_reliable_windows,
        "operating_point": (
            "attainable ROC point closest to FPR=0.05; ties choose lower FPR "
            "then higher TPR"
        ),
        "conditions": {},
    }
    for condition_id in conditions:
        positive = _read_jsonl_objects(
            root / "detection" / condition_id / "positive_details.jsonl"
        )
        negative = _read_jsonl_objects(
            root / "detection" / condition_id / "negative_details.jsonl"
        )
        condition: dict[str, object] = {
            "full": _condition_metrics(
                positive,
                negative,
                minimum_reliable_windows=minimum_reliable_windows,
                conditional=False,
            ),
            "conditional": _condition_metrics(
                positive,
                negative,
                minimum_reliable_windows=minimum_reliable_windows,
                conditional=True,
            ),
        }
        execution_path = root / "execution" / condition_id / "positive_summary.json"
        if execution_path.exists():
            condition["functionality"] = _read_json_object(execution_path)
        result["conditions"][condition_id] = condition

    first = conditions[0]
    baseline = result["conditions"][first]
    assert isinstance(baseline, Mapping)
    for condition_id in conditions[1:]:
        current = result["conditions"][condition_id]
        assert isinstance(current, dict)
        current["drop_from_" + first + "_pp"] = _drop_pp(baseline, current)

    metrics_dir = root / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    json_path = metrics_dir / f"{output_name}.json"
    markdown_path = metrics_dir / f"{output_name}.md"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(
        _condition_metrics_markdown(result, conditions),
        encoding="utf-8",
    )
    result["output_json"] = str(json_path)
    result["output_markdown"] = str(markdown_path)
    return result


def pass_at_k(n: int, c: int, k: int) -> float:
    if type(n) is not int or type(c) is not int or type(k) is not int:
        raise ValueError("n, c, and k must be integers")
    if k < 1 or n < k or c < 0 or c > n:
        raise ValueError("Pass@k requires 0 <= c <= n and n >= k >= 1")
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate_passk(
    *,
    root: Path,
    num_samples: int,
    k_values: Sequence[int] = (1, 10),
    expected_task_count: int | None = None,
) -> dict[str, object]:
    """Aggregate posthoc sampled-pool Pass@k from evaluation result rows."""

    if type(num_samples) is not int or num_samples < 1:
        raise ValueError("num_samples must be a positive integer")
    if not k_values:
        raise ValueError("at least one k value is required")
    k_tuple = tuple(int(k) for k in k_values)
    if any(k < 1 or k > num_samples for k in k_tuple):
        raise ValueError("every k must be in [1, num_samples]")

    generation_path = root / "generation" / "all_samples.jsonl"
    generation_rows = _read_jsonl_objects(generation_path)
    generation_index = {
        (str(row["id"]), int(row["sample_index"])): row
        for row in generation_rows
    }
    if len(generation_index) != len(generation_rows):
        raise ValueError("generation rows contain duplicate id/sample_index pairs")
    per_task: dict[str, list[bool]] = defaultdict(lambda: [False] * num_samples)
    result_hashes: dict[str, str] = {}
    candidate_passed = [0] * num_samples
    for sample_index in range(num_samples):
        result_path = (
            root / "evaluation" / f"sample_{sample_index:02d}" / "results.jsonl"
        )
        rows = _read_jsonl_objects(result_path)
        result_hashes[str(sample_index)] = _sha256_file(result_path)
        ids = [str(row["id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError(f"evaluation sample {sample_index} has duplicate ids")
        for row in rows:
            task_id = str(row["id"])
            key = (task_id, sample_index)
            if key not in generation_index:
                raise ValueError(f"evaluation row has no generation provenance: {key}")
            passed = bool(row.get("passed"))
            per_task[task_id][sample_index] = passed
            candidate_passed[sample_index] += int(passed)
    if expected_task_count is not None and len(per_task) != expected_task_count:
        raise ValueError(
            f"expected {expected_task_count} tasks, got {len(per_task)}"
        )
    task_rows: list[dict[str, object]] = []
    pass_values: dict[str, list[float]] = {str(k): [] for k in k_tuple}
    for task_id in sorted(per_task):
        outcomes = per_task[task_id]
        correct = sum(outcomes)
        task_row: dict[str, object] = {
            "id": task_id,
            "n": num_samples,
            "correct": correct,
            "outcomes": outcomes,
        }
        for k in k_tuple:
            value = pass_at_k(num_samples, correct, k)
            task_row[f"pass_at_{k}_estimate"] = value
            pass_values[str(k)].append(value)
        task_rows.append(task_row)
    summary = {
        "schema_version": PASSK_SCHEMA_VERSION,
        "tasks": len(task_rows),
        "candidates_per_task": num_samples,
        "total_candidates": len(task_rows) * num_samples,
        "total_correct_candidates": sum(candidate_passed),
        "k_values": list(k_tuple),
        "pass_at_k": {
            str(k): sum(pass_values[str(k)]) / len(pass_values[str(k)])
            for k in k_tuple
        },
        "candidate_index_pass_rates": [
            {
                "sample_index": index,
                "passed": passed,
                "total": len(task_rows),
                "pass_rate": passed / len(task_rows) if task_rows else 0.0,
            }
            for index, passed in enumerate(candidate_passed)
        ],
        "estimator": "mean_i[1 - C(n-c_i,k)/C(n,k)]",
        "posthoc_only": True,
        "not_used_for_generation": True,
        "not_used_for_retry": True,
        "not_used_for_selection": True,
        "not_used_for_calibration": True,
        "not_used_for_detection": True,
        "generation_sha256": _sha256_file(generation_path),
        "evaluation_result_sha256": result_hashes,
    }
    result_dir = root / "results"
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(result_dir / "per_task_passk.jsonl", task_rows)
    (result_dir / "passk_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def rename_code(
    *,
    code: str,
    entry_point: str,
    task_id: str,
    name_seed: int,
    selection_seed: int,
    fraction: float,
) -> tuple[str, dict[str, object]]:
    """Apply deterministic safe local-variable renaming to one Python program."""

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return code, {
            "attack_applied": False,
            "changed": False,
            "eligible_identifier_count": 0,
            "selected_identifier_count": 0,
            "realized_identifier_fraction": 0.0,
            "reason": "source_syntax_error",
            "error": str(exc),
        }

    before_signature = _signature_dump(tree, entry_point)
    names = _candidate_names(tree, entry_point)
    occupied = _all_identifiers(tree)
    rng = random.Random(name_seed)
    complete_mapping = {name: _random_name(rng, occupied) for name in names}
    selected_names = [
        name
        for name in names
        if fraction >= 1.0
        or _selection_value(selection_seed, task_id, name) < fraction
    ]
    mapping = {name: complete_mapping[name] for name in selected_names}
    if not names:
        return code, {
            "attack_applied": False,
            "changed": False,
            "eligible_identifier_count": 0,
            "selected_identifier_count": 0,
            "realized_identifier_fraction": 0.0,
            "reason": "no_safe_rename_target",
            "mapping": {},
        }
    if not mapping:
        return code, {
            "attack_applied": False,
            "changed": False,
            "eligible_identifier_count": len(names),
            "selected_identifier_count": 0,
            "realized_identifier_fraction": 0.0,
            "reason": "no_identifier_selected_at_strength",
            "mapping": {},
        }
    updated = _Renamer(mapping).visit(tree)
    ast.fix_missing_locations(updated)
    after_signature = _signature_dump(updated, entry_point)
    if before_signature != after_signature:
        raise RuntimeError(f"entry signature changed for {entry_point}")
    transformed = ast.unparse(updated).rstrip() + "\n"
    ast.parse(transformed)
    return transformed, {
        "attack_applied": True,
        "changed": transformed != code,
        "eligible_identifier_count": len(names),
        "selected_identifier_count": len(mapping),
        "realized_identifier_fraction": len(mapping) / len(names),
        "reason": "renamed",
        "mapping": mapping,
        "entry_point": entry_point,
        "entry_signature_preserved": True,
    }


def _validate_conditions(conditions: Mapping[str, float]) -> None:
    if not conditions:
        raise ValueError("at least one rename condition is required")
    for name, fraction in conditions.items():
        if not isinstance(name, str) or not name:
            raise ValueError("rename condition names must be non-empty strings")
        if (
            isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or not 0.0 <= float(fraction) <= 1.0
        ):
            raise ValueError("rename fractions must be finite values in [0, 1]")


def _load_entry_points(path: Path) -> dict[str, str]:
    rows = _read_jsonl_objects(path)
    result: dict[str, str] = {}
    for index, row in enumerate(rows, start=1):
        task_id = row.get("id")
        entry_point = row.get("entry_point")
        if (
            not isinstance(task_id, str)
            or not task_id
            or not isinstance(entry_point, str)
            or not entry_point
        ):
            raise ValueError(f"entry point row {index} is invalid")
        if task_id in result:
            raise ValueError(f"duplicate entry point id: {task_id}")
        result[task_id] = entry_point
    return result


def _rename_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    entry_index: Mapping[str, str],
    output_path: Path,
    condition_id: str,
    fraction: float,
    name_seed: int,
    selection_seed: int,
) -> tuple[Path, dict[str, object]]:
    attacked: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for row in rows:
        task_id = str(row["id"])
        if task_id not in entry_index:
            raise ValueError(f"missing entry point for {task_id}")
        transformed, info = rename_code(
            code=str(row["final_code"]),
            entry_point=entry_index[task_id],
            task_id=task_id,
            name_seed=name_seed,
            selection_seed=selection_seed,
            fraction=fraction,
        )
        payload = {
            "id": row["id"],
            "dataset": row["dataset"],
            "prompt": row["prompt"],
            "final_code": transformed,
        }
        validate_final_code_record_exact(payload)
        attacked.append(payload)
        metadata.append({"id": task_id, "condition": condition_id, **info})
    _write_jsonl(output_path, attacked)
    metadata_path = output_path.with_suffix(".metadata.jsonl")
    _write_jsonl(metadata_path, metadata)
    return output_path, _rename_summary(metadata)


def _rename_summary(metadata: Sequence[Mapping[str, object]]) -> dict[str, object]:
    samples = len(metadata)
    eligible = sum(int(row.get("eligible_identifier_count", 0)) for row in metadata)
    selected = sum(int(row.get("selected_identifier_count", 0)) for row in metadata)
    changed = sum(row.get("changed") is True for row in metadata)
    reasons = Counter(str(row.get("reason")) for row in metadata)
    return {
        "samples": samples,
        "eligible_identifier_count": eligible,
        "selected_identifier_count": selected,
        "realized_identifier_fraction": selected / eligible if eligible else 0.0,
        "changed_samples": changed,
        "changed_sample_rate": changed / samples if samples else 0.0,
        "reasons": dict(sorted(reasons.items())),
    }


class _Renamer(ast.NodeTransformer):
    def __init__(self, mapping: Mapping[str, str]) -> None:
        self._mapping = mapping

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._mapping:
            node.id = self._mapping[node.id]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name in self._mapping:
            node.name = self._mapping[node.name]
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if node.name in self._mapping:
            node.name = self._mapping[node.name]
        return self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> ast.AST:
        node.names = [self._mapping.get(name, name) for name in node.names]
        return node

    def visit_Nonlocal(self, node: ast.Nonlocal) -> ast.AST:
        node.names = [self._mapping.get(name, name) for name in node.names]
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> ast.AST:
        if isinstance(node.name, str) and node.name in self._mapping:
            node.name = self._mapping[node.name]
        return self.generic_visit(node)

    def visit_MatchAs(self, node: ast.MatchAs) -> ast.AST:
        if node.name in self._mapping:
            node.name = self._mapping[node.name]
        return self.generic_visit(node)

    def visit_MatchStar(self, node: ast.MatchStar) -> ast.AST:
        if node.name in self._mapping:
            node.name = self._mapping[node.name]
        return node

    def visit_MatchMapping(self, node: ast.MatchMapping) -> ast.AST:
        if node.rest in self._mapping:
            node.rest = self._mapping[node.rest]
        return self.generic_visit(node)


def _candidate_names(tree: ast.Module, entry_point: str) -> list[str]:
    top_level_functions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    formal_parameters = {node.arg for node in ast.walk(tree) if isinstance(node, ast.arg)}
    protected = {
        entry_point,
        *top_level_functions,
        *formal_parameters,
        *_imported_names(tree),
        *dir(builtins),
        "self",
        "cls",
    }
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            bound.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name != entry_point:
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.name, str):
            bound.add(node.name)
    return sorted(
        name
        for name in bound
        if name not in protected
        and not keyword.iskeyword(name)
        and not name.startswith("__")
    )


def _all_identifiers(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.arg):
            out.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.alias):
            out.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
    return out


def _imported_names(tree: ast.AST) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add(alias.asname or alias.name.split(".")[0])
    return out


def _signature_dump(tree: ast.Module, entry_point: str) -> str | None:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_point:
            return ast.dump(node.args, annotate_fields=True, include_attributes=False)
    return None


def _random_name(rng: random.Random, occupied: set[str]) -> str:
    while True:
        length = rng.randint(2, 5)
        name = "".join(rng.choice(string.ascii_lowercase) for _ in range(length))
        if name not in occupied and not keyword.iskeyword(name) and name not in dir(builtins):
            occupied.add(name)
            return name


def _selection_value(seed: int, task_id: str, identifier: str) -> float:
    payload = f"{seed}\0{task_id}\0{identifier}".encode("utf-8")
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return value / float(1 << 64)


def _condition_metrics(
    positive_rows: Sequence[Mapping[str, object]],
    negative_rows: Sequence[Mapping[str, object]],
    *,
    minimum_reliable_windows: int,
    conditional: bool,
) -> dict[str, object]:
    if conditional:
        positive = [
            row
            for row in positive_rows
            if _reliable_window_count(row) >= minimum_reliable_windows
        ]
        negative = [
            row
            for row in negative_rows
            if _reliable_window_count(row) >= minimum_reliable_windows
        ]
    else:
        positive = list(positive_rows)
        negative = list(negative_rows)
    if not positive or not negative:
        if not positive_rows or not negative_rows:
            raise ValueError("condition metrics require positive and negative rows")
        return {
            "auroc": 0.0,
            "threshold": None,
            "tp": 0,
            "fp": 0,
            "tpr": 0.0,
            "fpr": 0.0,
            "positive_n": len(positive),
            "negative_n": len(negative),
            "positive_population_n": len(positive_rows),
            "negative_population_n": len(negative_rows),
            "positive_coverage": len(positive) / len(positive_rows),
            "negative_coverage": len(negative) / len(negative_rows),
            "positive_mean_hit_rate": 0.0,
            "negative_mean_hit_rate": 0.0,
            "positive_mean_reliable_windows": (
                sum(_reliable_window_count(row) for row in positive_rows)
                / len(positive_rows)
            ),
            "negative_mean_reliable_windows": (
                sum(_reliable_window_count(row) for row in negative_rows)
                / len(negative_rows)
            ),
            "positive_zero_windows": sum(
                _reliable_window_count(row) == 0 for row in positive_rows
            ),
            "negative_zero_windows": sum(
                _reliable_window_count(row) == 0 for row in negative_rows
            ),
            "insufficient_population": True,
        }
    positive_scores = [_sample_score(row) for row in positive]
    negative_scores = [_sample_score(row) for row in negative]
    point = _closest_roc_point(positive_scores, negative_scores)
    return {
        "auroc": _auroc(positive_scores, negative_scores),
        **point,
        "positive_n": len(positive),
        "negative_n": len(negative),
        "positive_population_n": len(positive_rows),
        "negative_population_n": len(negative_rows),
        "positive_coverage": len(positive) / len(positive_rows),
        "negative_coverage": len(negative) / len(negative_rows),
        "positive_mean_hit_rate": sum(positive_scores) / len(positive_scores),
        "negative_mean_hit_rate": sum(negative_scores) / len(negative_scores),
        "positive_mean_reliable_windows": (
            sum(_reliable_window_count(row) for row in positive_rows)
            / len(positive_rows)
        ),
        "negative_mean_reliable_windows": (
            sum(_reliable_window_count(row) for row in negative_rows)
            / len(negative_rows)
        ),
        "positive_zero_windows": sum(
            _reliable_window_count(row) == 0 for row in positive_rows
        ),
        "negative_zero_windows": sum(
            _reliable_window_count(row) == 0 for row in negative_rows
        ),
    }


def _drop_pp(
    baseline: Mapping[str, object],
    current: Mapping[str, object],
) -> dict[str, float]:
    base_full = baseline["full"]
    curr_full = current["full"]
    base_cond = baseline["conditional"]
    curr_cond = current["conditional"]
    if not all(
        isinstance(value, Mapping)
        for value in (base_full, curr_full, base_cond, curr_cond)
    ):
        raise ValueError("condition metrics are malformed")
    return {
        "full_auroc": 100.0 * (float(base_full["auroc"]) - float(curr_full["auroc"])),
        "full_tpr": 100.0 * (float(base_full["tpr"]) - float(curr_full["tpr"])),
        "conditional_auroc": 100.0
        * (float(base_cond["auroc"]) - float(curr_cond["auroc"])),
        "conditional_tpr": 100.0
        * (float(base_cond["tpr"]) - float(curr_cond["tpr"])),
        "positive_coverage": 100.0
        * (
            float(base_cond["positive_coverage"])
            - float(curr_cond["positive_coverage"])
        ),
    }


def _sample_score(row: Mapping[str, object]) -> float:
    return float(row.get("hit_rate", 0.0) or 0.0) if _reliable_window_count(row) > 0 else 0.0


def _reliable_window_count(row: Mapping[str, object]) -> int:
    value = row.get("reliable_window_count", 0)
    if type(value) is not int or value < 0:
        raise ValueError("reliable_window_count must be a non-negative integer")
    return value


def _auroc(positive: Sequence[float], negative: Sequence[float]) -> float:
    if not positive or not negative:
        raise ValueError("AUROC requires both positive and negative scores")
    wins = 0.0
    for positive_score in positive:
        for negative_score in negative:
            if positive_score > negative_score:
                wins += 1.0
            elif positive_score == negative_score:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _closest_roc_point(
    positive: Sequence[float],
    negative: Sequence[float],
    *,
    target_fpr: float = 0.05,
) -> dict[str, object]:
    values = sorted(set([*positive, *negative]), reverse=True)
    candidates: list[dict[str, object]] = []
    for threshold in [math.inf, *values]:
        tp = sum(value >= threshold for value in positive)
        fp = sum(value >= threshold for value in negative)
        candidates.append(
            {
                "threshold": None if math.isinf(threshold) else threshold,
                "tp": tp,
                "fp": fp,
                "tpr": tp / len(positive),
                "fpr": fp / len(negative),
            }
        )
    return min(
        candidates,
        key=lambda row: (
            abs(float(row["fpr"]) - target_fpr),
            float(row["fpr"]),
            -float(row["tpr"]),
        ),
    )


def _condition_metrics_markdown(
    result: Mapping[str, object],
    conditions: Sequence[str],
) -> str:
    all_conditions = result["conditions"]
    if not isinstance(all_conditions, Mapping):
        raise ValueError("condition metrics are malformed")
    lines = [
        "# WFCLLM Supplementary Robustness Metrics",
        "",
        "## Full Sample",
        "",
        "| Condition | AUROC | TPR@about 5% FPR | Realized FPR | TP/Pos | FP/Neg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition_id in conditions:
        item = all_conditions[condition_id]
        assert isinstance(item, Mapping)
        metric = item["full"]
        assert isinstance(metric, Mapping)
        lines.append(_condition_metric_row(condition_id, metric))
    lines += [
        "",
        "## Conditional",
        "",
        "| Condition | AUROC | TPR@about 5% FPR | Realized FPR | TP/Pos | FP/Neg | Positive Coverage | Negative Coverage |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id in conditions:
        item = all_conditions[condition_id]
        assert isinstance(item, Mapping)
        metric = item["conditional"]
        assert isinstance(metric, Mapping)
        lines.append(_conditional_metric_row(condition_id, metric))
    return "\n".join(lines) + "\n"


def _condition_metric_row(condition_id: str, metric: Mapping[str, object]) -> str:
    return (
        f"| {condition_id} | {_pct(float(metric['auroc'])):.4f}% | "
        f"{_pct(float(metric['tpr'])):.4f}% | "
        f"{_pct(float(metric['fpr'])):.4f}% | "
        f"{metric['tp']}/{metric['positive_n']} | "
        f"{metric['fp']}/{metric['negative_n']} |"
    )


def _conditional_metric_row(condition_id: str, metric: Mapping[str, object]) -> str:
    return (
        f"| {condition_id} | {_pct(float(metric['auroc'])):.4f}% | "
        f"{_pct(float(metric['tpr'])):.4f}% | "
        f"{_pct(float(metric['fpr'])):.4f}% | "
        f"{metric['tp']}/{metric['positive_n']} | "
        f"{metric['fp']}/{metric['negative_n']} | "
        f"{_pct(float(metric['positive_coverage'])):.4f}% | "
        f"{_pct(float(metric['negative_coverage'])):.4f}% |"
    )


def _pct(value: float) -> float:
    return 100.0 * value


def _read_jsonl_objects(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"required JSONL input is missing: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must contain an object")
            rows.append(value)
    if not rows:
        raise ValueError(f"required JSONL input is empty: {path}")
    return rows


def _read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain an object")
    return value


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _current_study_commands(
    current_specs: Sequence[Mapping[str, object]],
    ablation_specs: Sequence[Mapping[str, object]],
    canonical_config_path: Path,
) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Set the standard WFCLLM runtime resource environment variables first:",
        "# GENERATION_MODEL_PATH, SEMANTIC_ENCODER_MODEL_PATH, GATE_BASE_MODEL_PATH,",
        "# PILOT_SOURCE_CATALOG, FULL_SOURCE_CATALOG, NEGATIVE_INPUT,",
        "# TEST_NEGATIVE_INPUT, SUPPLEMENTARY_TRAINING_KEY_BANK,",
        "# SUPPLEMENTARY_HOLDOUT_KEY_BANK, SUPPLEMENTARY_DEPLOYMENT_KEY,",
        "# and POSTHOC_STDLIB_PATH for Python profiles. Formal ablation commands",
        "# reuse SUPPLEMENTARY_* unless ABLATION_* overrides are already set.",
        "",
        f'WFCLLM_CONFIG="{canonical_config_path}"',
        "export WFCLLM_CONFIG",
        "",
    ]
    for spec in ablation_specs:
        factor = spec["factor"]
        level = str(spec["level"]).replace(".", "p")
        path = spec["path"]
        lines.extend(
            [
                f'# formal ablation {factor}={spec["level"]}',
                f'EXPERIMENT_ROOT="${{EXPERIMENT_ROOT_BASE:?set EXPERIMENT_ROOT_BASE}}/ablation-{factor}-{level}" \\',
                'ABLATION_TRAINING_KEY_BANK="${ABLATION_TRAINING_KEY_BANK:-${SUPPLEMENTARY_TRAINING_KEY_BANK:?set ABLATION_TRAINING_KEY_BANK or SUPPLEMENTARY_TRAINING_KEY_BANK}}" \\',
                'ABLATION_HOLDOUT_KEY_BANK="${ABLATION_HOLDOUT_KEY_BANK:-${SUPPLEMENTARY_HOLDOUT_KEY_BANK:?set ABLATION_HOLDOUT_KEY_BANK or SUPPLEMENTARY_HOLDOUT_KEY_BANK}}" \\',
                'ABLATION_DEPLOYMENT_KEY="${ABLATION_DEPLOYMENT_KEY:-${SUPPLEMENTARY_DEPLOYMENT_KEY:?set ABLATION_DEPLOYMENT_KEY or SUPPLEMENTARY_DEPLOYMENT_KEY}}" \\',
                f'ABLATION_SPEC="{path}" \\',
                'scripts/experiments/run_python_humaneval_full.sh',
                "",
            ]
        )
    for spec in current_specs:
        factor = spec["factor"]
        level = str(spec["level"]).replace(".", "p")
        path = spec["path"]
        lines.extend(
            [
                f'# current study {factor}={spec["level"]}',
                f'EXPERIMENT_ROOT="${{EXPERIMENT_ROOT_BASE:?set EXPERIMENT_ROOT_BASE}}/study-{factor}-{level}" \\',
                f'SUPPLEMENTARY_EXPERIMENT_SPEC="{path}" \\',
                'scripts/experiments/run_python_humaneval_full.sh',
                "",
            ]
        )
    return "\n".join(lines)


def _canonical_spec_level(
    factor: str,
    value: object,
    field_name: str,
) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"Supplementary Experiment factor={factor!r} {field_name} must be numeric"
        )
    if not math.isfinite(float(value)):
        raise ValueError(
            f"Supplementary Experiment factor={factor!r} {field_name} must be finite"
        )
    if factor in {"B", "d", "deployment_key_replicate"}:
        if type(value) is not int:
            raise ValueError(
                f"Supplementary Experiment factor={factor!r} {field_name} "
                "must be an integer"
            )
        return value
    return float(value)


def _mapping(parent: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"baseline {key} must be an object")
    return value


def _mutable_mapping(
    parent: dict[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        label = f"{prefix}.{key}" if prefix else key
        raise ValueError(f"baseline {label} must be an object")
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
        raise ValueError("Supplementary Experiment config must be canonical JSON") from exc
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "CONDITION_METRICS_SCHEMA_VERSION",
    "DEFAULT_RENAME_CONDITIONS",
    "PASSK_SCHEMA_VERSION",
    "PLAN_SCHEMA_VERSION",
    "RENAME_MANIFEST_SCHEMA_VERSION",
    "SPEC_SCHEMA_VERSION",
    "STUDY_KIND",
    "SUPPLEMENTARY_EXPERIMENT_DEFAULTS",
    "SUPPLEMENTARY_EXPERIMENT_LEVELS",
    "SupplementaryExperimentSpec",
    "aggregate_detection_conditions",
    "aggregate_passk",
    "build_all_experiment_plan",
    "load_supplementary_experiment_spec",
    "pass_at_k",
    "prepare_rename_intensity_inputs",
    "rename_code",
    "resolve_supplementary_experiment",
    "write_all_experiment_plan",
    "write_current_study_specs",
]
