from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.cli.experiment_preflight import validate_ablation_negative_corpora
from wfcllm.cli.runners import _gated_report_metrics_from_artifacts
from wfcllm.evaluation.supplementary import build_supplementary_mechanism_report
from wfcllm.gate.data import LshProbeResult
from wfcllm.gate.production import experiment_contract_hash
from wfcllm.method.ablation import (
    SUPPLEMENTARY_ABLATION_DEFAULTS,
    SUPPLEMENTARY_ABLATION_LEVELS,
    build_supplementary_artifact_binding,
    canonical_baseline_config_hash,
    load_supplementary_ablation_spec,
)


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_CONFIG = (
    ROOT / "configs" / "wfcllm" / "experiments" / "python_humaneval_full.json"
)


def test_public_config_resolves_dimension_ablation_from_canonical_full() -> None:
    canonical = load_config(CANONICAL_CONFIG)
    baseline_hash = canonical_baseline_config_hash(canonical)
    spec = {
        "schema_version": "wfcllm-supplementary-ablation-spec/v1",
        "study_kind": "supplementary_ablation",
        "family_id": "issue-3-fixed-seed",
        "factor": "d",
        "level": 8,
        "default_level": 12,
        "canonical_baseline_config_hash": baseline_hash,
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
    }

    baseline = resolve_method_config(canonical)
    resolved = resolve_method_config(canonical, supplementary_ablation=spec)

    assert resolved["method"]["name"] == "gated_semantic_window_v1"
    assert resolved["experiment"]["profile"] == "full"
    assert resolved["semantic_lsh"]["lsh_d"] == 8
    assert resolved["method"]["semantic"]["lsh"]["d"] == 8
    assert resolved["semantic_lsh"]["lsh_gamma"] == 0.45
    assert resolved["supplementary_ablation"] == {
        "schema_version": "wfcllm-supplementary-ablation-identity/v1",
        "study_kind": "supplementary_ablation",
        "family_id": "issue-3-fixed-seed",
        "factor": "d",
        "canonical_level": 8,
        "default_level": 12,
        "canonical_baseline_config_hash": baseline_hash,
        "resolved_config_hash": resolved["supplementary_ablation"][
            "resolved_config_hash"
        ],
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
        "one_factor_at_a_time": True,
        "single_fixed_seed": True,
        "multi_seed_significance_claim": False,
        "replaces_canonical_result": False,
        "formal_eligible": True,
        "diagnostic_only": False,
        "not_official_method": False,
        "runtime": {
            "d": 8,
            "delta": 0.0,
            "tau_suit": 0.5,
            "tau_close": 0.5,
            "closure_band_width": 0.1,
            "closure_low": 0.45,
            "closure_high": 0.55,
            "n_min": 2,
            "max_units": 3,
            "nominal_gamma": 0.45,
            "realized_region_count": 115,
            "realized_region_ratio": 115 / 256,
        },
    }
    assert len(resolved["supplementary_ablation"]["resolved_config_hash"]) == 64
    assert experiment_contract_hash(resolved) != experiment_contract_hash(baseline)
    assert json.dumps(resolved, sort_keys=True) == json.dumps(
        resolve_method_config(canonical, supplementary_ablation=spec),
        sort_keys=True,
    )


def test_artifact_binding_is_conditional_and_fixture_markers_are_fully_nonformal() -> None:
    canonical = load_config(CANONICAL_CONFIG)
    baseline = resolve_method_config(canonical)
    assert build_supplementary_artifact_binding(baseline) == {}

    resolved = resolve_method_config(
        canonical,
        supplementary_ablation={
            "schema_version": "wfcllm-supplementary-ablation-spec/v1",
            "study_kind": "supplementary_ablation",
            "family_id": "issue-3-fixed-seed",
            "factor": "d",
            "level": 8,
            "default_level": 12,
            "canonical_baseline_config_hash": canonical_baseline_config_hash(
                canonical
            ),
            "language": "python",
            "dataset": "humaneval",
            "profile": "full",
        },
    )
    production = build_supplementary_artifact_binding(resolved)
    fixture = build_supplementary_artifact_binding(
        resolved, diagnostic_test_backend=True
    )

    assert production["resolved_config_sha256"] == experiment_contract_hash(
        resolved
    )
    production_identity = production["supplementary_ablation"]
    assert production_identity["formal"] is True
    assert production_identity["formal_eligible"] is True
    assert production_identity["diagnostic_test_backend"] is False
    assert production_identity["diagnostic_only"] is False
    assert production_identity["not_official_method"] is False
    fixture_identity = fixture["supplementary_ablation"]
    assert fixture_identity["formal"] is False
    assert fixture_identity["formal_eligible"] is False
    assert fixture_identity["diagnostic_test_backend"] is True
    assert fixture_identity["diagnostic_only"] is True
    assert fixture_identity["not_official_method"] is True


def test_allowlisted_matrix_collapses_to_fifteen_unique_full_runs() -> None:
    canonical = load_config(CANONICAL_CONFIG)
    baseline_hash = canonical_baseline_config_hash(canonical)
    resolved_hashes: set[str] = set()
    baseline_variant_hashes: set[str] = set()

    for factor, levels in SUPPLEMENTARY_ABLATION_LEVELS.items():
        for level in levels:
            spec = {
                "schema_version": "wfcllm-supplementary-ablation-spec/v1",
                "study_kind": "supplementary_ablation",
                "family_id": "issue-3-fixed-seed",
                "factor": factor,
                "level": level,
                "default_level": SUPPLEMENTARY_ABLATION_DEFAULTS[factor],
                "canonical_baseline_config_hash": baseline_hash,
                "language": "python",
                "dataset": "humaneval",
                "profile": "full",
            }
            resolved = resolve_method_config(
                canonical, supplementary_ablation=spec
            )
            variant_hash = resolved["supplementary_ablation"][
                "resolved_config_hash"
            ]
            resolved_hashes.add(variant_hash)
            if level == SUPPLEMENTARY_ABLATION_DEFAULTS[factor]:
                baseline_variant_hashes.add(variant_hash)

    assert len(baseline_variant_hashes) == 1
    assert len(resolved_hashes) == 15


def test_each_allowlisted_factor_changes_only_its_operational_contract() -> None:
    canonical = load_config(CANONICAL_CONFIG)
    baseline_hash = canonical_baseline_config_hash(canonical)

    def resolve(factor: str, level: int | float) -> dict[str, object]:
        return resolve_method_config(
            canonical,
            supplementary_ablation={
                "schema_version": "wfcllm-supplementary-ablation-spec/v1",
                "study_kind": "supplementary_ablation",
                "family_id": "issue-3-fixed-seed",
                "factor": factor,
                "level": level,
                "default_level": SUPPLEMENTARY_ABLATION_DEFAULTS[factor],
                "canonical_baseline_config_hash": baseline_hash,
                "language": "python",
                "dataset": "humaneval",
                "profile": "full",
            },
        )

    dimension = resolve("d", 14)
    assert dimension["semantic_lsh"]["lsh_d"] == 14
    assert dimension["method"]["semantic"]["lsh"]["d"] == 14
    assert dimension["semantic_lsh"]["lsh_gamma"] == 0.45
    assert dimension["supplementary_ablation"]["runtime"][
        "realized_region_count"
    ] == round(0.45 * 2**14)

    margin = resolve("delta", 0.01)
    assert margin["semantic_lsh"]["semantic_margin"] == 0.01
    assert margin["method"]["semantic"]["lsh"]["margin"] == 0.01

    suitable = resolve("tau_suit", 0.7)
    assert suitable["supplementary_ablation"]["runtime"]["tau_suit"] == 0.7
    assert suitable["method"]["gate"] == resolve_method_config(canonical)[
        "method"
    ]["gate"]

    closure = resolve("tau_close", 0.4)
    assert closure["supplementary_ablation"]["runtime"]["closure_low"] == 0.35
    assert closure["supplementary_ablation"]["runtime"]["closure_high"] == 0.45
    assert closure["supplementary_ablation"]["runtime"][
        "closure_band_width"
    ] == 0.1

    minimum = resolve("n_min", 4)
    assert minimum["detector"]["minimum_reliable_windows"] == 4

    maximum = resolve("max_units", 1)
    assert maximum["method"]["windowing"]["max_units"] == 1
    assert maximum["gate_data"]["window_lengths"] == [1]


def test_all_canonical_full_profiles_are_unchanged_without_ablation_spec() -> None:
    config_dir = ROOT / "configs" / "wfcllm" / "experiments"

    for config_path in sorted(config_dir.glob("*_full.json")):
        canonical = load_config(config_path)
        resolved = resolve_method_config(canonical)

        assert "supplementary_ablation" not in resolved
        assert resolved == resolve_method_config(json.loads(json.dumps(canonical)))


def test_public_cli_loads_strict_ablation_spec_and_test_negative_input(
    tmp_path: Path,
) -> None:
    canonical = load_config(CANONICAL_CONFIG)
    spec_path = tmp_path / "delta.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-supplementary-ablation-spec/v1",
                "study_kind": "supplementary_ablation",
                "family_id": "issue-3-fixed-seed",
                "factor": "delta",
                "level": 0.005,
                "default_level": 0.0,
                "canonical_baseline_config_hash": (
                    canonical_baseline_config_hash(canonical)
                ),
                "language": "python",
                "dataset": "humaneval",
                "profile": "full",
            }
        ),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            "--ablation-spec",
            str(spec_path),
            "--test-negative-input",
            "test-negatives.jsonl",
        ]
    )
    spec = load_supplementary_ablation_spec(args.ablation_spec)
    resolved = resolve_method_config(canonical, supplementary_ablation=spec)

    assert args.test_negative_input == "test-negatives.jsonl"
    assert resolved["supplementary_ablation"]["factor"] == "delta"
    assert resolved["supplementary_ablation"]["canonical_level"] == 0.005


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"factor": "B", "level": 1, "default_level": 1}, "Paper-Covered"),
        ({"factor": "gamma", "level": 0.4, "default_level": 0.45}, "Paper-Covered"),
        ({"factor": "DIPPER", "level": 1, "default_level": 0}, "Paper-Covered"),
        ({"factor": "variable_renaming", "level": 1, "default_level": 0}, "Paper-Covered"),
        ({"factor": "d", "level": 16, "default_level": 12}, "level=16"),
        ({"factor": "d", "level": True, "default_level": 12}, "must be numeric"),
        ({"factor": "delta", "level": float("inf"), "default_level": 0.0}, "must be finite"),
        ({"factor": "n_min", "level": 2.0, "default_level": 2}, "must be an integer"),
        ({"factor": "max_units", "level": 0, "default_level": 3}, "level=0"),
        ({"factor": "tau_suit", "default_level": 0.3}, "default_level"),
        ({"canonical_baseline_config_hash": "0" * 64}, "does not match"),
        ({"language": "java"}, "only supports python/humaneval/full"),
        ({"dataset": "mbpp"}, "only supports python/humaneval/full"),
        ({"profile": "fast"}, "only supports python/humaneval/full"),
    ],
)
def test_ablation_spec_rejects_non_allowlisted_study_requests(
    patch: dict[str, object],
    message: str,
) -> None:
    canonical = load_config(CANONICAL_CONFIG)
    spec: dict[str, object] = {
        "schema_version": "wfcllm-supplementary-ablation-spec/v1",
        "study_kind": "supplementary_ablation",
        "family_id": "issue-3-fixed-seed",
        "factor": "tau_suit",
        "level": 0.7,
        "default_level": 0.5,
        "canonical_baseline_config_hash": canonical_baseline_config_hash(
            canonical
        ),
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
    }
    spec.update(patch)

    with pytest.raises(ValueError, match=message):
        resolve_method_config(canonical, supplementary_ablation=spec)


def test_ablation_spec_rejects_nested_overrides_and_unsafe_json(
    tmp_path: Path,
) -> None:
    canonical = load_config(CANONICAL_CONFIG)
    baseline_hash = canonical_baseline_config_hash(canonical)
    spec = {
        "schema_version": "wfcllm-supplementary-ablation-spec/v1",
        "study_kind": "supplementary_ablation",
        "family_id": "issue-3-fixed-seed",
        "factor": "d",
        "level": 8,
        "default_level": 12,
        "canonical_baseline_config_hash": baseline_hash,
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
        "method": {"semantic": {"lsh": {"d": 8}}},
    }
    with pytest.raises(ValueError, match="schema mismatch"):
        resolve_method_config(canonical, supplementary_ablation=spec)

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"factor":"d","factor":"delta"}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate key"):
        load_supplementary_ablation_spec(duplicate)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"level":NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite"):
        load_supplementary_ablation_spec(nonfinite)


def test_stable_margin_uses_the_same_strict_delta_boundary() -> None:
    at_boundary = LshProbeResult(
        signature=(1, 0),
        margin=0.005,
        hit=True,
        stable=True,
        stable_across_precision_modes=True,
        stable_across_batch_modes=True,
    )
    above_boundary = LshProbeResult(
        signature=(1, 0),
        margin=0.0050001,
        hit=True,
        stable=True,
        stable_across_precision_modes=True,
        stable_across_batch_modes=True,
    )

    assert at_boundary.is_reliable_hit(configured_margin=0.005) is False
    assert above_boundary.is_reliable_hit(configured_margin=0.005) is True


def test_ablation_negative_corpora_are_independent_and_cover_target_tasks(
    tmp_path: Path,
) -> None:
    calibration_path = tmp_path / "calibration.jsonl"
    test_path = tmp_path / "test.jsonl"
    calibration_path.write_text(
        json.dumps(
            {
                "id": "held-out/calibration/0",
                "dataset": "humaneval",
                "prompt": "calibration prompt",
                "final_code": "def calibration():\n    return 0\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    test_rows = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "target prompt 0",
            "final_code": "def candidate_0():\n    return 0\n",
        },
        {
            "id": "HumanEval/1",
            "dataset": "humaneval",
            "prompt": "target prompt 1",
            "final_code": "def candidate_1():\n    return 1\n",
        },
    ]
    test_path.write_text(
        "".join(json.dumps(row) + "\n" for row in test_rows),
        encoding="utf-8",
    )

    identity = validate_ablation_negative_corpora(
        calibration_path,
        test_path,
        target_task_ids={"HumanEval/0", "HumanEval/1"},
        dataset="humaneval",
    )

    assert identity["calibration_count"] == 1
    assert identity["test_count"] == 2
    assert identity["calibration_sha256"] != identity["test_sha256"]

    calibration_path.write_text(
        json.dumps(
            {
                **test_rows[0],
                "id": "held-out/calibration/0",
                "prompt": "target prompt 0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap by prompt"):
        validate_ablation_negative_corpora(
            calibration_path,
            test_path,
            target_task_ids={"HumanEval/0", "HumanEval/1"},
            dataset="humaneval",
        )


def test_ablation_actual_fpr_uses_independent_test_negative_denominator(
    tmp_path: Path,
) -> None:
    canonical = load_config(CANONICAL_CONFIG)
    resolved = resolve_method_config(
        canonical,
        supplementary_ablation={
            "schema_version": "wfcllm-supplementary-ablation-spec/v1",
            "study_kind": "supplementary_ablation",
            "family_id": "issue-3-fixed-seed",
            "factor": "n_min",
            "level": 3,
            "default_level": 2,
            "canonical_baseline_config_hash": (
                canonical_baseline_config_hash(canonical)
            ),
            "language": "python",
            "dataset": "humaneval",
            "profile": "full",
        },
    )
    detection_dir = tmp_path / "detection"
    calibration_dir = tmp_path / "calibration"
    generation_dir = tmp_path / "generation"
    detection_dir.mkdir()
    calibration_dir.mkdir()
    generation_dir.mkdir()

    def detail(
        sample_id: str,
        *,
        decision: str,
        hit_rate: float,
        reliable_windows: int,
    ) -> dict[str, object]:
        return {
            "id": sample_id,
            "decision": decision,
            "is_watermarked": decision == "watermarked",
            "insufficient_evidence": decision == "insufficient_evidence",
            "hit_rate": hit_rate,
            "reliable_window_count": reliable_windows,
            "hit_count": int(hit_rate > 0),
            "miss_count": int(hit_rate == 0 and reliable_windows > 0),
            "abstain_count": int(reliable_windows == 0),
            "fpr_target": 0.05,
        }

    positive_rows = [
        detail(
            "HumanEval/0",
            decision="watermarked",
            hit_rate=1.0,
            reliable_windows=3,
        ),
        detail(
            "HumanEval/1",
            decision="insufficient_evidence",
            hit_rate=0.0,
            reliable_windows=0,
        ),
    ]
    negative_rows = [
        detail(
            "HumanEval/0",
            decision="watermarked",
            hit_rate=1.0,
            reliable_windows=3,
        ),
        detail(
            "HumanEval/1",
            decision="insufficient_evidence",
            hit_rate=0.0,
            reliable_windows=0,
        ),
    ]
    (detection_dir / "positive_details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in positive_rows),
        encoding="utf-8",
    )
    (detection_dir / "test_negative_details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in negative_rows),
        encoding="utf-8",
    )
    (calibration_dir / "reference_calibration.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (generation_dir / "manifest.json").write_text(
        json.dumps({"sample_count": 2, "carrier_count": 0}) + "\n",
        encoding="utf-8",
    )
    candidate_zero_rows = [
        {
            "id": sample_id,
            "window_index": 0,
            "candidate_index": 0,
            "text": "x = 1",
            "selected": True,
            "evaluation_status": "gate_skipped",
        }
        for sample_id in ("HumanEval/0", "HumanEval/1")
    ]
    (generation_dir / "candidate_sidecar.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in candidate_zero_rows),
        encoding="utf-8",
    )
    (generation_dir / "latency_sidecar.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "id": sample_id,
                    "generation_latency_seconds": 1.0,
                }
            )
            + "\n"
            for sample_id in ("HumanEval/0", "HumanEval/1")
        ),
        encoding="utf-8",
    )

    metrics = _gated_report_metrics_from_artifacts(tmp_path, resolved)
    ablation = metrics["ablation_detection"]

    assert ablation["actual_false_positive_count"] == 1
    assert ablation["actual_negative_count"] == 2
    assert ablation["actual_fpr"] == 0.5
    assert ablation["detected_count"] == 1
    assert ablation["positive_count"] == 2
    assert ablation["tpr"] == 0.5
    assert ablation["positive_insufficient_evidence_count"] == 1
    assert ablation["negative_insufficient_evidence_count"] == 1


def test_mechanism_funnel_and_end_to_end_latency_have_explicit_denominators() -> None:
    candidates = [
        {
            "id": "a",
            "window_index": 0,
            "candidate_index": 0,
            "text": "x = 1",
            "selected": False,
            "evaluation_status": "evaluated_original",
            "structure_ok": True,
            "semantic_preservation_passed": True,
            "keyed_lsh_scored": True,
            "semantic_stable": False,
            "semantic_hit": False,
        },
        {
            "id": "a",
            "window_index": 0,
            "candidate_index": 1,
            "text": "x = 1",
            "selected": False,
            "evaluation_status": "keyed_lsh_evaluated",
            "structure_ok": True,
            "semantic_preservation_passed": True,
            "keyed_lsh_scored": True,
            "semantic_stable": False,
            "semantic_hit": False,
        },
        {
            "id": "a",
            "window_index": 0,
            "candidate_index": 2,
            "text": "x = 2",
            "selected": True,
            "evaluation_status": "keyed_lsh_evaluated",
            "structure_ok": True,
            "semantic_preservation_passed": True,
            "keyed_lsh_scored": True,
            "semantic_stable": True,
            "semantic_hit": True,
        },
        {
            "id": "b",
            "window_index": 0,
            "candidate_index": 0,
            "text": "y = 1",
            "selected": True,
            "evaluation_status": "evaluated_original",
            "structure_ok": True,
            "semantic_preservation_passed": True,
            "keyed_lsh_scored": True,
            "semantic_stable": False,
            "semantic_hit": False,
        },
        *[
            {
                "id": "b",
                "window_index": 0,
                "candidate_index": index,
                "text": f"y = {index + 1}",
                "selected": False,
                "evaluation_status": "keyed_lsh_evaluated",
                "structure_ok": True,
                "semantic_preservation_passed": True,
                "keyed_lsh_scored": True,
                "semantic_stable": index == 3,
                "semantic_hit": index == 3,
            }
            for index in (1, 2, 3)
        ],
    ]

    report = build_supplementary_mechanism_report(
        generation_manifest={"sample_count": 2, "carrier_count": 2},
        candidate_rows=candidates,
        latency_rows=[
            {"id": "a", "generation_latency_seconds": 1.0},
            {"id": "b", "generation_latency_seconds": 3.0},
        ],
        max_rewrites=3,
        baseline_mean_latency_seconds=1.0,
    )

    funnel = report["mechanism_funnel"]
    assert funnel["carrier"]["count"] == 2
    assert funnel["carrier"]["denominator"] == 2
    assert funnel["carrier"]["per_program"] == {"a": 1, "b": 1}
    assert funnel["carrier"]["mean_per_program"] == 1.0
    assert funnel["suitable_pass"]["count"] == 2
    assert funnel["candidate_zero_accept"]["count"] == 1
    assert funnel["rewrite_attempt"]["count"] == 5
    assert funnel["duplicate"]["count"] == 1
    assert funnel["structurally_valid"]["count"] == 5
    assert funnel["detector_recoverable"]["count"] == 5
    assert funnel["stable_margin"]["count"] == 2
    assert funnel["keyed_hit"]["count"] == 2
    assert funnel["final_accepted"]["count"] == 1
    assert funnel["budget_exhausted"]["count"] == 1
    assert report["latency"] == {
        "definition": "base_generation_start_to_final_program_delivery_monotonic/v1",
        "sample_count": 2,
        "per_sample": [
            {"id": "a", "generation_latency_seconds": 1.0},
            {"id": "b", "generation_latency_seconds": 3.0},
        ],
        "total_seconds": 4.0,
        "mean_seconds": 2.0,
        "median_seconds": 2.0,
        "p95_seconds": 3.0,
        "baseline_relative_mean_multiplier": 2.0,
    }
