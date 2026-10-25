from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.evaluation.supplementary_experiments import (
    PASSK_SCHEMA_VERSION,
    PLAN_SCHEMA_VERSION,
    RENAME_MANIFEST_SCHEMA_VERSION,
    SPEC_SCHEMA_VERSION,
    STUDY_KIND,
    SupplementaryExperimentSpec,
    aggregate_detection_conditions,
    aggregate_passk,
    build_all_experiment_plan,
    load_supplementary_experiment_spec,
    pass_at_k,
    prepare_rename_intensity_inputs,
    rename_code,
    write_current_study_specs,
)
from wfcllm.cli.runners import _supplementary_binding
from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.gate.production import experiment_contract_hash


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_all_experiment_plan_covers_server_experiment_families() -> None:
    plan = build_all_experiment_plan()

    assert plan["schema_version"] == PLAN_SCHEMA_VERSION
    families = {
        str(family["family_id"]): family
        for family in plan["families"]
        if isinstance(family, dict)
    }

    assert {
        "multilingual-full-matrix",
        "formal-mechanism-ablation",
        "proposal-budget",
        "gamma-d-key-confirmation",
        "rename-intensity",
        "dipper-rename-robustness",
        "passk",
    } <= set(families)
    assert families["formal-mechanism-ablation"]["execution_kind"] == (
        "fresh_current_ablation_runs"
    )
    assert families["rename-intensity"]["execution_kind"] == (
        "post_run_attack_and_redetect"
    )


def test_current_study_spec_resolves_paper_covered_runtime_factor() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/wfcllm/experiments/python_humaneval_full.json"
    canonical = load_config(config_path)
    baseline = resolve_method_config(canonical)
    spec = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "study_kind": STUDY_KIND,
        "family_id": "paper-covered",
        "factor": "B",
        "level": 6,
        "default_level": 18,
        "canonical_baseline_config_hash": experiment_contract_hash(baseline),
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
    }

    resolved = resolve_method_config(
        canonical,
        supplementary_experiment=SupplementaryExperimentSpec.from_mapping(spec),
    )

    assert resolved["method"]["rewrite"]["max_attempts"] == 6
    assert resolved["gate_data"]["rewrite_count"] == 6
    assert resolved["gate_data"]["rewrite_budgets"] == [1, 3, 6]
    study = resolved["supplementary_experiment"]
    assert study["study_kind"] == STUDY_KIND
    assert study["not_official_method"] is True
    assert len(study["resolved_config_hash"]) == 64


def test_current_study_artifact_binding_is_nonformal() -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/wfcllm/experiments/python_humaneval_full.json"
    canonical = load_config(config_path)
    baseline = resolve_method_config(canonical)
    spec = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "study_kind": STUDY_KIND,
        "family_id": "paper-covered",
        "factor": "B",
        "level": 6,
        "default_level": 18,
        "canonical_baseline_config_hash": experiment_contract_hash(baseline),
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
    }
    resolved = resolve_method_config(
        canonical,
        supplementary_experiment=SupplementaryExperimentSpec.from_mapping(spec),
    )

    binding = _supplementary_binding(resolved, diagnostic_test_backend=True)

    assert binding["resolved_config_sha256"] == experiment_contract_hash(
        resolved
    )
    identity = binding["supplementary_experiment"]
    assert identity["study_kind"] == STUDY_KIND
    assert identity["formal"] is False
    assert identity["formal_eligible"] is False
    assert identity["diagnostic_only"] is True
    assert identity["diagnostic_test_backend"] is True
    assert identity["not_official_method"] is True


def test_smoke_runtime_limits_do_not_change_current_study_baseline_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config_path = root / "configs/wfcllm/experiments/python_humaneval_full.json"
    canonical = load_config(config_path)
    baseline = resolve_method_config(canonical)
    spec = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "study_kind": STUDY_KIND,
        "family_id": "paper-covered",
        "factor": "B",
        "level": 6,
        "default_level": 18,
        "canonical_baseline_config_hash": experiment_contract_hash(baseline),
        "language": "python",
        "dataset": "humaneval",
        "profile": "full",
    }
    formal_resolved = resolve_method_config(
        canonical,
        supplementary_experiment=SupplementaryExperimentSpec.from_mapping(spec),
    )

    monkeypatch.setenv("WFCLLM_SMOKE_SAMPLE_COUNT", "20")
    smoke_resolved = resolve_method_config(
        canonical,
        supplementary_experiment=SupplementaryExperimentSpec.from_mapping(spec),
    )

    assert smoke_resolved["supplementary_experiment"] == formal_resolved[
        "supplementary_experiment"
    ]
    assert smoke_resolved["runtime"]["smoke_sample_count"] == 20
    assert smoke_resolved["calibration"]["target_negative_count"] == 20
    assert (
        smoke_resolved["calibration"]["method"]
        == formal_resolved["calibration"]["method"]
    )
    assert (
        smoke_resolved["calibration"]["group_by"]
        == formal_resolved["calibration"]["group_by"]
    )
    assert (
        smoke_resolved["detector"]["minimum_reliable_windows"]
        == formal_resolved["detector"]["minimum_reliable_windows"]
    )


def test_current_study_spec_rejects_unsupported_factor_and_hash(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": SPEC_SCHEMA_VERSION,
                "study_kind": STUDY_KIND,
                "family_id": "paper-covered",
                "factor": "temperature",
                "level": 0.2,
                "default_level": 0.0,
                "canonical_baseline_config_hash": "0" * 64,
                "language": "python",
                "dataset": "humaneval",
                "profile": "full",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not allowlisted"):
        load_supplementary_experiment_spec(path)


def test_write_specs_outputs_formal_and_current_study_specs(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = write_current_study_specs(
        canonical_config_path=(
            root / "configs/wfcllm/experiments/python_humaneval_full.json"
        ),
        output_root=tmp_path,
        family_id="issue-3",
    )

    assert manifest["formal_ablation_spec_count"] == 15
    assert manifest["current_study_spec_count"] == 12
    commands = (tmp_path / "run_commands.sh").read_text(encoding="utf-8")
    assert "ABLATION_SPEC=" in commands
    assert "SUPPLEMENTARY_EXPERIMENT_SPEC=" in commands
    assert "run_python_humaneval_full.sh" in commands


def test_rename_code_preserves_public_entry_signature() -> None:
    transformed, metadata = rename_code(
        code=(
            "def solve(x):\n"
            "    total = 0\n"
            "    for item in x:\n"
            "        total += item\n"
            "    return total\n"
        ),
        entry_point="solve",
        task_id="HumanEval/0",
        name_seed=7,
        selection_seed=20260722,
        fraction=1.0,
    )

    assert transformed.startswith("def solve(x):")
    assert "total" not in transformed
    assert metadata["attack_applied"] is True
    assert metadata["entry_signature_preserved"] is True


def test_prepare_rename_intensity_writes_strict_detector_inputs(tmp_path: Path) -> None:
    positive = tmp_path / "positive.jsonl"
    negative = tmp_path / "negative.jsonl"
    entry_points = tmp_path / "entry_points.jsonl"
    rows = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def solve(x):",
            "final_code": (
                "def solve(x):\n"
                "    total = 0\n"
                "    for item in x:\n"
                "        total += item\n"
                "    return total\n"
            ),
        }
    ]
    write_jsonl(positive, rows)
    write_jsonl(negative, [{**rows[0], "id": "HumanEval/1"}])
    write_jsonl(
        entry_points,
        [
            {"id": "HumanEval/0", "entry_point": "solve"},
            {"id": "HumanEval/1", "entry_point": "solve"},
        ],
    )

    manifest = prepare_rename_intensity_inputs(
        positive_input=positive,
        negative_input=negative,
        entry_points=entry_points,
        output_root=tmp_path / "rename",
        conditions={"rename100": 1.0},
    )

    assert manifest["schema_version"] == RENAME_MANIFEST_SCHEMA_VERSION
    attacked = [
        json.loads(line)
        for line in (tmp_path / "rename/inputs/rename100/positive.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert set(attacked[0]) == {"id", "dataset", "prompt", "final_code"}
    assert attacked[0]["final_code"].startswith("def solve(x):")
    assert "total" not in attacked[0]["final_code"]


def test_aggregate_detection_conditions_reports_full_and_conditional(tmp_path: Path) -> None:
    for condition, pos_rate in (("original", 1.0), ("rename", 0.5)):
        write_jsonl(
            tmp_path / "detection" / condition / "positive_details.jsonl",
            [
                {"id": "p0", "reliable_window_count": 3, "hit_rate": pos_rate},
                {"id": "p1", "reliable_window_count": 0, "hit_rate": 0.0},
            ],
        )
        write_jsonl(
            tmp_path / "detection" / condition / "negative_details.jsonl",
            [
                {"id": "n0", "reliable_window_count": 3, "hit_rate": 0.0},
                {"id": "n1", "reliable_window_count": 0, "hit_rate": 0.0},
            ],
        )

    result = aggregate_detection_conditions(
        root=tmp_path,
        conditions=("original", "rename"),
        output_name="rename_intensity_metrics",
    )

    assert "drop_from_original_pp" in result["conditions"]["rename"]
    assert result["conditions"]["original"]["full"]["positive_population_n"] == 2
    assert (tmp_path / "metrics/rename_intensity_metrics.json").is_file()
    assert (tmp_path / "metrics/rename_intensity_metrics.md").is_file()


def test_aggregate_detection_conditions_marks_empty_conditional_subset(
    tmp_path: Path,
) -> None:
    for condition in ("original", "rename"):
        write_jsonl(
            tmp_path / "detection" / condition / "positive_details.jsonl",
            [{"id": "p0", "reliable_window_count": 1, "hit_rate": 1.0}],
        )
        write_jsonl(
            tmp_path / "detection" / condition / "negative_details.jsonl",
            [{"id": "n0", "reliable_window_count": 0, "hit_rate": 0.0}],
        )

    result = aggregate_detection_conditions(
        root=tmp_path,
        conditions=("original", "rename"),
        output_name="rename_intensity_metrics",
        minimum_reliable_windows=1,
    )

    conditional = result["conditions"]["original"]["conditional"]
    assert conditional["insufficient_population"] is True
    assert conditional["positive_coverage"] == 1.0
    assert conditional["negative_coverage"] == 0.0
    assert (tmp_path / "metrics/rename_intensity_metrics.json").is_file()
    assert (tmp_path / "metrics/rename_intensity_metrics.md").is_file()


def test_pass_at_k_and_aggregate_passk(tmp_path: Path) -> None:
    write_jsonl(
        tmp_path / "generation" / "all_samples.jsonl",
        [
            {"id": "task/0", "sample_index": 0},
            {"id": "task/0", "sample_index": 1},
            {"id": "task/1", "sample_index": 0},
            {"id": "task/1", "sample_index": 1},
        ],
    )
    write_jsonl(
        tmp_path / "evaluation" / "sample_00" / "results.jsonl",
        [{"id": "task/0", "passed": True}, {"id": "task/1", "passed": False}],
    )
    write_jsonl(
        tmp_path / "evaluation" / "sample_01" / "results.jsonl",
        [{"id": "task/0", "passed": False}, {"id": "task/1", "passed": True}],
    )

    assert pass_at_k(10, 1, 10) == 1.0
    with pytest.raises(ValueError, match="n >= k"):
        pass_at_k(2, 1, 3)

    summary = aggregate_passk(
        root=tmp_path,
        num_samples=2,
        k_values=(1, 2),
        expected_task_count=2,
    )

    assert summary["schema_version"] == PASSK_SCHEMA_VERSION
    assert summary["pass_at_k"]["1"] == 0.5
    assert summary["pass_at_k"]["2"] == 1.0
    assert summary["posthoc_only"] is True
    assert (tmp_path / "results/passk_summary.json").is_file()
