from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.evaluation.metric_contract import (
    AUROC_DEFINITION,
    SCHEMA_VERSION,
    extract_metric_contract,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _standard_report(
    *,
    ids: list[str],
    decisions: list[str],
    hit_rates: list[float],
    buckets: dict[str, list[float]] | None,
    target_fpr: float = 0.31,
    minimum_reliable_windows: int = 1,
    window_contract_version: str = "js-statement-window/v1",
    extra: dict | None = None,
) -> dict:
    calibration = {
        "target_fpr": target_fpr,
        "minimum_reliable_windows": minimum_reliable_windows,
        "window_contract_version": window_contract_version,
        "key_identifier_sha256": "k" * 64,
        "semantic_encoder_sha256": "s" * 64,
        "schema_version": "wfcllm-gated-calibration/v1",
    }
    if buckets is not None:
        calibration["reliable_window_count_buckets"] = buckets
    report = {
        "method": "gated_semantic_window_v1",
        "detector_mode": "wfcllm-gated-semantic-window/v1",
        "calibration": calibration,
        "detection_curve": [
            {"id": ids[i], "decision": decisions[i], "hit_rate": hit_rates[i], "p_value": 0.2}
            for i in range(len(ids))
        ],
    }
    if extra:
        report.update(extra)
    return report


def _make_standard_run(tmp_path: Path, name: str = "synthetic-run") -> Path:
    """Full synthetic run: report + curve + gate-data manifest + run_state."""
    run_dir = tmp_path / name
    report = _standard_report(
        ids=["JS/0", "JS/1", "JS/2", "JS/3"],
        decisions=[
            "watermarked",
            "not_watermarked",
            "insufficient_evidence",
            "watermarked",
        ],
        hit_rates=[0.9, 0.2, 0.0, 0.8],
        buckets={"1": [0.1, 0.2, 0.5]},
        extra={"gate_validation_skipped": True, "unvalidated_gate_candidate": True},
    )
    _write_json(run_dir / "reports" / "reference_report.json", report)
    _write_json(
        run_dir / "gate-data" / "manifest.json",
        {
            "schema_version": "wfcllm-gate-data-manifest/v1",
            "parser_contract": "js-statement-window/v1",
            "formal_eligible": False,
            "diagnostic_only": True,
            "experimental_only": True,
            "not_official_method": True,
        },
    )
    _write_json(
        run_dir / "run_state.json",
        {
            "generate": {
                "done": True,
                "output_path": "/srv/run/inputs/final_code.jsonl",
                "gate_validation_skipped": True,
                "unvalidated_gate_candidate": True,
            },
            "detect": {
                "done": True,
                "details_path": "/srv/run/detection/positive_details_regen_failed84_v1.jsonl",
                "gate_validation_skipped": True,
                "unvalidated_gate_candidate": True,
            },
        },
    )
    return run_dir


def test_standard_run_field_completeness_and_values(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["run_id"] == "synthetic-run"
    assert record["dataset"] == "humanevalpack"
    assert record["language"] == "js"
    assert record["sample_count"] == 4
    assert record["target_fpr"] == pytest.approx(0.31)
    assert record["minimum_reliable_windows"] == 1
    assert record["key_identifier_sha256"] == "k" * 64
    assert record["semantic_encoder_sha256"] == "s" * 64
    assert record["auroc_definition"] == AUROC_DEFINITION


def test_standard_run_full_sample_tpr_is_not_conditional(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    # insufficient_evidence stays in the denominator, out of the numerator.
    assert record["detected_count"] == 2
    assert record["tpr"] == pytest.approx(0.5)
    assert record["tpr_is_conditional"] is False
    assert record["conditional_rule"] is None


def test_standard_run_auroc_pooled_hit_rate_rank_with_ties(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    # positives [0.9, 0.2, 0.0, 0.8] vs pooled negatives [0.1, 0.2, 0.5]
    # Mann-Whitney with ties as 0.5: (3 + 1.5 + 0 + 3) / 12 = 0.625
    assert record["auroc"] == pytest.approx(0.625)


def test_standard_run_eligibility_and_bypass_passthrough(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    assert record["eligibility"] == {
        "formal_eligible": False,
        "diagnostic_only": True,
        "experimental_only": True,
    }
    assert record["experimental_bypass"] is True
    assert record["gate_validation_skipped"] is True
    assert record["unvalidated_gate_candidate"] is True


def test_standard_run_regen_details_path_fills_caveat(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    assert any("regen" in caveat for caveat in record["caveats"])
    assert any("84" in caveat for caveat in record["caveats"])


def test_standard_run_relaxed_target_fpr_fills_caveat(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    assert any("target_fpr" in caveat for caveat in record["caveats"])


def test_standard_run_missing_pass_artifacts_yield_null_and_caveat(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    assert record["pass_rate"] is None
    assert record["pass_metric"] is None
    assert any("pass evaluation artifacts" in caveat for caveat in record["caveats"])


def test_generation_model_null_with_caveat_when_not_in_artifacts(tmp_path) -> None:
    record = extract_metric_contract(_make_standard_run(tmp_path))

    assert record["generation_model"] is None
    assert any("generation model" in caveat.lower() for caveat in record["caveats"])


def test_missing_auroc_source_yields_null_and_caveat(tmp_path) -> None:
    run_dir = tmp_path / "no-buckets"
    report = _standard_report(
        ids=["Java/0", "Java/1"],
        decisions=["watermarked", "not_watermarked"],
        hit_rates=[0.7, 0.1],
        buckets=None,
        window_contract_version="java-statement-window/v1",
    )
    _write_json(run_dir / "reports" / "reference_report.json", report)

    record = extract_metric_contract(run_dir)

    assert record["auroc"] is None
    assert record["auroc_definition"] == AUROC_DEFINITION
    assert any("auroc" in caveat.lower() for caveat in record["caveats"])
    assert record["language"] == "java"
    assert record["dataset"] == "humanevalpack"


def test_identity_from_window_contract_when_report_has_no_identity(tmp_path) -> None:
    run_dir = tmp_path / "legacy-python"
    report = _standard_report(
        ids=["HumanEval/0", "HumanEval/1"],
        decisions=["watermarked", "watermarked"],
        hit_rates=[0.9, 0.8],
        buckets={"2": [0.1]},
        target_fpr=0.05,
        minimum_reliable_windows=2,
        window_contract_version="python-statement-window/v1",
    )
    assert "language" not in report and "dataset" not in report
    _write_json(run_dir / "reports" / "reference_report.json", report)

    record = extract_metric_contract(run_dir)

    assert record["language"] == "python"
    assert record["dataset"] == "humaneval"
    # target_fpr at the conventional budget: no relaxation caveat.
    assert not any("relaxed" in caveat for caveat in record["caveats"])


def test_new_report_identity_fields_take_priority(tmp_path) -> None:
    run_dir = tmp_path / "new-report"
    report = _standard_report(
        ids=["HumanEval/0"],
        decisions=["watermarked"],
        hit_rates=[0.9],
        buckets={"2": [0.1]},
        target_fpr=0.05,
        minimum_reliable_windows=2,
        window_contract_version="python-statement-window/v1",
        extra={
            "dataset": "mbpp",
            "language": "python",
            "generation_model": "opencoder-8b",
        },
    )
    _write_json(run_dir / "reports" / "reference_report.json", report)

    record = extract_metric_contract(run_dir)

    assert record["dataset"] == "mbpp"
    assert record["language"] == "python"
    assert record["generation_model"] == "opencoder-8b"
    assert not any("generation model" in caveat.lower() for caveat in record["caveats"])


def test_unconditional_output_with_poor_metrics(tmp_path) -> None:
    """A record is produced even when every metric misses every target."""
    run_dir = tmp_path / "poor-run"
    report = _standard_report(
        ids=["JS/0", "JS/1"],
        decisions=["not_watermarked", "not_watermarked"],
        hit_rates=[0.0, 0.0],
        buckets=None,
    )
    _write_json(run_dir / "reports" / "reference_report.json", report)

    record = extract_metric_contract(run_dir)

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["tpr"] == pytest.approx(0.0)
    assert record["detected_count"] == 0
    assert record["auroc"] is None
    assert record["pass_rate"] is None


def test_basic_report_layout(tmp_path) -> None:
    run_dir = tmp_path / "basic-target" / "run"
    _write_json(
        run_dir / "reports" / "reference_report.json",
        {
            "basic_experiment": True,
            "dataset": "humanevalpack",
            "language": "cpp",
            "profile": "full",
            "sample_count": 164,
            "decision_counts": {"generated": 164},
            "official_watermark_claim": False,
            "schema_version": "wfcllm-basic-report/v1",
        },
    )

    record = extract_metric_contract(run_dir)

    assert record["run_id"] == "basic-target"
    assert record["dataset"] == "humanevalpack"
    assert record["language"] == "cpp"
    assert record["sample_count"] == 164
    assert record["tpr"] is None
    assert record["detected_count"] is None
    assert record["auroc"] is None
    assert record["pass_rate"] is None
    assert record["target_fpr"] is None
    assert any("basic" in caveat.lower() for caveat in record["caveats"])


def test_js_layout_probes_run_state_and_gate_data_across_levels(tmp_path) -> None:
    """run_state.json above run/, gate-data inside run/ (JS target layout)."""
    outer = tmp_path / "fast-fullgate-js-full164"
    run_dir = outer / "run"
    report = _standard_report(
        ids=["JS/0", "JS/1"],
        decisions=["watermarked", "not_watermarked"],
        hit_rates=[0.9, 0.1],
        buckets={"1": [0.2]},
    )
    _write_json(run_dir / "reports" / "reference_report.json", report)
    _write_json(
        run_dir / "gate-data" / "manifest.json",
        {
            "parser_contract": "js-statement-window/v1",
            "formal_eligible": False,
            "diagnostic_only": True,
            "experimental_only": True,
        },
    )
    _write_json(
        outer / "run_state.json",
        {
            "detect": {
                "done": True,
                "details_path": "/srv/run/detection/positive_details_regen_failed84_v1.jsonl",
            }
        },
    )

    record = extract_metric_contract(run_dir)

    assert record["run_id"] == "fast-fullgate-js-full164"
    assert record["eligibility"]["experimental_only"] is True
    assert any("regen" in caveat for caveat in record["caveats"])


def test_top_level_layout_probes_nested_run_directory(tmp_path) -> None:
    """reports under run/, run_state.json at the top level (OpenCoder layout)."""
    run_dir = tmp_path / "opencoder-style"
    report = _standard_report(
        ids=["HumanEval/0"],
        decisions=["watermarked"],
        hit_rates=[0.9],
        buckets={"2": [0.1]},
        target_fpr=0.05,
        minimum_reliable_windows=2,
        window_contract_version="python-statement-window/v1",
    )
    _write_json(run_dir / "run" / "reports" / "reference_report.json", report)
    _write_json(run_dir / "run_state.json", {"detect": {"done": True}})

    record = extract_metric_contract(run_dir)

    assert record["run_id"] == "opencoder-style"
    assert record["language"] == "python"
    assert record["tpr"] == pytest.approx(1.0)


def test_missing_gate_manifest_yields_null_eligibility_and_caveat(tmp_path) -> None:
    run_dir = tmp_path / "no-gate-data"
    report = _standard_report(
        ids=["HumanEval/0"],
        decisions=["watermarked"],
        hit_rates=[0.9],
        buckets={"2": [0.1]},
        target_fpr=0.05,
        minimum_reliable_windows=2,
        window_contract_version="python-statement-window/v1",
    )
    _write_json(run_dir / "reports" / "reference_report.json", report)

    record = extract_metric_contract(run_dir)

    assert record["eligibility"] == {
        "formal_eligible": None,
        "diagnostic_only": None,
        "experimental_only": None,
    }
    assert record["experimental_bypass"] is None
    assert any("eligibility" in caveat.lower() for caveat in record["caveats"])


def _make_gamma25_run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "gamma25_gate_pass113"
    _write_jsonl(
        run_dir / "final_code.jsonl",
        [
            {"id": "HumanEval/0", "dataset": "humaneval", "prompt": "p", "final_code": "c"},
            {"id": "HumanEval/1", "dataset": "humaneval", "prompt": "p", "final_code": "c"},
            {"id": "HumanEval/2", "dataset": "humaneval", "prompt": "p", "final_code": "c"},
        ],
    )
    _write_json(
        run_dir / "n_ge_3_tpr_summary.json",
        {
            "artifact_type": "wfcllm_conditional_detection_summary",
            "schema_version": "wfcllm-conditional-detection-summary/v1",
            "eligibility_rule": {
                "field": "reliable_window_count",
                "operator": ">=",
                "value": 3,
            },
            "detection_rule": {
                "field": "hit_rate",
                "operator": ">=",
                "threshold": 0.6,
            },
            "positive": {
                "total": 3,
                "eligible": 2,
                "excluded": 1,
                "coverage": 2 / 3,
                "detected": 1,
                "conditional_tpr": 0.5,
            },
            "heldout_negative": {"total": 8, "false_positives": 1, "fpr": 0.125},
        },
    )
    _write_json(
        run_dir / "humaneval_summary.json",
        {
            "artifact_type": "wfcllm_execution_summary",
            "schema_version": "wfcllm-execution-summary/v1",
            "dataset": "humaneval",
            "total": 3,
            "passed": 2,
            "pass_at_1": 2 / 3,
            "pass_at_10": 1.0,
        },
    )
    return run_dir


def test_gamma25_layout_conditional_tpr_labelled(tmp_path) -> None:
    record = extract_metric_contract(_make_gamma25_run(tmp_path))

    assert record["tpr"] == pytest.approx(0.5)
    assert record["tpr_is_conditional"] is True
    assert record["conditional_rule"] == "reliable_window_count>=3;hit_rate>=0.6"
    assert record["detected_count"] == 1
    assert record["sample_count"] == 3
    assert any("conditional" in caveat.lower() for caveat in record["caveats"])


def test_gamma25_layout_identity_and_pass(tmp_path) -> None:
    record = extract_metric_contract(_make_gamma25_run(tmp_path))

    assert record["dataset"] == "humaneval"
    assert record["language"] == "python"
    assert record["pass_rate"] == pytest.approx(2 / 3)
    assert record["pass_metric"] == "pass@1"
    assert record["generation_model"] is None


def test_gamma25_layout_missing_reference_artifacts_are_null(tmp_path) -> None:
    record = extract_metric_contract(_make_gamma25_run(tmp_path))

    assert record["target_fpr"] is None
    assert record["minimum_reliable_windows"] is None
    assert record["key_identifier_sha256"] is None
    assert record["semantic_encoder_sha256"] is None
    assert record["auroc"] is None
    assert any("reference" in caveat.lower() for caveat in record["caveats"])


def test_unrecognized_run_dir_still_produces_record(tmp_path) -> None:
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    record = extract_metric_contract(run_dir)

    assert record["schema_version"] == SCHEMA_VERSION
    assert record["tpr"] is None
    assert any("no recognized" in caveat.lower() for caveat in record["caveats"])


def test_missing_run_dir_raises_value_error(tmp_path) -> None:
    with pytest.raises(ValueError, match="run directory"):
        extract_metric_contract(tmp_path / "does-not-exist")
