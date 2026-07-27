from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.wfcllm_metric_contract as metric_contract_cli
from wfcllm.evaluation.metric_contract import SCHEMA_VERSION


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_current_run(
    tmp_path: Path,
    name: str,
    *,
    include_scores: bool = True,
) -> Path:
    run_dir = tmp_path / name
    _write_json(
        run_dir / "gate-data" / "manifest.json",
        {
            "schema_version": "wfcllm-gate-data-manifest/v1",
            "formal_eligible": True,
            "diagnostic_test_backend": False,
            "diagnostic_only": False,
            "not_official_method": False,
        },
    )
    calibration = {
        "schema_version": "wfcllm-gated-calibration/v1",
        "target_fpr": 0.05,
        "minimum_reliable_windows": 2,
        "key_identifier_sha256": "a" * 64,
        "semantic_encoder_sha256": "b" * 64,
        "gate_bundle_sha256": "c" * 64,
        "thresholds_by_reliable_window_count": {"2": 0.5},
        "empirical_p_value_rule": "right_tail_plus_one/v1",
    }
    if include_scores:
        calibration["reliable_window_count_buckets"] = {"2": [0.1, 0.3]}
    report_path = run_dir / "reports" / "reference_report.json"
    _write_json(
        report_path,
        {
            "method": "gated_semantic_window_v1",
            "detector_mode": "wfcllm-gated-semantic-window/v1",
            "dataset": "humaneval",
            "language": "python",
            "generation_model": "local-model:sha256:" + "d" * 64,
            "calibration": calibration,
            "detection_curve": (
                [
                    {
                        "id": "HumanEval/0",
                        "decision": "watermarked",
                        "hit_rate": 0.9,
                    },
                    {
                        "id": "HumanEval/1",
                        "decision": "not_watermarked",
                        "hit_rate": 0.2,
                    },
                ]
                if include_scores
                else []
            ),
        },
    )
    current_artifacts = {
        "final_code_sha256": run_dir / "inputs" / "final_code.jsonl",
        "calibration_sha256": (
            run_dir / "calibration" / "reference_calibration.json"
        ),
        "positive_details_sha256": (
            run_dir / "detection" / "positive_details.jsonl"
        ),
        "reference_report_sha256": report_path,
    }
    _write_json(
        current_artifacts["final_code_sha256"],
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "prompt",
            "final_code": "code",
        },
    )
    _write_json(current_artifacts["calibration_sha256"], calibration)
    _write_json(
        current_artifacts["positive_details_sha256"],
        {"id": "HumanEval/0", "decision": "watermarked"},
    )
    _write_json(
        run_dir / "audit" / "artifact_integrity.json",
        {
            "ok": True,
            **{
                field: hashlib.sha256(path.read_bytes()).hexdigest()
                for field, path in current_artifacts.items()
            },
        },
    )
    return run_dir


def test_cli_emits_one_current_row_per_run_to_output(tmp_path: Path) -> None:
    first = _make_current_run(tmp_path, "first-run")
    second = _make_current_run(tmp_path, "second-run", include_scores=False)
    output = tmp_path / "contract.jsonl"

    result = metric_contract_cli.main(
        [str(first), str(second), "--output", str(output)]
    )

    assert result == 0
    rows = [
        json.loads(line)
        for line in output.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["run_id"] for row in rows] == ["first-run", "second-run"]
    assert all(row["schema_version"] == SCHEMA_VERSION for row in rows)
    assert rows[0]["tpr"] == pytest.approx(0.5)
    assert rows[1]["tpr"] is None
    assert rows[1]["caveats"]


def test_cli_writes_current_row_to_stdout(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = _make_current_run(tmp_path, "stdout-run")

    assert metric_contract_cli.main(["--run-dir", str(run_dir)]) == 0

    row = json.loads(capsys.readouterr().out)
    assert row["schema_version"] == SCHEMA_VERSION
    assert row["run_id"] == "stdout-run"
    assert row["dataset"] == "humaneval"
    assert row["language"] == "python"


def test_cli_rejects_incomplete_current_layout(tmp_path: Path) -> None:
    incomplete = tmp_path / "incomplete-run"
    _write_json(
        incomplete / "reports" / "reference_report.json",
        {"method": "unsupported"},
    )

    with pytest.raises(ValueError, match="current Gate-only schema|missing"):
        metric_contract_cli.main([str(incomplete)])


def test_cli_requires_at_least_one_run_dir() -> None:
    with pytest.raises(SystemExit):
        metric_contract_cli.main([])
