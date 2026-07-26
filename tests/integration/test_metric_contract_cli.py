from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.wfcllm_metric_contract as metric_contract_cli
from wfcllm.evaluation.metric_contract import SCHEMA_VERSION


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_standard_run(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / name
    _write_json(
        run_dir / "reports" / "reference_report.json",
        {
            "method": "gated_semantic_window_v1",
            "detector_mode": "wfcllm-gated-semantic-window/v1",
            "calibration": {
                "target_fpr": 0.05,
                "minimum_reliable_windows": 2,
                "window_contract_version": "python-statement-window/v1",
                "key_identifier_sha256": "k" * 64,
                "semantic_encoder_sha256": "s" * 64,
                "reliable_window_count_buckets": {"2": [0.1, 0.3]},
            },
            "detection_curve": [
                {"id": "HumanEval/0", "decision": "watermarked", "hit_rate": 0.9},
                {"id": "HumanEval/1", "decision": "not_watermarked", "hit_rate": 0.2},
            ],
        },
    )
    return run_dir


def _make_basic_run(tmp_path: Path, name: str) -> Path:
    run_dir = tmp_path / name
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
    return run_dir


def test_cli_multiple_run_dirs_to_output_file(tmp_path) -> None:
    standard = _make_standard_run(tmp_path, "standard-run")
    basic = _make_basic_run(tmp_path, "basic-run")
    output = tmp_path / "contract.jsonl"

    exit_code = metric_contract_cli.main(
        [str(standard), str(basic), "--output", str(output)]
    )

    assert exit_code == 0
    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert [record["run_id"] for record in records] == ["standard-run", "basic-run"]
    for record in records:
        assert record["schema_version"] == SCHEMA_VERSION
    assert records[0]["tpr"] == pytest.approx(0.5)
    assert records[1]["tpr"] is None


def test_cli_writes_jsonl_to_stdout(tmp_path, capsys) -> None:
    standard = _make_standard_run(tmp_path, "stdout-run")

    exit_code = metric_contract_cli.main([str(standard)])

    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["run_id"] == "stdout-run"
    assert record["dataset"] == "humaneval"
    assert record["language"] == "python"


def test_cli_accepts_repeated_run_dir_option(tmp_path, capsys) -> None:
    first = _make_standard_run(tmp_path, "first-run")
    second = _make_basic_run(tmp_path, "second-run")

    exit_code = metric_contract_cli.main(
        ["--run-dir", str(first), "--run-dir", str(second)]
    )

    assert exit_code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert [json.loads(line)["run_id"] for line in lines] == [
        "first-run",
        "second-run",
    ]


def test_cli_unconditional_output_for_poor_run(tmp_path, capsys) -> None:
    """Missing metrics never suppress the record (ADR 0003)."""
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    exit_code = metric_contract_cli.main([str(run_dir)])

    assert exit_code == 0
    record = json.loads(capsys.readouterr().out.strip())
    assert record["schema_version"] == SCHEMA_VERSION
    assert record["tpr"] is None
    assert record["caveats"]


def test_cli_requires_at_least_one_run_dir() -> None:
    with pytest.raises(SystemExit):
        metric_contract_cli.main([])
