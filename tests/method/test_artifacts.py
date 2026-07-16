from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.method.artifacts import (
    FinalCodeRecord,
    RunPaths,
    build_run_paths,
    write_jsonl_rows,
)


def test_final_code_record_exact_four_fields() -> None:
    record = FinalCodeRecord(
        id="HumanEval/0",
        dataset="humaneval",
        prompt="def foo():\n",
        final_code="def foo():\n    return 1\n",
    )

    assert record.to_dict() == {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }


def test_final_code_record_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="id"):
        FinalCodeRecord(id="", dataset="humaneval", prompt="", final_code="")


def test_build_run_paths_uses_new_layout(tmp_path: Path) -> None:
    paths = build_run_paths(tmp_path, "20260708_153000_evidence_retry_seed7x3")

    assert isinstance(paths, RunPaths)
    assert (
        paths.final_code_input
        == tmp_path
        / "20260708_153000_evidence_retry_seed7x3"
        / "inputs"
        / "final_code.jsonl"
    )
    assert (
        paths.audit_log
        == tmp_path
        / "20260708_153000_evidence_retry_seed7x3"
        / "generation"
        / "audit.jsonl"
    )
    assert (
        paths.detector_integrity
        == tmp_path
        / "20260708_153000_evidence_retry_seed7x3"
        / "audit"
        / "detector_input_integrity.json"
    )


def test_run_paths_include_all_gate_artifacts_without_changing_final_code_path(
    tmp_path: Path,
) -> None:
    paths = build_run_paths(tmp_path, "run-1")

    assert paths.final_code_input == tmp_path / "run-1" / "inputs" / "final_code.jsonl"
    assert paths.gate_data_manifest == tmp_path / "run-1" / "gate-data" / "manifest.json"
    assert paths.gate_window_groups == tmp_path / "run-1" / "gate-data" / "window_groups.jsonl"
    assert paths.gate_candidate_attempts == tmp_path / "run-1" / "gate-data" / "candidate_attempts.jsonl"
    assert paths.gate_labels == tmp_path / "run-1" / "gate-data" / "labels.jsonl"
    assert paths.gate_split_manifest == tmp_path / "run-1" / "gate-data" / "split_manifest.json"
    assert paths.gate_training_key_bank_manifest == tmp_path / "run-1" / "gate-data" / "training_key_bank_manifest.json"
    assert paths.gate_feasibility_summary == tmp_path / "run-1" / "gate-data" / "feasibility_summary.json"
    assert paths.gate_resolved_training_config == tmp_path / "run-1" / "gate-train" / "resolved_training_config.json"
    assert paths.gate_training_metrics == tmp_path / "run-1" / "gate-train" / "training_metrics.jsonl"
    assert paths.gate_development_summary == tmp_path / "run-1" / "gate-train" / "development_summary.json"
    assert paths.gate_checkpoints_dir == tmp_path / "run-1" / "gate-train" / "checkpoints"
    assert paths.gate_candidate_bundle_dir == tmp_path / "run-1" / "gate-train" / "candidate_bundle"
    assert paths.gate_candidate_bundle_manifest == tmp_path / "run-1" / "gate-train" / "candidate_bundle_manifest.json"
    assert paths.gate_validation_summary == tmp_path / "run-1" / "gate-validate" / "validation_summary.json"
    assert paths.gate_agreement_details == tmp_path / "run-1" / "gate-validate" / "agreement_details.jsonl"
    assert paths.gate_bundle_dir == tmp_path / "run-1" / "gate-validate" / "bundle"
    assert paths.gate_bundle_manifest == tmp_path / "run-1" / "gate-validate" / "gate_bundle_manifest.json"
    assert paths.audit_log == tmp_path / "run-1" / "generation" / "audit.jsonl"
    assert paths.candidate_sidecar == tmp_path / "run-1" / "generation" / "candidate_sidecar.jsonl"
    assert paths.raw_attempt_summary == tmp_path / "run-1" / "generation" / "raw_attempt_summary.jsonl"
    assert paths.reference_calibration == tmp_path / "run-1" / "calibration" / "reference_calibration.json"
    assert paths.positive_details == tmp_path / "run-1" / "detection" / "positive_details.jsonl"
    assert paths.reference_negative_details == tmp_path / "run-1" / "detection" / "reference_negative_details.jsonl"
    assert paths.model_negative_details == tmp_path / "run-1" / "detection" / "model_negative_details.jsonl"


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "/absolute", "a/b", "a\\b"])
def test_build_run_paths_rejects_unsafe_run_id(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_run_paths(tmp_path, run_id)


@pytest.mark.parametrize("run_id", ["line\nbreak", "tab\tname", "c1\x85name", "\x7fdel"])
def test_build_run_paths_rejects_control_characters(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_run_paths(tmp_path, run_id)


def test_build_run_paths_limits_encoded_component_bytes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="run_id"):
        build_run_paths(tmp_path, "中" * 100)


def test_write_jsonl_rows_rejects_non_json_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON-safe"):
        write_jsonl_rows(tmp_path / "rows.jsonl", [{"score": float("nan")}])


def test_write_jsonl_rows_writes_utf8_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl_rows(path, [{"id": "a", "text": "中文"}])

    assert json.loads(path.read_text(encoding="utf-8").strip()) == {
        "id": "a",
        "text": "中文",
    }
