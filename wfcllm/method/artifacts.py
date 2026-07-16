from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FinalCodeRecord:
    id: str
    dataset: str
    prompt: str
    final_code: str

    def __post_init__(self) -> None:
        for field_name in ("id", "dataset", "prompt", "final_code"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
        if not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "prompt": self.prompt,
            "final_code": self.final_code,
        }


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    resolved_config: Path
    method_preset: Path
    final_code_input: Path
    gate_data_manifest: Path
    gate_window_groups: Path
    gate_candidate_attempts: Path
    gate_labels: Path
    gate_split_manifest: Path
    gate_training_key_bank_manifest: Path
    gate_feasibility_summary: Path
    gate_resolved_training_config: Path
    gate_training_metrics: Path
    gate_development_summary: Path
    gate_checkpoints_dir: Path
    gate_candidate_bundle_dir: Path
    gate_candidate_bundle_manifest: Path
    gate_validation_summary: Path
    gate_agreement_details: Path
    gate_bundle_dir: Path
    gate_bundle_manifest: Path
    audit_log: Path
    candidate_sidecar: Path
    raw_attempt_summary: Path
    reference_calibration: Path
    positive_details: Path
    reference_negative_details: Path
    model_negative_details: Path
    reference_report: Path
    model_negative_report: Path
    pass_report_posthoc: Path
    detector_integrity: Path
    no_quality_gate_integrity: Path
    artifact_integrity: Path
    run_log: Path


def _component_name_max(run_root: str | Path) -> int:
    """Return a conservative byte limit for one component under run_root."""

    probe = Path(run_root)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        detected = int(os.pathconf(probe, "PC_NAME_MAX"))
    except (OSError, TypeError, ValueError):
        return 255
    return min(255, detected) if detected > 0 else 255


def build_run_paths(run_root: str | Path, run_id: str) -> RunPaths:
    try:
        encoded_run_id = os.fsencode(run_id) if isinstance(run_id, str) else b""
    except (TypeError, UnicodeError) as exc:
        raise ValueError("run_id must be a safe, non-empty path component") from exc
    if (
        not isinstance(run_id, str)
        or not run_id
        or run_id in {".", ".."}
        or len(encoded_run_id) > _component_name_max(run_root)
        or "\x00" in run_id
        or any(
            ord(character) <= 0x1F or 0x7F <= ord(character) <= 0x9F
            for character in run_id
        )
        or "/" in run_id
        or "\\" in run_id
        or Path(run_id).name != run_id
    ):
        raise ValueError("run_id must be a safe, non-empty path component")
    run_dir = Path(run_root) / run_id
    return RunPaths(
        run_dir=run_dir,
        resolved_config=run_dir / "config" / "resolved_config.json",
        method_preset=run_dir / "config" / "method_preset.json",
        final_code_input=run_dir / "inputs" / "final_code.jsonl",
        gate_data_manifest=run_dir / "gate-data" / "manifest.json",
        gate_window_groups=run_dir / "gate-data" / "window_groups.jsonl",
        gate_candidate_attempts=run_dir / "gate-data" / "candidate_attempts.jsonl",
        gate_labels=run_dir / "gate-data" / "labels.jsonl",
        gate_split_manifest=run_dir / "gate-data" / "split_manifest.json",
        gate_training_key_bank_manifest=(
            run_dir / "gate-data" / "training_key_bank_manifest.json"
        ),
        gate_feasibility_summary=run_dir / "gate-data" / "feasibility_summary.json",
        gate_resolved_training_config=(
            run_dir / "gate-train" / "resolved_training_config.json"
        ),
        gate_training_metrics=run_dir / "gate-train" / "training_metrics.jsonl",
        gate_development_summary=run_dir / "gate-train" / "development_summary.json",
        gate_checkpoints_dir=run_dir / "gate-train" / "checkpoints",
        gate_candidate_bundle_dir=run_dir / "gate-train" / "candidate_bundle",
        gate_candidate_bundle_manifest=(
            run_dir / "gate-train" / "candidate_bundle_manifest.json"
        ),
        gate_validation_summary=run_dir / "gate-validate" / "validation_summary.json",
        gate_agreement_details=run_dir / "gate-validate" / "agreement_details.jsonl",
        gate_bundle_dir=run_dir / "gate-validate" / "bundle",
        gate_bundle_manifest=(
            run_dir / "gate-validate" / "gate_bundle_manifest.json"
        ),
        audit_log=run_dir / "generation" / "audit.jsonl",
        candidate_sidecar=run_dir / "generation" / "candidate_sidecar.jsonl",
        raw_attempt_summary=run_dir / "generation" / "raw_attempt_summary.jsonl",
        reference_calibration=run_dir / "calibration" / "reference_calibration.json",
        positive_details=run_dir / "detection" / "positive_details.jsonl",
        reference_negative_details=run_dir / "detection" / "reference_negative_details.jsonl",
        model_negative_details=run_dir / "detection" / "model_negative_details.jsonl",
        reference_report=run_dir / "reports" / "reference_report.json",
        model_negative_report=run_dir / "reports" / "model_negative_report.json",
        pass_report_posthoc=run_dir / "reports" / "pass_report_posthoc.json",
        detector_integrity=run_dir / "audit" / "detector_input_integrity.json",
        no_quality_gate_integrity=run_dir / "audit" / "no_quality_gate_integrity.json",
        artifact_integrity=run_dir / "audit" / "artifact_integrity.json",
        run_log=run_dir / "logs" / "run.log",
    )


def write_jsonl_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            try:
                handle.write(
                    json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n"
                )
            except (TypeError, ValueError) as exc:
                raise ValueError("JSONL rows must be JSON-safe") from exc
    return str(output_path)
