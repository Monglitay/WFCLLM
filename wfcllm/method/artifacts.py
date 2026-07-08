from __future__ import annotations

import json
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


def build_run_paths(run_root: str | Path, run_id: str) -> RunPaths:
    run_dir = Path(run_root) / run_id
    return RunPaths(
        run_dir=run_dir,
        resolved_config=run_dir / "config" / "resolved_config.json",
        method_preset=run_dir / "config" / "method_preset.json",
        final_code_input=run_dir / "inputs" / "final_code.jsonl",
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
