#!/usr/bin/env python3
"""Run calibration, extraction, and report with one shared detector runtime."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.cli.runners import run_calibrate, run_detect, run_report
from wfcllm.orchestration.state import DEFAULT_STATE_FILE, RunStateManager


def _summarize_window_details(rows: list[dict]) -> dict:
    decisions = Counter(str(row["decision"]) for row in rows)
    return {
        "sample_count": len(rows),
        "sample_count_with_at_least_two_reliable_windows": sum(
            int(row["reliable_window_count"]) >= 2 for row in rows
        ),
        "total_window_count": sum(len(row["windows"]) for row in rows),
        "total_reliable_window_count": sum(
            int(row["reliable_window_count"]) for row in rows
        ),
        "total_hit_count": sum(int(row["hit_count"]) for row in rows),
        "total_miss_count": sum(int(row["miss_count"]) for row in rows),
        "total_abstain_count": sum(int(row["abstain_count"]) for row in rows),
        "decision_counts": dict(sorted(decisions.items())),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = resolve_method_config(load_config(args.config))
        setattr(args, "_config_cache", config)
        state_path = Path(args.state_file) if args.state_file else DEFAULT_STATE_FILE
        state = RunStateManager(state_path)
        if run_calibrate(args, state) != 0 or run_detect(args, state) != 0:
            return 1
        from wfcllm.detection.gated_pipeline import load_gated_calibration_artifact

        run_dir = Path(args.run_dir)
        calibration_path = Path(args.calibration)
        positive_details_path = run_dir / "detection" / "positive_details.jsonl"
        negative_details_path = run_dir / "detection" / "negative_details.jsonl"
        pipeline = getattr(args, "_gated_detection_pipeline")
        artifact = load_gated_calibration_artifact(calibration_path)
        negative_results = pipeline.detect_jsonl(
            args.negative_input,
            artifact=artifact,
            output_path=negative_details_path,
        )
        positive_rows = _read_jsonl(positive_details_path)
        negative_rows = [asdict(result) for result in negative_results]
        diagnostics = {
            "schema_version": "wfcllm-window-diagnostics/v1",
            "gate_validation_skipped": True,
            "minimum_reliable_windows": artifact.minimum_reliable_windows,
            "target_fpr": artifact.target_fpr,
            "positive": _summarize_window_details(positive_rows),
            "negative": _summarize_window_details(negative_rows),
            "positive_details_sha256": _sha256(positive_details_path),
            "negative_details_sha256": _sha256(negative_details_path),
            "calibration_sha256": _sha256(calibration_path),
        }
        diagnostics_path = run_dir / "reports" / "window_diagnostics.json"
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_path.write_text(
            json.dumps(diagnostics, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if run_report(args, state) != 0:
            return 1
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"wfcllm_fast_postprocess: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    raise SystemExit(main())
