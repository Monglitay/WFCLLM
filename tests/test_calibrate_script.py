"""Smoke tests for scripts/calibrate.py adaptive CLI entry points."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_entropy_profile_subcommand_writes_json(tmp_path):
    log_path = tmp_path / "watermark.log"
    log_path.write_text(
        "\n".join(
            [
                "wfcllm.watermark.generator DEBUG [simple block] entropy=0.1200",
                "wfcllm.watermark.generator DEBUG [simple block] entropy=0.2400",
                "wfcllm.watermark.generator DEBUG [simple block] entropy=0.3600",
                "wfcllm.watermark.generator DEBUG [simple block] entropy=0.4800",
                "wfcllm.watermark.generator DEBUG [simple block] entropy=0.6000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profile.json"

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate.py",
            "build-entropy-profile",
            "--input-log",
            str(log_path),
            "--output",
            str(output_path),
            "--language",
            "python",
            "--model-family",
            "demo-model",
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert completed.returncode == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["language"] == "python"
    assert payload["model_family"] == "demo-model"
    assert payload["quantiles_units"] == {
        "p10": 1200,
        "p50": 3600,
        "p75": 4800,
        "p90": 6000,
        "p95": 6000,
    }


def test_calibrate_threshold_subcommand_help_succeeds():
    completed = subprocess.run(
        [sys.executable, "scripts/calibrate.py", "calibrate-threshold", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )

    assert completed.returncode == 0
    assert "--input" in completed.stdout
    assert "--output" in completed.stdout
