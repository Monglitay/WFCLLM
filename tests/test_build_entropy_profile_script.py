"""Smoke tests for scripts/build_entropy_profile.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def test_build_entropy_profile_script_writes_json(tmp_path):
    log_path = tmp_path / "watermark.log"
    log_path.write_text(
        "\n".join(
            [
                "wfcllm.watermark.generator DEBUG entropy=0.1200",
                "wfcllm.watermark.generator DEBUG entropy=0.2400",
                "wfcllm.watermark.generator DEBUG entropy=0.3600",
                "wfcllm.watermark.generator DEBUG entropy=0.4800",
                "wfcllm.watermark.generator DEBUG entropy=0.6000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "profile.json"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/build_entropy_profile.py",
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
    assert payload["quantiles_units"]["p10"] == 1200
    assert payload["quantiles_units"]["p95"] == 6000


def test_build_entropy_profile_script_help_succeeds():
    completed = subprocess.run(
        [sys.executable, "scripts/build_entropy_profile.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )
    assert completed.returncode == 0
    assert "--input-log" in completed.stdout
    assert "--language" in completed.stdout
