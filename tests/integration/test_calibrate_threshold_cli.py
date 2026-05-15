"""Smoke test for scripts/calibrate_threshold.py: argparse layer wires args correctly."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch


def _load_calibrate_threshold_module():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "calibrate_threshold.py"
    spec = importlib.util.spec_from_file_location("calibrate_threshold_cli", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_calibrate_threshold_cli_passes_args_to_runner(tmp_path, monkeypatch):
    module = _load_calibrate_threshold_module()
    input_path = tmp_path / "corpus.jsonl"
    input_path.write_text('{"generated_code": "x"}\n', encoding="utf-8")
    output_path = tmp_path / "thr.json"

    captured = {}

    def fake_runner(**kwargs):
        captured.update(kwargs)
        out = Path(kwargs["output"])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"fpr": kwargs["fpr"]}), encoding="utf-8")
        return out

    monkeypatch.setattr(module, "calibrate_threshold_from_corpus", fake_runner)
    monkeypatch.setattr(
        sys, "argv",
        [
            "calibrate_threshold.py",
            "--input", str(input_path),
            "--output", str(output_path),
            "--secret-key", "k",
            "--model", "data/models/codet5-base",
            "--device", "cpu",
            "--fpr", "0.02",
            "--gamma", "0.5",
            "--embed-dim", "128",
            "--lsh-d", "3",
        ],
    )
    rc = module.main()
    assert rc == 0
    assert captured["secret_key"] == "k"
    assert captured["fpr"] == 0.02
    assert captured["device"] == "cpu"
    assert Path(captured["output"]) == output_path
    assert output_path.exists()
