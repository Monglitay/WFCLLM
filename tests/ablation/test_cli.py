"""Tests for scripts/run_ablation.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_ablation.py"


def _load_script_module():
    spec = importlib.util.spec_from_file_location("_run_ablation_under_test", SCRIPT_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def script_module():
    return _load_script_module()


def test_help_does_not_crash(capsys, script_module):
    with pytest.raises(SystemExit) as exc:
        script_module.main(["--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert "yaml" in captured.out.lower() or "yaml" in captured.err.lower()


def test_main_invokes_runner_with_resume_flag(tmp_path, monkeypatch, script_module):
    base = tmp_path / "base.json"
    base.write_text('{"watermark": {}, "extract": {}}', encoding="utf-8")
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text(
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
  watermark.token_channel.mode: [dual-channel]
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
        encoding="utf-8",
    )

    captured_specs = []

    class _FakeRunner:
        def __init__(self, *a, **kw):  # noqa: ARG002
            pass
        def run(self, spec):
            captured_specs.append(spec)

    monkeypatch.setattr(script_module, "AblationRunner", _FakeRunner)

    rc = script_module.main([str(yaml_path)])
    assert rc == 0
    assert len(captured_specs) == 1
    assert captured_specs[0].name == "t"


def test_main_returns_nonzero_on_missing_yaml(capsys, script_module):
    rc = script_module.main(["does/not/exist.yaml"])
    assert rc != 0
    err = capsys.readouterr().err
    assert "not found" in err.lower() or "no such" in err.lower()
