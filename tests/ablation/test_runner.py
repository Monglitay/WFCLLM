"""Tests for AblationRunner (spec §5.3)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wfcllm.ablation.runner import AblationRunner
from wfcllm.ablation.sweep import SweepSpec


def _write_base(tmp_path: Path) -> Path:
    p = tmp_path / "base.json"
    p.write_text(
        '{"watermark": {"token_channel": {"mode": "semantic-only"}}, '
        '"extract": {"fpr_threshold": 3.0}}',
        encoding="utf-8",
    )
    return p


def _build_spec(tmp_path: Path, axes_yaml_block: str) -> SweepSpec:
    base = _write_base(tmp_path)
    yaml_path = tmp_path / "sweep.yaml"
    yaml_path.write_text(
        f"""
name: t
base_config: {base.name}
phases: [watermark]
axes:
{axes_yaml_block}
output_dir: {tmp_path}/out
metrics: [z_score_mean]
""",
        encoding="utf-8",
    )
    return SweepSpec.from_yaml(yaml_path)


class _FakeOrchestrator:
    """Stand-in for PhaseOrchestrator. Records calls; never invokes real phases."""

    def __init__(self):
        self.calls = []

    def dispatch_phase(self, phase, args):
        self.calls.append((phase, dict(getattr(args, "_config_cache", {}))))
        return 0


def test_runner_iterates_combos_and_writes_jsonl(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [semantic-only, dual-channel]",
    )
    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)

    # Stub metric extractors so we don't depend on real artefacts:
    monkeypatch.setattr(
        "wfcllm.ablation.runner.get_metric",
        lambda name: (lambda artefacts: 1.23),
    )

    runner.run(spec)

    results_path = spec.output_dir / "results.jsonl"
    rows = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 2
    assert rows[0]["axes_values"]["watermark.token_channel.mode"] == "semantic-only"
    assert rows[1]["axes_values"]["watermark.token_channel.mode"] == "dual-channel"
    assert rows[0]["metrics"] == {"z_score_mean": 1.23}
    assert rows[0]["run_id"].startswith("0000_")
    assert rows[1]["run_id"].startswith("0001_")


def test_runner_persists_per_run_config(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [dual-channel]",
    )
    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.0)

    runner.run(spec)

    [combo] = list(spec.expand())
    cfg_path = spec.output_dir / "runs" / combo.run_id / "config.json"
    assert cfg_path.exists()
    payload = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert payload["watermark"]["token_channel"]["mode"] == "dual-channel"
    assert payload["extract"]["fpr_threshold"] == 3.0


def test_runner_dispatches_each_phase_in_spec(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [dual-channel]",
    )
    fake = _FakeOrchestrator()
    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: fake)
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.0)

    runner.run(spec)

    assert [c[0] for c in fake.calls] == ["watermark"]


def test_runner_continues_on_phase_failure_and_records_error(tmp_path, monkeypatch):
    spec = _build_spec(
        tmp_path,
        "  watermark.token_channel.mode: [semantic-only, dual-channel]",
    )

    class _FailingOrchestrator:
        def __init__(self):
            self.n = 0

        def dispatch_phase(self, phase, args):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("boom")
            return 0

    runner = AblationRunner(orchestrator_factory=lambda *_a, **_kw: _FailingOrchestrator())
    monkeypatch.setattr("wfcllm.ablation.runner.get_metric", lambda name: lambda a: 0.0)

    runner.run(spec)

    rows = [json.loads(l) for l in (spec.output_dir / "results.jsonl").read_text().splitlines() if l]
    assert len(rows) == 2
    assert "error" in rows[0]
    assert "boom" in rows[0]["error"]
    assert "error" not in rows[1]
