"""Offline wiring tests for the experimental gated fast workflow.

The fake backend in this module is deliberately test-only and always marks its
artifacts diagnostic.  It verifies phase order and artifact boundaries without
pretending that fake model output is a validated formal gate bundle.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from wfcllm.audit.detector_input_integrity import audit_detector_input_file
from wfcllm.cli.runners import (
    _gated_report_metrics_from_artifacts,
    run_audit,
    run_report,
)
from wfcllm.method.artifacts import RunPaths, build_run_paths
from wfcllm.method.presets import (
    EVIDENCE_RETRY_SEED7X3_NAME,
    GATED_SEMANTIC_WINDOW_V1_NAME,
    load_method_preset,
)
from wfcllm.orchestration.state import RunStateManager


GATED_PHASES = [
    "gate-data",
    "gate-train",
    "gate-validate",
    "generate",
    "calibrate",
    "detect",
    "report",
]
FAST_EXPERIMENT_PHASES = [
    "gate-data",
    "gate-train",
    "generate",
    "calibrate",
    "detect",
    "report",
]


@dataclass
class _TracingOpen:
    reads: dict[str, list[Path]] = field(default_factory=dict)

    def record(self, phase: str, path: Path) -> str:
        self.reads.setdefault(phase, []).append(path)
        return path.read_text(encoding="utf-8")

    def paths_for_phase(self, phase: str) -> list[Path]:
        return list(self.reads.get(phase, ()))


@dataclass(frozen=True)
class _WorkflowResult:
    phase_order: list[str]
    run_paths: RunPaths
    audit_summary: dict[str, str]
    diagnostic_only: bool


class _FakeGatedCliRunner:
    """Controlled diagnostic harness; never registered by production runners."""

    def __init__(self, tracing: _TracingOpen | None = None) -> None:
        self.tracing = tracing or _TracingOpen()

    def run_gated(
        self,
        root: Path,
        *,
        backend: str = "fake",
        bundle: Path | None = None,
        start_phase: str | None = None,
        fail_validation: bool = False,
        expected_bundle_hash: str | None = None,
        completed: set[str] | None = None,
    ) -> _WorkflowResult:
        assert backend == "fake"
        paths = build_run_paths(root, "fake-gated-run")
        paths.run_dir.mkdir(parents=True, exist_ok=True)
        completed = set(completed or ())
        phases = GATED_PHASES
        if start_phase is not None:
            phases = phases[phases.index(start_phase) :]
        phase_order: list[str] = []

        if bundle is not None:
            actual_hash = _tree_hash(bundle)
            if expected_bundle_hash is not None and actual_hash != expected_bundle_hash:
                raise ValueError("validated gate bundle hash mismatch")

        for phase in phases:
            if phase in completed:
                continue
            if phase == "gate-data":
                _write_json(
                    paths.gate_data_manifest,
                    _diagnostic_artifact("wfcllm-gate-data-manifest/v1"),
                )
            elif phase == "gate-train":
                _write_json(
                    paths.gate_candidate_bundle_manifest,
                    _diagnostic_artifact("wfcllm-gate-train-candidate/v1"),
                )
            elif phase == "gate-validate":
                if fail_validation:
                    raise ValueError("fake gate validation failed before generate")
                paths.gate_bundle_dir.mkdir(parents=True, exist_ok=True)
                (paths.gate_bundle_dir / "diagnostic.bin").write_bytes(b"fake")
                _write_json(
                    paths.gate_bundle_manifest,
                    {
                        **_diagnostic_artifact("wfcllm-test-gate-bundle/v1"),
                        "validated": False,
                    },
                )
            elif phase == "generate":
                if bundle is None and not paths.gate_bundle_manifest.exists():
                    raise ValueError("generate requires a validated bundle")
                paths.final_code_input.parent.mkdir(parents=True, exist_ok=True)
                paths.final_code_input.write_text(
                    json.dumps(
                        {
                            "id": "fake/0",
                            "dataset": "humaneval",
                            "prompt": "def f():\n",
                            "final_code": "def f():\n    return 1\n",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                _write_json(paths.audit_log, _diagnostic_artifact("generation-audit/v1"))
                _write_json(
                    paths.candidate_sidecar,
                    _diagnostic_artifact("generation-candidate-sidecar/v1"),
                )
            elif phase == "calibrate":
                _write_json(
                    paths.reference_calibration,
                    _diagnostic_artifact("gated-calibration/v1"),
                )
            elif phase == "detect":
                self.tracing.record("detect", paths.final_code_input)
                _write_json(
                    paths.positive_details,
                    _diagnostic_artifact("gated-detection-details/v1"),
                )
            elif phase == "report":
                _write_json(
                    paths.reference_report,
                    {
                        **_diagnostic_artifact("gated-report/v1"),
                        "method": GATED_SEMANTIC_WINDOW_V1_NAME,
                        "detector_mode": "wfcllm-gated-semantic-window/v1",
                    },
                )
            elif phase == "audit":
                summary = audit_detector_input_file(paths.final_code_input)
                _write_json(paths.detector_integrity, summary)
            phase_order.append(phase)

        integrity = audit_detector_input_file(paths.final_code_input)
        return _WorkflowResult(
            phase_order=phase_order,
            run_paths=paths,
            audit_summary={"detector_input_integrity": "pass" if integrity["ok"] else "fail"},
            diagnostic_only=True,
        )


def _diagnostic_artifact(schema: str) -> dict[str, object]:
    return {
        "schema_version": schema,
        "diagnostic_test_backend": True,
        "formal_eligible": False,
        "diagnostic_only": True,
        "not_official_method": True,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@pytest.fixture
def tracing_open() -> _TracingOpen:
    return _TracingOpen()


@pytest.fixture
def cli_runner(tracing_open: _TracingOpen) -> _FakeGatedCliRunner:
    return _FakeGatedCliRunner(tracing_open)


@pytest.fixture
def fake_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "external-diagnostic-bundle"
    bundle.mkdir()
    _write_json(bundle / "manifest.json", _diagnostic_artifact("wfcllm-test-gate-bundle/v1"))
    return bundle


def test_gated_workflow_runs_all_phases_with_fake_backends(
    tmp_path: Path, cli_runner: _FakeGatedCliRunner
) -> None:
    result = cli_runner.run_gated(tmp_path, backend="fake")
    assert result.phase_order == GATED_PHASES
    assert result.run_paths.final_code_input.exists()
    assert result.run_paths.gate_bundle_manifest.exists()
    assert result.audit_summary["detector_input_integrity"] == "pass"
    assert result.diagnostic_only is True
    assert json.loads(result.run_paths.gate_bundle_manifest.read_text())["validated"] is False


def test_detector_opens_no_generation_sidecars(
    tmp_path: Path, tracing_open: _TracingOpen, cli_runner: _FakeGatedCliRunner
) -> None:
    cli_runner.run_gated(tmp_path, backend="fake")
    detector_reads = tracing_open.paths_for_phase("detect")
    assert detector_reads
    assert all("generation/audit.jsonl" not in str(path) for path in detector_reads)
    assert all("generation/candidate_sidecar.jsonl" not in str(path) for path in detector_reads)
    assert all("gate-data" not in str(path) for path in detector_reads)


def test_external_validated_bundle_skips_gate_phases(
    tmp_path: Path, cli_runner: _FakeGatedCliRunner, fake_bundle: Path
) -> None:
    result = cli_runner.run_gated(tmp_path, bundle=fake_bundle, start_phase="generate")
    assert result.phase_order == ["generate", "calibrate", "detect", "report"]


def test_fake_workflow_resume_skips_completed_phases(
    tmp_path: Path, cli_runner: _FakeGatedCliRunner
) -> None:
    result = cli_runner.run_gated(tmp_path, completed={"gate-data", "gate-train"})
    assert result.phase_order == GATED_PHASES[2:]


def test_fake_bundle_hash_mismatch_fails_before_generate(
    tmp_path: Path, cli_runner: _FakeGatedCliRunner, fake_bundle: Path
) -> None:
    with pytest.raises(ValueError, match="hash mismatch"):
        cli_runner.run_gated(
            tmp_path,
            bundle=fake_bundle,
            start_phase="generate",
            expected_bundle_hash="0" * 64,
        )


def test_failed_validation_cannot_reach_generate(
    tmp_path: Path, cli_runner: _FakeGatedCliRunner
) -> None:
    with pytest.raises(ValueError, match="validation failed"):
        cli_runner.run_gated(tmp_path, fail_validation=True)
    assert not build_run_paths(tmp_path, "fake-gated-run").final_code_input.exists()


def test_old_preset_keeps_five_phase_contract() -> None:
    preset = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict()
    assert preset["method"]["name"] == EVIDENCE_RETRY_SEED7X3_NAME
    assert preset.get("runtime", {}).get("default_phases", [
        "generate", "calibrate", "detect", "report", "audit"
    ]) == ["generate", "calibrate", "detect", "report", "audit"]


def test_gated_config_captures_complete_offline_contract_without_secrets() -> None:
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    assert config["runtime"]["default_phases"] == FAST_EXPERIMENT_PHASES
    assert config["method"]["windowing"]["max_units"] == 3
    assert config["method"]["rewrite"]["candidate_zero"] == "original_window"
    assert config["method"]["rewrite"]["experiment_budgets"] == [1, 3]
    assert config["gate_data"]["training_key_count"] == 32
    assert config["gate_data"]["holdout_key_count"] == 8
    assert config["method"]["gate"]["max_input_tokens"] == 256
    assert config["gate_validate"]["formal_quantization"] == "torch-dynamic-qint8-linear"
    assert config["detector"]["target_fpr"] == 0.05
    assert (
        config["calibration"]["posthoc_pass_at_1_noninferiority_absolute_drop_max"]
        >= 0
    )

    rendered = json.dumps(config, sort_keys=True)
    assert "raw_training_key" not in rendered
    assert "deployment_key" not in rendered
    assert '"key_material"' not in rendered
    for value in _string_values(config):
        assert not value.startswith("/")


def _string_values(value: object):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _string_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _string_values(child)


def test_gated_and_proxy_report_state_are_separated(tmp_path: Path) -> None:
    gated_state = RunStateManager(tmp_path / "gated-state.json")
    gated = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    gated_args = argparse.Namespace(_config_cache=gated)
    assert run_report(gated_args, gated_state) == 0

    proxy_state = RunStateManager(tmp_path / "proxy-state.json")
    proxy = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict()
    proxy_args = argparse.Namespace(_config_cache=proxy)
    assert run_report(proxy_args, proxy_state) == 0

    assert gated_state.get("report", "detector_mode") == "wfcllm-gated-semantic-window/v1"
    assert gated_state.get("report", "method") == GATED_SEMANTIC_WINDOW_V1_NAME
    assert proxy_state.get("report", "detector_mode") != gated_state.get(
        "report", "detector_mode"
    )


def test_gated_report_records_required_metrics_and_posthoc_marker(tmp_path: Path) -> None:
    state = RunStateManager(tmp_path / "report-state.json")
    config = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).to_dict()
    args = argparse.Namespace(
        _config_cache=config,
        _gated_report_metrics={
            "gate_coverage": 0.75,
            "hit_count": 3,
            "miss_count": 1,
            "abstain_count": 2,
            "rewrite_cost": {"attempts": 4},
            "detection_curve": [{"threshold": 0.5, "tpr": 0.8}],
            "calibration": {"target_fpr": 0.05},
        },
        _posthoc_pass_report={
            "pass_at_1": 0.5,
            "posthoc_only": True,
            "not_used_for_generation": True,
            "not_used_for_retry": True,
            "not_used_for_selection": True,
            "not_used_for_calibration": True,
            "not_used_for_detection": True,
        },
    )
    assert run_report(args, state) == 0
    assert state.get("report", "gate_coverage") == 0.75
    assert state.get("report", "hit_count") == 3
    assert state.get("report", "posthoc_pass_report")["posthoc_only"] is True


def test_gated_report_counts_selected_rewrites_without_claiming_unknown_attempts(
    tmp_path: Path,
) -> None:
    detection = tmp_path / "detection"
    generation = tmp_path / "generation"
    detection.mkdir()
    generation.mkdir()
    (detection / "positive_details.jsonl").write_text(
        json.dumps({"hit_count": 3, "miss_count": 1, "abstain_count": 0}) + "\n",
        encoding="utf-8",
    )
    (generation / "audit.jsonl").write_text(
        "".join(
            json.dumps({"selected_candidate_index": value}) + "\n"
            for value in (0, 1, 2, 0)
        ),
        encoding="utf-8",
    )

    metrics = _gated_report_metrics_from_artifacts(tmp_path, {})

    assert metrics["rewrite_cost"] == {
        "attempts": None,
        "attempts_lower_bound": 3,
        "selected_rewrite_count": 2,
        "selected_candidate_index_counts": {"0": 2, "1": 1, "2": 1},
    }


def test_run_audit_dispatches_detector_and_gate_checks(tmp_path: Path) -> None:
    paths = build_run_paths(tmp_path, "audit-run")
    paths.final_code_input.parent.mkdir(parents=True)
    paths.final_code_input.write_text(
        '{"id":"x","dataset":"humaneval","prompt":"p","final_code":"return 1"}\n',
        encoding="utf-8",
    )
    _write_json(paths.gate_data_manifest, {
        "schema_version": "wfcllm-gate-data-manifest/v1",
        "artifact_type": "gate-data-jsonl",
        "diagnostic_test_backend": False,
        "formal_eligible": True,
    })
    state = RunStateManager(tmp_path / "audit-state.json")
    args = SimpleNamespace(run_dir=str(paths.run_dir), _config_cache={})
    assert run_audit(args, state) == 0
    assert state.get("audit", "detector_input_integrity") == "pass"
    assert state.get("audit", "gate_artifact_integrity") == "pass"
