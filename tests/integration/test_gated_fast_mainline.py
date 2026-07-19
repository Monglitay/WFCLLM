from __future__ import annotations

import json
from pathlib import Path

from wfcllm.cli.runners import _unvalidated_artifact_markers
from wfcllm.gate.config import GateDataConfig
from wfcllm.gate.feasibility import FEASIBILITY_THRESHOLD_ITEMS
from wfcllm.gate.pipeline import GateDataPipelineConfig
from wfcllm.method.presets import (
    GATED_SEMANTIC_WINDOW_V1_NAME,
    load_method_preset,
)
from scripts.wfcllm_fast_postprocess import _summarize_window_details


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "wfcllm" / "gated_semantic_window_v1.json"
RUNNER_PATH = ROOT / "scripts" / "run_gated_single_gpu.sh"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_gated_mainline_uses_three_hundred_group_batched_r3_contract() -> None:
    config = _config()
    gate_data = config["gate_data"]

    assert gate_data["pilot_independent_group_min"] == 100
    assert gate_data["pilot_independent_group_max"] == 300
    assert gate_data["full_independent_group_min"] == 300
    assert gate_data["full_independent_group_max"] == 300
    assert gate_data["learning_curve_group_counts"] == ["full"]
    assert gate_data["rewrite_count"] == 3
    assert gate_data["rewrite_budgets"] == [1, 3]
    assert GateDataConfig().rewrite_count == 3
    assert GateDataConfig().rewrite_budgets == (1, 3)

    preset_data = load_method_preset(GATED_SEMANTIC_WINDOW_V1_NAME).gate_data
    for key in (
        "pilot_independent_group_min",
        "pilot_independent_group_max",
        "full_independent_group_min",
        "full_independent_group_max",
        "learning_curve_group_counts",
        "rewrite_count",
        "rewrite_budgets",
    ):
        assert preset_data[key] == gate_data[key]


def test_full_gate_data_no_longer_requires_a_separate_pilot() -> None:
    thresholds = tuple(FEASIBILITY_THRESHOLD_ITEMS)

    value = GateDataPipelineConfig(
        output_root=Path("data/fast-run"),
        scale="full",
        config_hash="0" * 64,
        parser_contract="python-statement-window/v1",
        rewriter_config_hash="1" * 64,
        semantic_encoder_hash="2" * 64,
        lsh_config_hash="3" * 64,
        feasibility_contract="gate-data-feasibility/v1",
        feasibility_thresholds=thresholds,
        pilot_feasibility_path=None,
    )

    assert value.pilot_feasibility_path is None


def test_gated_training_and_validation_use_fast_contract() -> None:
    config = _config()

    assert config["method"]["gate"]["require_validated"] is False
    assert config["method"]["gate"]["max_input_tokens"] == 256
    assert config["method"]["rewrite"]["max_attempts"] == 3
    assert config["gate_train"]["max_tokens"] == 256
    assert config["gate_train"]["max_epochs"] == 4
    assert config["gate_train"]["early_stopping_patience"] == 1
    assert config["gate_train"]["losses"] == [
        "close_bce",
        "suitable_bce",
        "dangerous_negative_fp",
    ]
    assert config["gate_validate"]["batch_sizes"] == [1]
    assert config["gate_validate"]["orders"] == ["original"]
    assert config["gate_validate"]["gpu_float_if_available"] is False
    assert config["gate_validate"]["independent_reloads"] == 1


def test_skipped_gate_validation_is_disclosed_without_diagnostic_markers() -> None:
    markers = _unvalidated_artifact_markers(_config())

    assert markers == {
        "gate_validation_skipped": True,
        "unvalidated_gate_candidate": True,
    }
    assert "diagnostic_only" not in markers
    assert "not_official_method" not in markers


def test_window_diagnostic_summary_counts_reliable_evidence() -> None:
    summary = _summarize_window_details(
        [
            {
                "decision": "watermarked",
                "hit_count": 2,
                "miss_count": 1,
                "abstain_count": 1,
                "reliable_window_count": 3,
                "windows": [{}, {}, {}, {}],
            },
            {
                "decision": "insufficient_evidence",
                "hit_count": 0,
                "miss_count": 0,
                "abstain_count": 1,
                "reliable_window_count": 0,
                "windows": [{}],
            },
        ]
    )

    assert summary == {
        "sample_count": 2,
        "sample_count_with_at_least_two_reliable_windows": 1,
        "total_window_count": 5,
        "total_reliable_window_count": 3,
        "total_hit_count": 2,
        "total_miss_count": 1,
        "total_abstain_count": 2,
        "decision_counts": {
            "insufficient_evidence": 1,
            "watermarked": 1,
        },
    }


def test_single_gpu_runner_executes_minimal_experimental_pipeline() -> None:
    config = _config()
    phases = config["runtime"]["default_phases"]
    runner = RUNNER_PATH.read_text(encoding="utf-8")

    assert phases == [
        "gate-data",
        "gate-train",
        "generate",
        "calibrate",
        "detect",
        "report",
    ]
    assert "PILOT_CONFIG" not in runner
    assert "PILOT_RUN" not in runner
    assert "PILOT_STATE" not in runner
    assert "--pilot-feasibility" not in runner
    assert "--phase audit" not in runner
    assert "--phase gate-validate" not in runner
    assert 'SAMPLE_LIMIT="${SAMPLE_LIMIT:-32}"' in runner
    assert 'CALIBRATION_LIMIT="${CALIBRATION_LIMIT:-32}"' in runner
    assert 'GATE_EPOCHS="${GATE_EPOCHS:-}"' in runner
    assert 'GATE_EARLY_STOPPING_PATIENCE="${GATE_EARLY_STOPPING_PATIENCE:-}"' in runner
    assert 'PYTHONPATH=.' in runner
    assert '--gate-epochs "${GATE_EPOCHS}"' in runner
    assert '--gate-early-stopping-patience "${GATE_EARLY_STOPPING_PATIENCE}"' in runner
    assert 'head -n "${CALIBRATION_LIMIT}"' in runner
    assert "scripts/wfcllm_fast_postprocess.py" in runner
    assert "--phase calibrate" not in runner
    assert "--phase detect" not in runner
    assert "--phase report" not in runner
