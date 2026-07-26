from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_gated_wrapper_requires_separate_pilot_and_full_catalogs() -> None:
    script = (
        ROOT / "scripts" / "run_gated_single_gpu.sh"
    ).read_text(encoding="utf-8")

    assert 'PILOT_SOURCE_CATALOG:?set PILOT_SOURCE_CATALOG' in script
    assert 'FULL_SOURCE_CATALOG:?set FULL_SOURCE_CATALOG' in script
    assert '--gate-source-catalog "${PILOT_SOURCE_CATALOG}"' in script
    assert '--gate-source-catalog "${FULL_SOURCE_CATALOG}"' in script
    assert 'CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda/envs}"' in script
    assert '--phase gate-validate' not in script
    assert 'SAMPLE_LIMIT="${SAMPLE_LIMIT:-164}"' in script
    assert "wfcllm_fast_postprocess.py" not in script


def test_gated_wrapper_trains_encoder_before_gate_data() -> None:
    script = (
        ROOT / "scripts" / "run_gated_single_gpu.sh"
    ).read_text(encoding="utf-8")

    assert "--phase encoder" in script
    assert script.index("--phase encoder") < script.index("--phase gate-data")
    # Both the pilot and the full chain must train their encoder first.
    pilot_encoder = script.index('--phase encoder --config "${PILOT_CONFIG}"')
    full_encoder = script.index('--phase encoder --config "${FULL_CONFIG}"')
    pilot_gate_data = script.index('--phase gate-data --config "${PILOT_CONFIG}"')
    full_gate_data = script.index('--phase gate-data --config "${FULL_CONFIG}"')
    assert pilot_encoder < pilot_gate_data
    assert full_encoder < full_gate_data


def test_official_no_carrier_config_uses_semantic_lsh_rule_and_gated_phases() -> None:
    config = json.loads(
        (ROOT / "configs" / "wfcllm" / "gated_semantic_window_nocarrier_v1.json")
        .read_text(encoding="utf-8")
    )

    assert config["semantic_lsh"]["rule_name"] == "semantic_lsh"
    assert "require_validated" not in config["method"]["gate"]
    assert config["runtime"]["default_phases"] == [
        "encoder",
        "gate-data",
        "gate-train",
        "generate",
        "calibrate",
        "detect",
        "report",
    ]
