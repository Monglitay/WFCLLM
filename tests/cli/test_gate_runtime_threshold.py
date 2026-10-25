from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wfcllm.cli.config_resolver import load_config, resolve_method_config
from wfcllm.cli.runners import _current_run_gate_suitable_accept_threshold

ROOT = Path(__file__).resolve().parents[2]


def _args(tmp_path: Path) -> SimpleNamespace:
    run_dir = tmp_path / "run"
    (run_dir / "gate-train").mkdir(parents=True)
    config = resolve_method_config(
        load_config(ROOT / "configs/wfcllm/experiments/python_humaneval_full.json")
    )
    config["method"]["windowing"].pop("suitable_accept_threshold_cap", None)
    return SimpleNamespace(run_dir=run_dir, _config_cache=config)


def test_runtime_suitable_threshold_comes_from_current_gate_train_manifest(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    manifest_path = args.run_dir / "gate-train" / "candidate_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-gate-train-candidate/v1",
                "training_result": {
                    "suitable_accept_threshold": 0.36,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _current_run_gate_suitable_accept_threshold(args, 0.5) == pytest.approx(
        0.36
    )


def test_runtime_suitable_threshold_can_be_capped_by_windowing_config(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args._config_cache["method"]["windowing"]["suitable_accept_threshold_cap"] = 0.2
    manifest_path = args.run_dir / "gate-train" / "candidate_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-gate-train-candidate/v1",
                "training_result": {
                    "suitable_accept_threshold": 0.36,
                },
            }
        ),
        encoding="utf-8",
    )

    assert _current_run_gate_suitable_accept_threshold(args, 0.5) == pytest.approx(
        0.2
    )


def test_runtime_suitable_threshold_is_required_from_current_gate_train_manifest(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    manifest_path = args.run_dir / "gate-train" / "candidate_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-gate-train-candidate/v1",
                "training_result": {},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="suitable_accept_threshold is missing"):
        _current_run_gate_suitable_accept_threshold(args, 0.5)


@pytest.mark.parametrize("value", [-0.1, 1.1, "0.36", True])
def test_runtime_suitable_threshold_rejects_invalid_manifest_value(
    tmp_path: Path,
    value: object,
) -> None:
    args = _args(tmp_path)
    manifest_path = args.run_dir / "gate-train" / "candidate_bundle_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-gate-train-candidate/v1",
                "training_result": {
                    "suitable_accept_threshold": value,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="suitable_accept_threshold"):
        _current_run_gate_suitable_accept_threshold(args, 0.5)
