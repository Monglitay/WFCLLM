from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.gate.prepare import prepare_gated_experiment


def test_prepare_writes_manifest_and_disjoint_private_banks(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "source_family": "oss_python",
                "source_id": "repo:module:f",
                "code": "x = 1\n",
                "repository_id": "repo",
                "task_id": "task",
                "function_id": "f",
                "source_model_id": None,
                "license_id": "MIT",
                "contract_or_hard_set": False,
                "prompt": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    training = tmp_path / "private" / "training.json"
    holdout = tmp_path / "private" / "holdout.json"
    deployment = tmp_path / "private" / "deployment.key"

    value = prepare_gated_experiment(
        source_catalog=catalog,
        source_manifest=manifest,
        training_key_bank=training,
        holdout_key_bank=holdout,
        deployment_key=deployment,
    )

    assert value["source_count"] == 1
    training_values = json.loads(training.read_text(encoding="utf-8"))
    holdout_values = json.loads(holdout.read_text(encoding="utf-8"))
    assert len(training_values) == 32
    assert len(holdout_values) == 8
    assert set(training_values).isdisjoint(holdout_values)
    assert training.stat().st_mode & 0o777 == 0o600
    assert holdout.stat().st_mode & 0o777 == 0o600
    assert len(deployment.read_text(encoding="utf-8").strip()) == 64
    assert deployment.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="overwrite"):
        prepare_gated_experiment(
            source_catalog=catalog,
            source_manifest=manifest,
            training_key_bank=training,
            holdout_key_bank=holdout,
            deployment_key=deployment,
        )
