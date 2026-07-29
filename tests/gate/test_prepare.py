from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.gate.prepare import (
    prepare_gated_experiment,
    prepare_gated_source_manifest,
)


def _catalog_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(3):
        rows.append(
            {
                "source_family": "main_generation",
                "source_id": f"main:{index}",
                "code": f"def main_{index}():\n    return {index}\n",
                "repository_id": None,
                "task_id": f"main-task-{index}",
                "function_id": None,
                "source_model_id": f"model-{index}",
                "license_id": None,
                "contract_or_hard_set": False,
                "prompt": "",
            }
        )
        rows.append(
            {
                "source_family": "oss_python",
                "source_id": f"oss:{index}",
                "code": f"def oss_{index}():\n    return {index}\n",
                "repository_id": f"repo-{index}",
                "task_id": None,
                "function_id": f"function-{index}",
                "source_model_id": None,
                "license_id": "MIT",
                "contract_or_hard_set": False,
                "prompt": "",
            }
        )
    rows.append(
        {
            "source_family": "parser_boundary",
            "source_id": "parser:0",
            "code": "if True:\n    x = 1\n",
            "repository_id": None,
            "task_id": "parser-task-0",
            "function_id": None,
            "source_model_id": None,
            "license_id": None,
            "contract_or_hard_set": True,
            "prompt": "",
        }
    )
    return rows


def test_prepare_writes_manifest_and_disjoint_private_banks(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(json.dumps(row) + "\n" for row in _catalog_rows()),
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

    assert value["source_count"] == 7
    assert value["source_model_ids"] == ["model-0", "model-1", "model-2"]
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


def test_prepare_rejects_incomplete_formal_catalog_before_outputs(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "incomplete.jsonl"
    catalog.write_text(
        json.dumps(_catalog_rows()[0]) + "\n",
        encoding="utf-8",
    )
    outputs = (
        tmp_path / "manifest.json",
        tmp_path / "training.json",
        tmp_path / "holdout.json",
        tmp_path / "deployment.key",
    )

    with pytest.raises(ValueError, match="at least three"):
        prepare_gated_experiment(
            source_catalog=catalog,
            source_manifest=outputs[0],
            training_key_bank=outputs[1],
            holdout_key_bank=outputs[2],
            deployment_key=outputs[3],
        )
    assert all(not path.exists() for path in outputs)


def test_ablation_source_preparation_does_not_create_or_replace_family_keys(
    tmp_path: Path,
) -> None:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        "".join(json.dumps(row) + "\n" for row in _catalog_rows()),
        encoding="utf-8",
    )
    manifest = tmp_path / "variant" / "source_manifest.json"
    family_keys = tmp_path / "family-private" / "training.json"
    family_keys.parent.mkdir()
    family_keys.write_text('["shared"]\n', encoding="utf-8")
    before = family_keys.read_bytes()

    value = prepare_gated_source_manifest(
        source_catalog=catalog,
        source_manifest=manifest,
    )

    assert value["source_count"] == 7
    assert manifest.is_file()
    assert family_keys.read_bytes() == before
    assert list((tmp_path / "variant").iterdir()) == [manifest]
