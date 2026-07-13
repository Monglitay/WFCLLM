from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_prepare_v2_splits as split_cli


def _write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _prereg(path: Path, *, overlap: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact_type": "wfcllm_v2_preregistration",
                "schema_version": "wfcllm-v2-preregistration/v1",
                "seed": 20260713,
                "pilot_task_selection": {"ids": ["task-0"]},
                "negative_split": {
                    "source": "unused.jsonl",
                    "pilot_development_ids": ["task-0"],
                    "calibration_ids": ["task-0" if overlap else "task-1"],
                    "heldout_ids": ["task-2"],
                },
            }
        ),
        encoding="utf-8",
    )


def test_prepare_v2_splits_writes_strict_rows_and_hash_manifest(
    tmp_path: Path,
) -> None:
    prereg = tmp_path / "prereg.json"
    source = tmp_path / "negative.jsonl"
    output = tmp_path / "splits"
    _prereg(prereg)
    _write_jsonl(
        source,
        [
            {
                "id": f"task-{index}",
                "dataset": "humaneval",
                "prompt": f"def f{index}():\n",
                "generated_code": f"    return {index}\n",
            }
            for index in range(3)
        ],
    )

    rc = split_cli.main(
        [
            "--preregistration",
            str(prereg),
            "--negative-source",
            str(source),
            "--output-dir",
            str(output),
        ]
    )

    assert rc == 0
    pilot = [json.loads(line) for line in (output / "pilot.jsonl").read_text().splitlines()]
    calibration = [
        json.loads(line)
        for line in (output / "calibration.jsonl").read_text().splitlines()
    ]
    heldout = [
        json.loads(line)
        for line in (output / "heldout.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in pilot] == ["task-0"]
    assert [row["id"] for row in calibration] == ["task-1"]
    assert [row["id"] for row in heldout] == ["task-2"]
    assert set(pilot[0]) == {"id", "dataset", "prompt", "final_code"}
    assert pilot[0]["final_code"] == "def f0():\n    return 0\n"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["schema_version"] == "wfcllm-v2-negative-splits/v1"
    assert manifest["seed"] == 20260713
    assert manifest["splits"]["pilot"]["count"] == 1
    assert len(manifest["splits"]["pilot"]["sha256"]) == 64
    assert manifest["source"]["sha256"]


def test_prepare_v2_splits_rejects_overlapping_preregistered_ids(
    tmp_path: Path,
    capsys,
) -> None:
    prereg = tmp_path / "prereg.json"
    source = tmp_path / "negative.jsonl"
    _prereg(prereg, overlap=True)
    _write_jsonl(
        source,
        [
            {
                "id": f"task-{index}",
                "dataset": "humaneval",
                "prompt": f"def f{index}():\n",
                "generated_code": f"    return {index}\n",
            }
            for index in range(3)
        ],
    )

    rc = split_cli.main(
        [
            "--preregistration",
            str(prereg),
            "--negative-source",
            str(source),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert rc == 1
    assert "overlap" in capsys.readouterr().err
