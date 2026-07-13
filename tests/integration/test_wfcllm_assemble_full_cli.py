from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_assemble_full as assemble_cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _row(sample_id: str, code: str) -> dict[str, object]:
    return {
        "id": sample_id,
        "dataset": "humaneval",
        "prompt": f"prompt for {sample_id}",
        "final_code": code,
    }


def test_assemble_full_orders_and_hashes_complete_partition(tmp_path: Path) -> None:
    pilot = tmp_path / "pilot.jsonl"
    remaining = tmp_path / "remaining.jsonl"
    output = tmp_path / "full.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(pilot, [_row("HumanEval/2", "c2"), _row("HumanEval/0", "c0")])
    _write_jsonl(remaining, [_row("HumanEval/3", "c3"), _row("HumanEval/1", "c1")])

    rc = assemble_cli.main(
        [
            "--pilot",
            str(pilot),
            "--remaining",
            str(remaining),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
            "--expected-count",
            "4",
        ]
    )

    assert rc == 0
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == [
        "HumanEval/0",
        "HumanEval/1",
        "HumanEval/2",
        "HumanEval/3",
    ]
    report = json.loads(manifest.read_text())
    assert report["schema_version"] == "wfcllm-full-assembly/v1"
    assert report["row_count"] == 4
    assert report["pilot_count"] == 2
    assert report["remaining_count"] == 2
    assert report["output_sha256"]


def test_assemble_full_rejects_duplicate_ids(tmp_path: Path, capsys) -> None:
    pilot = tmp_path / "pilot.jsonl"
    remaining = tmp_path / "remaining.jsonl"
    _write_jsonl(pilot, [_row("HumanEval/0", "a")])
    _write_jsonl(remaining, [_row("HumanEval/0", "b"), _row("HumanEval/1", "c")])

    rc = assemble_cli.main(
        [
            "--pilot",
            str(pilot),
            "--remaining",
            str(remaining),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--expected-count",
            "2",
        ]
    )

    assert rc == 1
    assert "duplicate id" in capsys.readouterr().err


def test_assemble_full_rejects_missing_ids_and_extra_fields(
    tmp_path: Path,
    capsys,
) -> None:
    pilot = tmp_path / "pilot.jsonl"
    remaining = tmp_path / "remaining.jsonl"
    invalid = _row("HumanEval/2", "c")
    invalid["retry_score"] = 99
    _write_jsonl(pilot, [_row("HumanEval/0", "a")])
    _write_jsonl(remaining, [invalid])

    rc = assemble_cli.main(
        [
            "--pilot",
            str(pilot),
            "--remaining",
            str(remaining),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
            "--expected-count",
            "3",
        ]
    )

    assert rc == 1
    assert "exactly the final-code-only fields" in capsys.readouterr().err
