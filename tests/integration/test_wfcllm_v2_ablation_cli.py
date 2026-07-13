from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_v2_ablation as ablation_cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _ledger_row(sample_id: str, attempt: int) -> dict[str, object]:
    accepted = 3 if attempt in {4, 7} else 1
    closed = 0 if attempt == 7 else 1
    return {
        "artifact_type": "wfcllm_v2_retry_attempt",
        "schema_version": "wfcllm-v2-retry-ledger/v2",
        "id": sample_id,
        "audit_only": True,
        "detector_input_allowed": False,
        "attempt_index": attempt,
        "seed": 20260713 + attempt * 101,
        "selected": attempt == 3,
        "v1_accepted_hit_count": accepted,
        "v1_closed_without_hit_count": closed,
        "v1_fallback_count": 0,
        "v1_candidate_count": 10 + attempt,
        "final_code": f"def target():\n    return {sample_id!r}, {attempt}\n",
    }


def test_v2_ablation_selects_v1_evidence_key_from_exactly_twenty_attempts(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "v2_final.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    output_path = tmp_path / "ablation_final.jsonl"
    manifest_path = tmp_path / "manifest.json"
    base_rows = [
        {
            "id": sample_id,
            "dataset": "humaneval",
            "prompt": "def target():\n",
            "final_code": "def target():\n    return 0\n",
        }
        for sample_id in ("HumanEval/0", "HumanEval/1")
    ]
    ledger_rows = [
        _ledger_row(sample_id, attempt)
        for sample_id in ("HumanEval/0", "HumanEval/1")
        for attempt in range(20)
    ]
    _write_jsonl(base_path, base_rows)
    _write_jsonl(ledger_path, ledger_rows)

    rc = ablation_cli.main(
        [
            "--v2-final-code",
            str(base_path),
            "--retry-ledger",
            str(ledger_path),
            "--output",
            str(output_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert rc == 0
    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["id"] for row in output_rows] == ["HumanEval/0", "HumanEval/1"]
    assert all(set(row) == {"id", "dataset", "prompt", "final_code"} for row in output_rows)
    assert all("7" in row["final_code"] for row in output_rows)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == "wfcllm-v2-ablation-selection/v1"
    assert manifest["retry"] == 20
    assert manifest["sample_count"] == 2
    assert [row["attempt_index"] for row in manifest["selections"]] == [7, 7]
    assert manifest["selection_rule"] == "v1_evidence_retry_key"


def test_v2_ablation_rejects_incomplete_retry_ledger(tmp_path: Path, capsys) -> None:
    base_path = tmp_path / "v2_final.jsonl"
    ledger_path = tmp_path / "ledger.jsonl"
    _write_jsonl(
        base_path,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def target():\n",
                "final_code": "def target():\n    return 0\n",
            }
        ],
    )
    _write_jsonl(
        ledger_path,
        [_ledger_row("HumanEval/0", attempt) for attempt in range(19)],
    )

    rc = ablation_cli.main(
        [
            "--v2-final-code",
            str(base_path),
            "--retry-ledger",
            str(ledger_path),
            "--output",
            str(tmp_path / "out.jsonl"),
            "--manifest",
            str(tmp_path / "manifest.json"),
        ]
    )

    assert rc == 1
    assert "exactly attempt indices 0..19" in capsys.readouterr().err
