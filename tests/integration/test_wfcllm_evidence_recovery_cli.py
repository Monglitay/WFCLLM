from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.wfcllm_evidence_recovery as recovery_cli
from wfcllm.detection.proxy_windows import extract_structure_contexts


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_evidence_recovery_compares_audit_hashes_with_final_code_windows(
    tmp_path: Path,
) -> None:
    code = "def target():\n    value = 1\n    return value\n"
    contexts = extract_structure_contexts(
        code,
        prompt=None,
        max_group_statements=2,
    )
    recovered_hash = _hash(contexts[0].proxy_windows[0].normalized_text)
    final_path = tmp_path / "final.jsonl"
    sidecar_path = tmp_path / "sidecar.jsonl"
    output_path = tmp_path / "recovery.json"
    _write_jsonl(
        final_path,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def target():\n",
                "final_code": code,
            }
        ],
    )
    _write_jsonl(
        sidecar_path,
        [
            {
                "id": "HumanEval/0",
                "event": "accepted_generation_time_window",
                "normalized_text_hash": recovered_hash,
                "audit_only": True,
                "detector_input_allowed": False,
            },
            {
                "id": "HumanEval/0",
                "event": "accepted_generation_time_window",
                "normalized_text_hash": recovered_hash,
                "audit_only": True,
                "detector_input_allowed": False,
            },
            {
                "id": "HumanEval/0",
                "event": "accepted_generation_time_window",
                "normalized_text_hash": "f" * 64,
                "audit_only": True,
                "detector_input_allowed": False,
            },
            {
                "id": "HumanEval/0",
                "event": "simple_candidate_observed",
                "normalized_text_hash": "0" * 64,
                "audit_only": True,
                "detector_input_allowed": False,
            },
        ],
    )

    rc = recovery_cli.main(
        [
            "--final-code",
            str(final_path),
            "--candidate-sidecar",
            str(sidecar_path),
            "--output",
            str(output_path),
            "--max-group-statements",
            "2",
        ]
    )

    assert rc == 0
    report = json.loads(output_path.read_text())
    assert report["schema_version"] == "wfcllm-evidence-recovery/v1"
    assert report["sample_count"] == 1
    assert report["accepted_count"] == 3
    assert report["accepted_unique_count"] == 2
    assert report["recovered_accepted_count"] == 2
    assert report["recovered_accepted_unique_count"] == 1
    assert report["accepted_to_recovered_ratio"] == 2 / 3
    assert report["accepted_unique_to_recovered_ratio"] == 0.5
    assert report["per_sample"][0]["final_proxy_window_count"] >= 1


def test_evidence_recovery_rejects_detector_enabled_sidecar(tmp_path: Path) -> None:
    final_path = tmp_path / "final.jsonl"
    sidecar_path = tmp_path / "sidecar.jsonl"
    _write_jsonl(final_path, [])
    _write_jsonl(
        sidecar_path,
        [
            {
                "id": "HumanEval/0",
                "event": "accepted_generation_time_window",
                "normalized_text_hash": "f" * 64,
                "audit_only": False,
                "detector_input_allowed": True,
            }
        ],
    )

    rc = recovery_cli.main(
        [
            "--final-code",
            str(final_path),
            "--candidate-sidecar",
            str(sidecar_path),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 1
