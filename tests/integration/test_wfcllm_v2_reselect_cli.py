from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import scripts.wfcllm_v2_reselect as reselect_cli
from wfcllm.generation.selection_v2 import V2_RETRY_LEDGER_SCHEMA_VERSION


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_reselect_builds_strict_final_code_with_standardized_bit_sum(
    tmp_path: Path,
) -> None:
    prompt = 'def target(x):\n    """Return x."""\n'
    base = tmp_path / "base.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    output = tmp_path / "repaired.jsonl"
    manifest = tmp_path / "manifest.json"
    _write_jsonl(
        base,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": prompt,
                "final_code": prompt + "    return x\n",
            }
        ],
    )
    rows = []
    for attempt_index in range(20):
        code = prompt + f"    value = x + {attempt_index}\n    return value\n"
        matched_bits = 22 if attempt_index == 1 else 11
        total_bits = 32 if attempt_index == 1 else 16
        rows.append(
            {
                "artifact_type": "wfcllm_v2_retry_attempt",
                "schema_version": V2_RETRY_LEDGER_SCHEMA_VERSION,
                "id": "HumanEval/0",
                "audit_only": True,
                "detector_input_allowed": False,
                "attempt_index": attempt_index,
                "seed": 20260713 + 101 * attempt_index,
                "selected": attempt_index == 0,
                "generation_score": matched_bits / total_bits,
                "unit_count": total_bits // 16,
                "total_signature_bits": total_bits,
                "matched_signature_bits": matched_bits,
                "quality": {"eligible": attempt_index in {0, 1}, "quality_tier": 2},
                "v1_fallback_count": 0,
                "final_code": code,
                "final_code_sha256": hashlib.sha256(code.encode()).hexdigest(),
            }
        )
    _write_jsonl(ledger, rows)

    rc = reselect_cli.main(
        [
            "--base-final-code",
            str(base),
            "--retry-ledger",
            str(ledger),
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ]
    )

    assert rc == 0
    selected = json.loads(output.read_text(encoding="utf-8"))
    assert set(selected) == {"id", "dataset", "prompt", "final_code"}
    assert "x + 1" in selected["final_code"]
    report = json.loads(manifest.read_text(encoding="utf-8"))
    assert report["schema_version"] == "wfcllm-v2-repair-selection/v1"
    assert report["aggregation"] == "standardized_bit_sum"
    assert report["retry"] == 20
    assert report["development_iteration"] is True
    assert report["selections"][0]["attempt_index"] == 1
    assert report["selections"][0]["generation_score"] == pytest.approx(
        (22 - 16) / (32 / 4) ** 0.5
    )
