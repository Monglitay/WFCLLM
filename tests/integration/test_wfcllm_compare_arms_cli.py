from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_compare_arms as compare_cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _detail(
    sample_id: str,
    *,
    score: float,
    decision: bool,
    insufficient: bool = False,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "score": score,
        "is_watermarked": decision,
        "insufficient_evidence": insufficient,
        "fpr_target": 0.05,
    }


def _execution(sample_id: str, passed: bool) -> dict[str, object]:
    return {
        "id": sample_id,
        "passed": passed,
        "reason": "passed" if passed else "failed_tests",
    }


def test_compare_arms_reports_paired_deltas_bootstrap_and_taxonomy(
    tmp_path: Path,
) -> None:
    ids = [f"HumanEval/{index}" for index in range(4)]
    current_positive = tmp_path / "current_positive.jsonl"
    v2_positive = tmp_path / "v2_positive.jsonl"
    current_negative = tmp_path / "current_negative.jsonl"
    v2_negative = tmp_path / "v2_negative.jsonl"
    current_execution = tmp_path / "current_execution.jsonl"
    v2_execution = tmp_path / "v2_execution.jsonl"
    output = tmp_path / "comparison.json"
    _write_jsonl(
        current_positive,
        [
            _detail(ids[0], score=0.8, decision=True),
            _detail(ids[1], score=0.7, decision=True),
            _detail(ids[2], score=0.2, decision=False, insufficient=True),
            _detail(ids[3], score=0.1, decision=False),
        ],
    )
    _write_jsonl(
        v2_positive,
        [
            _detail(ids[0], score=0.9, decision=True),
            _detail(ids[1], score=0.8, decision=True),
            _detail(ids[2], score=0.7, decision=True),
            _detail(ids[3], score=0.2, decision=False),
        ],
    )
    negative_ids = ["negative-0", "negative-1"]
    _write_jsonl(
        current_negative,
        [
            _detail(negative_ids[0], score=0.1, decision=False),
            _detail(negative_ids[1], score=0.2, decision=False),
        ],
    )
    _write_jsonl(
        v2_negative,
        [
            _detail(negative_ids[0], score=0.2, decision=False),
            _detail(negative_ids[1], score=0.3, decision=False),
        ],
    )
    _write_jsonl(
        current_execution,
        [
            _execution(ids[0], True),
            _execution(ids[1], False),
            _execution(ids[2], True),
            _execution(ids[3], False),
        ],
    )
    _write_jsonl(
        v2_execution,
        [
            _execution(ids[0], True),
            _execution(ids[1], True),
            _execution(ids[2], True),
            _execution(ids[3], False),
        ],
    )

    rc = compare_cli.main(
        [
            "--current-positive-details",
            str(current_positive),
            "--v2-positive-details",
            str(v2_positive),
            "--current-negative-details",
            str(current_negative),
            "--v2-negative-details",
            str(v2_negative),
            "--current-execution",
            str(current_execution),
            "--v2-execution",
            str(v2_execution),
            "--output",
            str(output),
            "--bootstrap-repetitions",
            "1000",
            "--seed",
            "20260713",
        ]
    )

    assert rc == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == "wfcllm-arm-comparison/v1"
    assert report["sample_count"] == 4
    assert report["negative_panel_count"] == 2
    assert report["current"]["passed"] == 2
    assert report["v2"]["passed"] == 3
    assert report["deltas"]["pass_count"] == 1
    assert report["deltas"]["pass_at_1"] == 0.25
    assert report["current"]["tpr"] == 0.5
    assert report["v2"]["tpr"] == 0.75
    assert report["deltas"]["tpr"] == 0.25
    assert report["paired"]["pass_mcnemar"] == {
        "current_only": 0,
        "v2_only": 1,
        "both": 2,
        "neither": 1,
    }
    assert report["paired"]["detection_mcnemar"] == {
        "current_only": 0,
        "v2_only": 1,
        "both": 2,
        "neither": 1,
    }
    assert report["paired"]["pass_delta_bootstrap_95"][0] <= 0.25
    assert report["paired"]["pass_delta_bootstrap_95"][1] >= 0.25
    assert report["failure_taxonomy"]["current"]["insufficient_evidence"] == 1
    assert report["failure_taxonomy"]["v2"]["sufficient_but_low_score"] == 1


def test_compare_arms_rejects_unpaired_positive_ids(tmp_path: Path, capsys) -> None:
    paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in (
            "current_positive",
            "v2_positive",
            "current_negative",
            "v2_negative",
            "current_execution",
            "v2_execution",
        )
    }
    _write_jsonl(paths["current_positive"], [_detail("a", score=0.1, decision=False)])
    _write_jsonl(paths["v2_positive"], [_detail("b", score=0.1, decision=False)])
    _write_jsonl(paths["current_negative"], [_detail("n", score=0.1, decision=False)])
    _write_jsonl(paths["v2_negative"], [_detail("n", score=0.1, decision=False)])
    _write_jsonl(paths["current_execution"], [_execution("a", False)])
    _write_jsonl(paths["v2_execution"], [_execution("b", False)])

    rc = compare_cli.main(
        [
            "--current-positive-details",
            str(paths["current_positive"]),
            "--v2-positive-details",
            str(paths["v2_positive"]),
            "--current-negative-details",
            str(paths["current_negative"]),
            "--v2-negative-details",
            str(paths["v2_negative"]),
            "--current-execution",
            str(paths["current_execution"]),
            "--v2-execution",
            str(paths["v2_execution"]),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )

    assert rc == 1
    assert "paired positive ids" in capsys.readouterr().err
