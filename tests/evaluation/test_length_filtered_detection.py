from __future__ import annotations

import pytest

from wfcllm.evaluation.length_filtered_detection import (
    build_length_filtered_detection_report,
)


def _final(sample_id: str, code: str) -> dict[str, str]:
    return {
        "id": sample_id,
        "dataset": "humaneval",
        "prompt": f"prompt {sample_id}",
        "final_code": code,
    }


def _detail(
    sample_id: str,
    *,
    decision: str,
    windows: int,
    reliable: int,
    hit_rate: float,
    insufficient: bool = False,
    p_value: float = 0.5,
) -> dict[str, object]:
    return {
        "id": sample_id,
        "decision": decision,
        "windows": [{} for _ in range(windows)],
        "reliable_window_count": reliable,
        "hit_rate": hit_rate,
        "insufficient_evidence": insufficient,
        "p_value": p_value,
    }


def test_builds_predeclared_length_filters() -> None:
    report = build_length_filtered_detection_report(
        positive_final_code_rows=[
            _final("p0", "x " * 250),
            _final("p1", "short"),
        ],
        positive_detail_rows=[
            _detail("p0", decision="watermarked", windows=2, reliable=2, hit_rate=1.0, p_value=0.01),
            _detail(
                "p1",
                decision="insufficient_evidence",
                windows=0,
                reliable=0,
                hit_rate=0.0,
                insufficient=True,
            ),
        ],
        negative_final_code_rows=[_final("n0", "y " * 250)],
        negative_detail_rows=[
            _detail("n0", decision="not_watermarked", windows=2, reliable=2, hit_rate=0.0, p_value=0.8)
        ],
    )

    assert report["all_sample"]["TPR_all_sample"] == pytest.approx(0.5)
    char_400 = report["filters"]["char_len >= 400"]
    assert char_400["kept_count"] == 1
    assert char_400["dropped_count"] == 1
    assert char_400["TPR_filtered"] == pytest.approx(1.0)
    assert char_400["FPR_filtered"] == pytest.approx(0.0)
    assert char_400["AUROC_filtered"] == pytest.approx(1.0)
    assert char_400["insufficient_evidence_count_in_dropped_set"] == 1


def test_rejects_missing_detail_rows() -> None:
    with pytest.raises(ValueError, match="details missing id"):
        build_length_filtered_detection_report(
            positive_final_code_rows=[_final("p0", "code")],
            positive_detail_rows=[],
        )
