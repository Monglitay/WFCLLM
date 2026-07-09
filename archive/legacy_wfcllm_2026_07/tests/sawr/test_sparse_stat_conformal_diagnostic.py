from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "analyze_sawr_sparse_stat_conformal.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_sawr_sparse_stat_conformal",
        SCRIPT_PATH,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(*scores: float, insufficient: bool = False) -> dict[str, object]:
    return {
        "id": "sample",
        "score": sum(scores) / len(scores) if scores else 0.0,
        "insufficient_evidence": insufficient,
        "context_summaries": [
            {
                "context_score": score,
                "context_raw": score / 10,
                "context_id": f"context-{index}",
            }
            for index, score in enumerate(scores)
        ],
    }


def test_compute_sparse_statistics_from_context_scores() -> None:
    module = _load_module()
    row = _row(0.5, 2.0, 1.0)

    assert module.compute_statistic(row, "current_mean", alpha_context=0.05) == pytest.approx(3.5 / 3)
    assert module.compute_statistic(row, "context_max", alpha_context=0.05) == pytest.approx(2.0)
    assert module.compute_statistic(row, "top2_mean", alpha_context=0.05) == pytest.approx(1.5)
    assert module.compute_statistic(row, "top2_sum", alpha_context=0.05) == pytest.approx(3.0)
    assert module.compute_statistic(row, "count_sig_contexts", alpha_context=0.05) == 1
    assert module.compute_statistic(row, "tippett_min_p", alpha_context=0.05) == pytest.approx(2.0)
    assert module.compute_statistic(row, "fisher", alpha_context=0.05) == pytest.approx(
        2 * module.LN10 * 3.5,
    )
    assert module.compute_statistic(row, "stouffer", alpha_context=0.05) > 0
    assert module.compute_statistic(row, "cauchy", alpha_context=0.05) > 0
    assert module.compute_statistic(row, "berk_jones", alpha_context=0.05) > 0


def test_conformal_detection_uses_upper_tail_and_gate() -> None:
    module = _load_module()
    calibration = [0.1, 0.2, 0.3, 0.4]

    assert module.conformal_p_value(0.45, calibration) == pytest.approx(1 / 5)
    assert module.conformal_p_value(0.25, calibration) == pytest.approx(3 / 5)

    detected = module.detect_with_conformal(
        score=0.45,
        calibration_scores=calibration,
        target_fpr=0.25,
        insufficient_evidence=False,
    )
    gated = module.detect_with_conformal(
        score=0.45,
        calibration_scores=calibration,
        target_fpr=0.25,
        insufficient_evidence=True,
    )

    assert detected.p_conformal == pytest.approx(0.2)
    assert detected.is_detected is True
    assert gated.is_detected is False
    assert gated.is_detected_without_gate is True
