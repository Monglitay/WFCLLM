from __future__ import annotations

import pytest

from wfcllm.evaluation.anchor_validation.schema import RegionMetricRow, SelectionSimulationRow
from wfcllm.evaluation.anchor_validation.summary import build_anchor_validation_summary


def _metric(
    context_id: str,
    method: str,
    entropy: float,
    key_id: str | None = None,
    gamma_deviation: float | None = None,
    projection_key_id: str = "proj-00",
):
    return RegionMetricRow(
        context_id=context_id,
        dataset="humaneval",
        task_id=context_id,
        method=method,
        projection_key_id=projection_key_id,
        key_id=key_id,
        gamma=0.5 if key_id else None,
        candidate_count=16,
        normalized_entropy=entropy,
        collapse_ratio=1.0 - entropy,
        effective_region_count=2.0,
        hamming_diversity=0.5,
        valid_hit_rate=0.5 if key_id else None,
        gamma_deviation=gamma_deviation,
    )


def test_summary_computes_required_anchor_gate_fields():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "random", 0.23),
        _metric("ctx1", "slot_context", 0.36),
        _metric("ctx1", "seqmark_oracle", 0.50),
        _metric("ctx1", "slot_context", 0.36, key_id="key-00", gamma_deviation=0.02),
        _metric("ctx1", "slot_context", 0.36, key_id="key-01", gamma_deviation=0.03),
    ]
    selection = [
        SelectionSimulationRow("ctx1", "vanilla", "key-00", 0.5, 4, "c0", 0, False, True, -1.0),
        SelectionSimulationRow("ctx1", "slot_context", "key-00", 0.5, 4, "c1", 1, True, False, 1.0),
        SelectionSimulationRow("ctx1", "slot_context", "key-00", 0.5, 8, "c1", 1, True, False, 1.0),
    ]

    summary = build_anchor_validation_summary(
        metrics,
        selection,
        context_count=1,
        methods=("vanilla", "random", "slot_context", "seqmark_oracle"),
    )

    evidence = summary["go_no_go"]["evidence"]
    assert evidence["paired_entropy_delta"]["slot_context_vs_vanilla"]["mean"] == pytest.approx(0.16)
    assert evidence["random_anchor_gap"]["slot_context_minus_random"]["mean"] == pytest.approx(0.13)
    assert evidence["seqmark_oracle_gain_ratio"]["slot_context"] == pytest.approx(0.533333, rel=1e-5)
    assert evidence["valid_hit_balance"]["slot_context"]["max_delta_gamma"] == pytest.approx(0.03)
    assert evidence["retry"]["slot_context"]["budget_4_hit_acquisition"] == pytest.approx(1.0)


def test_summary_averages_geometry_across_projection_keys():
    row_a = _metric("ctx1", "vanilla", 0.20, projection_key_id="proj-00")
    row_b = _metric("ctx1", "vanilla", 0.40, projection_key_id="proj-01")
    slot = _metric("ctx1", "slot_context", 0.50)

    summary = build_anchor_validation_summary(
        [row_a, row_b, slot],
        [],
        context_count=1,
        methods=("vanilla", "slot_context"),
    )

    assert summary["summary"]["mean_entropy_by_method"]["vanilla"] == pytest.approx(0.30)
