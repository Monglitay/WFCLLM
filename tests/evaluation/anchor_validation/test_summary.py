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


def _selection(
    context_id: str,
    method: str,
    budget: int,
    hit: bool,
    fallback: bool,
    key_id: str = "key-00",
) -> SelectionSimulationRow:
    return SelectionSimulationRow(
        context_id,
        method,
        key_id,
        0.5,
        budget,
        "c0",
        0,
        hit,
        fallback,
        1.0 if hit else -1.0,
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
    assert evidence["oracle_gap"]["slot_context"]["gain_ratio"] == pytest.approx(0.533333, rel=1e-5)
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


def test_summary_reports_layered_gates_and_failed_reasons():
    metrics = [
        _metric("ctx1", "vanilla", 0.40),
        _metric("ctx1", "random", 0.41),
        _metric("ctx1", "role_aware_slot_context", 0.44),
        _metric("ctx1", "seqmark_oracle", 0.70),
        _metric("ctx1", "role_aware_slot_context", 0.44, key_id="key-00", gamma_deviation=0.40),
    ]
    selection = [
        _selection("ctx1", "vanilla", 4, hit=True, fallback=False),
        _selection("ctx1", "random", 4, hit=True, fallback=False),
        _selection("ctx1", "role_aware_slot_context", 4, hit=False, fallback=True),
    ]
    pool_quality = {
        "context_count": 15,
        "candidates_per_context": {"median": 6},
        "node_type_distribution": {"return_statement": 15},
        "parse_valid_rate": 1.0,
    }

    summary = build_anchor_validation_summary(
        metrics,
        selection,
        context_count=15,
        methods=("vanilla", "random", "role_aware_slot_context", "seqmark_oracle"),
        pool_quality=pool_quality,
    )

    gates = summary["go_no_go"]["gates"]
    assert summary["go_no_go"]["first_stage_passed"] is False
    assert gates["data_quality_gate"]["passed"] is False
    assert "context_count 15 < 40" in gates["data_quality_gate"]["failed_reasons"]
    assert gates["balance_gate"]["passed"] is False
    assert "max_delta_gamma 0.4000 > 0.35" in gates["balance_gate"]["failed_reasons"]


def test_summary_passes_layered_gates_for_role_aware_signal():
    metrics = []
    selection = []
    for index in range(40):
        context_id = f"ctx{index}"
        role_entropy = 0.56 if index < 30 else 0.48
        random_entropy = 0.45
        vanilla_entropy = 0.43
        oracle_entropy = 0.75
        metrics.extend(
            [
                _metric(context_id, "vanilla", vanilla_entropy),
                _metric(context_id, "random", random_entropy),
                _metric(context_id, "role_aware_slot_context", role_entropy),
                _metric(context_id, "seqmark_oracle", oracle_entropy),
                _metric(context_id, "role_aware_slot_context", role_entropy, key_id="key-00", gamma_deviation=0.10),
            ]
        )
        selection.extend(
            [
                _selection(context_id, "vanilla", 4, hit=False, fallback=True),
                _selection(context_id, "random", 4, hit=False, fallback=True),
                _selection(context_id, "role_aware_slot_context", 4, hit=True, fallback=False),
            ]
        )
    pool_quality = {
        "context_count": 40,
        "candidates_per_context": {"median": 8},
        "node_type_distribution": {
            "return_statement": 10,
            "expression_statement": 10,
            "if_statement": 10,
            "import_statement": 10,
        },
        "parse_valid_rate": 1.0,
    }

    summary = build_anchor_validation_summary(
        metrics,
        selection,
        context_count=40,
        methods=("vanilla", "random", "role_aware_slot_context", "seqmark_oracle"),
        pool_quality=pool_quality,
    )

    gates = summary["go_no_go"]["gates"]
    assert summary["go_no_go"]["first_stage_passed"] is True
    assert all(gate["passed"] for gate in gates.values())
    assert summary["go_no_go"]["evidence"]["oracle_gap"]["role_aware_slot_context"]["gain_ratio"] >= 0.15


def test_summary_reports_method_oracle_agreement_and_gamma_calibration():
    metrics = [
        _metric("ctx1", "vanilla", 0.20),
        _metric("ctx1", "role_aware_slot_context", 0.40),
        _metric("ctx1", "seqmark_oracle", 0.60),
        _metric("ctx1", "role_aware_slot_context", 0.40, key_id="key-00", gamma_deviation=0.20),
    ]

    summary = build_anchor_validation_summary(
        metrics,
        [],
        context_count=1,
        methods=("vanilla", "role_aware_slot_context", "seqmark_oracle"),
        empirical_gamma_rows=[
            {
                "context_id": "ctx1",
                "method": "role_aware_slot_context",
                "key_id": "key-00",
                "target_gamma": 0.5,
                "empirical_gamma": 0.7,
                "delta": 0.2,
            }
        ],
        method_oracle_agreement={
            "role_aware_slot_context": {
                "proxy": "top_signature_match",
                "agreement_rate": 0.25,
            }
        },
    )

    evidence = summary["go_no_go"]["evidence"]
    assert evidence["method_oracle_agreement"]["role_aware_slot_context"]["agreement_rate"] == pytest.approx(0.25)
    assert evidence["gamma_calibration"]["role_aware_slot_context"]["mean_delta"] == pytest.approx(0.2)
