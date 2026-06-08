from __future__ import annotations

import pytest

from wfcllm.evaluation.anchor_validation.metrics import (
    bootstrap_mean_ci,
    effective_region_count,
    normalized_region_entropy,
    summarize_signature_metrics,
    valid_hit_balance,
)


def test_normalized_region_entropy_is_zero_for_full_collapse():
    signatures = [(0, 0), (0, 0), (0, 0)]
    assert normalized_region_entropy(signatures, region_count=4) == 0.0


def test_normalized_region_entropy_is_one_for_even_two_region_split():
    signatures = [(0,), (1,)]
    assert normalized_region_entropy(signatures, region_count=2) == pytest.approx(1.0)


def test_effective_region_count_matches_exp_entropy():
    signatures = [(0,), (1,)]
    assert effective_region_count(signatures) == pytest.approx(2.0)


def test_valid_hit_balance_reports_gamma_deviation():
    signatures = [(0,), (1,), (1,), (1,)]
    balance = valid_hit_balance(signatures, valid_set=frozenset({(1,)}), gamma=0.5)

    assert balance.hit_rate == pytest.approx(0.75)
    assert balance.gamma_deviation == pytest.approx(0.25)


def test_summarize_signature_metrics_includes_collapse_ratio():
    row = summarize_signature_metrics(
        context_id="ctx",
        dataset="humaneval",
        task_id="HumanEval/0",
        method="vanilla",
        signatures=[(0,), (1,)],
        region_count=2,
        projection_key_id="proj-00",
        key_id=None,
        gamma=None,
        valid_set=None,
    )

    assert row.normalized_entropy == pytest.approx(1.0)
    assert row.collapse_ratio == pytest.approx(0.0)


def test_bootstrap_mean_ci_is_deterministic():
    lo, hi = bootstrap_mean_ci([0.1, 0.2, 0.3], iterations=100, seed=7)

    assert 0.1 <= lo <= hi <= 0.3
