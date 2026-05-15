"""Tests for ablation metric extractors (spec §5.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from wfcllm.ablation.metrics import (
    METRIC_REGISTRY,
    get_metric,
    register_metric,
)

DATA_DIR = Path(__file__).parent / "data"


def _artefacts() -> dict[str, Path]:
    return {
        "extract_summary": DATA_DIR / "example_summary.json",
        "evaluation_summary": DATA_DIR / "example_evaluation_summary.json",
    }


def test_z_score_mean_extractor():
    metric = get_metric("z_score_mean")
    assert metric(_artefacts()) == pytest.approx(4.21)


def test_hit_rate_extractor():
    metric = get_metric("hit_rate")
    assert metric(_artefacts()) == pytest.approx(0.78)


def test_false_positive_rate_extractor_returns_none_when_field_absent():
    """Extract summary alone has no FPR — it lives in calibration outputs.

    The extractor must return None gracefully so JSONL row stays well-formed.
    """
    metric = get_metric("false_positive_rate")
    assert metric(_artefacts()) is None


def test_pass_at_1_extractor():
    metric = get_metric("pass_at_1")
    # Pulls the first mode's pass_at_1 from evaluation_summary
    assert metric(_artefacts()) == pytest.approx(0.42)


def test_extractor_returns_none_when_artefact_missing(tmp_path):
    metric = get_metric("z_score_mean")
    artefacts = {"extract_summary": tmp_path / "does_not_exist.json"}
    assert metric(artefacts) is None


def test_register_metric_rejects_duplicate():
    with pytest.raises(ValueError, match="already registered"):
        @register_metric("z_score_mean")
        def _conflicting(artefacts):  # noqa: ARG001
            return 1.0


def test_get_metric_rejects_unknown_name():
    with pytest.raises(KeyError, match="unknown metric"):
        get_metric("not_a_real_metric")


def test_registry_lists_all_four_spec_metrics():
    assert set(METRIC_REGISTRY.names()) >= {
        "z_score_mean", "hit_rate", "false_positive_rate", "pass_at_1",
    }
