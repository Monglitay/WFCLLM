"""Unit tests for SemanticChannel — the LSH+gamma slice of WatermarkGenerator."""
from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_semantic_channel_module_exists():
    from wfcllm.watermark import semantic_channel
    assert hasattr(semantic_channel, "SemanticChannel")


def test_semantic_channel_holds_orchestrator_reference():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace()
    sc = SemanticChannel(orch)
    assert sc._orch is orch


def test_resolve_gamma_for_entropy_units_falls_back_to_fixed_when_no_schedule():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    from wfcllm.watermark.adaptive_gamma.schedule import quantize_gamma

    orch = SimpleNamespace(
        _config=SimpleNamespace(lsh_d=4, lsh_gamma=0.5),
        _gamma_schedule=None,
        _entropy_est=SimpleNamespace(estimate_block_entropy_units=lambda _t: 7),
    )
    sc = SemanticChannel(orch)
    result = sc.resolve_gamma_for_entropy_units(7)
    assert result == quantize_gamma(0.5, 4)


def test_is_adaptive_runtime_enabled_tracks_orch_gamma_schedule():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(_gamma_schedule=None)
    sc = SemanticChannel(orch)
    assert sc.is_adaptive_runtime_enabled() is False
    orch._gamma_schedule = object()
    assert sc.is_adaptive_runtime_enabled() is True


def test_adaptive_mode_reports_fixed_when_disabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=None,
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(strategy="piecewise_quantile")),
    )
    sc = SemanticChannel(orch)
    assert sc.adaptive_mode() == "fixed"


def test_adaptive_mode_reports_strategy_when_enabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=object(),
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(strategy="piecewise_quantile")),
    )
    sc = SemanticChannel(orch)
    assert sc.adaptive_mode() == "piecewise_quantile"


def test_profile_id_returns_none_when_disabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=None,
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(profile_id="x")),
    )
    sc = SemanticChannel(orch)
    assert sc.profile_id() is None


def test_profile_id_returns_config_value_when_enabled():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    orch = SimpleNamespace(
        _gamma_schedule=object(),
        _config=SimpleNamespace(adaptive_gamma=SimpleNamespace(profile_id="entropy-profile-v1")),
    )
    sc = SemanticChannel(orch)
    assert sc.profile_id() == "entropy-profile-v1"


def test_build_alignment_summary_is_pure():
    from wfcllm.watermark.semantic_channel import SemanticChannel
    sc = SemanticChannel(SimpleNamespace())
    summary = sc.build_alignment_summary(5, [object(), object(), object(), object(), object()])
    assert summary == {
        "final_block_count": 5,
        "generator_total_blocks": 5,
        "block_count_matches_total_blocks": True,
    }
    summary2 = sc.build_alignment_summary(7, [object(), object(), object()])
    assert summary2 == {
        "final_block_count": 3,
        "generator_total_blocks": 7,
        "block_count_matches_total_blocks": False,
    }
