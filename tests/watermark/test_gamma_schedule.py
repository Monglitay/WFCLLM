"""Tests for wfcllm.watermark.gamma_schedule."""

from wfcllm.watermark.entropy_profile import EntropyProfile
from wfcllm.watermark.gamma_schedule import PiecewiseQuantileSchedule


def _profile() -> EntropyProfile:
    return EntropyProfile(
        language="python",
        model_family="codellama",
        quantiles_units_map={
            "p10": 100,
            "p50": 500,
            "p75": 800,
            "p90": 900,
            "p95": 1000,
        },
    )


def test_resolve_returns_target_k_and_effective_for_anchor_entropy():
    schedule = PiecewiseQuantileSchedule(profile=_profile())

    resolution = schedule.resolve(entropy_units=800, lsh_d=3)

    assert resolution.gamma_target == 0.55
    assert resolution.k == 4
    assert resolution.gamma_effective == 0.5


def test_resolve_clips_below_and_above_anchor_range():
    schedule = PiecewiseQuantileSchedule(profile=_profile())

    below = schedule.resolve(entropy_units=50, lsh_d=2)
    above = schedule.resolve(entropy_units=2000, lsh_d=2)

    assert below.gamma_target == 0.95
    assert below.k == 3
    assert below.gamma_effective == 0.75

    assert above.gamma_target == 0.25
    assert above.k == 1
    assert above.gamma_effective == 0.25


def test_resolve_interpolates_between_quantile_anchors_and_quantizes():
    schedule = PiecewiseQuantileSchedule(profile=_profile())

    resolution = schedule.resolve(entropy_units=650, lsh_d=4)

    assert resolution.gamma_target == 0.65
    assert resolution.k == 10
    assert resolution.gamma_effective == 0.625


def test_quantization_clips_k_to_valid_range():
    schedule = PiecewiseQuantileSchedule(profile=_profile())

    resolution = schedule.resolve(entropy_units=100, lsh_d=1)

    assert resolution.k == 1
    assert resolution.gamma_effective == 0.5


def test_resolve_rejects_lsh_dimension_below_one():
    schedule = PiecewiseQuantileSchedule(profile=_profile())

    try:
        schedule.resolve(entropy_units=100, lsh_d=0)
    except ValueError as exc:
        assert str(exc) == "lsh_d must be >= 1"
    else:
        raise AssertionError("Expected ValueError for lsh_d < 1")
