"""Tests for wfcllm/watermark/adaptive_gamma/* migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import importlib
import sys
import warnings

import pytest


# --- entropy: new path works ---

def test_entropy_new_path_importable_and_callable():
    from wfcllm.watermark.adaptive_gamma.entropy import (
        ENTROPY_SCALE,
        NodeEntropyEstimator,
    )
    estimator = NodeEntropyEstimator()
    assert ENTROPY_SCALE == 10000
    assert estimator.estimate_block_entropy("") == 0.0
    assert estimator.estimate_block_entropy("x = 1") > 0.0


# --- entropy: old path is a deprecated shim ---

def test_entropy_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.watermark.entropy", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.watermark.entropy")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.watermark.adaptive_gamma.entropy" in str(w.message)
        for w in caught
    )
    assert callable(module.NodeEntropyEstimator)
    assert module.ENTROPY_SCALE == 10000


def test_entropy_old_and_new_paths_share_symbols():
    sys.modules.pop("wfcllm.watermark.entropy", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.watermark.entropy")
    new = importlib.import_module("wfcllm.watermark.adaptive_gamma.entropy")
    assert old.NodeEntropyEstimator is new.NodeEntropyEstimator
    assert old.ENTROPY_SCALE is new.ENTROPY_SCALE


# --- profile: new path works ---

def test_profile_new_path_importable_and_callable(tmp_path):
    from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
    profile = EntropyProfile(
        language="python",
        model_family="codet5",
        quantiles_units_map={"p10": 1, "p50": 2, "p75": 3, "p90": 4, "p95": 5},
    )
    assert profile.quantile_units("p50") == 2


# --- profile: old path is a deprecated shim ---

def test_entropy_profile_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.watermark.entropy_profile", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.watermark.entropy_profile")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.watermark.adaptive_gamma.profile" in str(w.message)
        for w in caught
    )
    assert hasattr(module, "EntropyProfile")


def test_entropy_profile_old_and_new_paths_share_symbols():
    sys.modules.pop("wfcllm.watermark.entropy_profile", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.watermark.entropy_profile")
    new = importlib.import_module("wfcllm.watermark.adaptive_gamma.profile")
    assert old.EntropyProfile is new.EntropyProfile


# --- schedule: new path works ---

def test_schedule_new_path_importable_and_callable():
    from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
    from wfcllm.watermark.adaptive_gamma.schedule import (
        GammaResolution,
        PiecewiseQuantileSchedule,
        quantize_gamma,
    )
    profile = EntropyProfile(
        language="python",
        model_family="codet5",
        quantiles_units_map={"p10": 100, "p50": 200, "p75": 300, "p90": 400, "p95": 500},
    )
    schedule = PiecewiseQuantileSchedule(profile=profile)
    resolution = schedule.resolve(entropy_units=200, lsh_d=3)
    assert isinstance(resolution, GammaResolution)
    assert 1 <= resolution.k <= 7  # 2**3 - 1


def test_quantize_gamma_clamps_low():
    from wfcllm.watermark.adaptive_gamma.schedule import quantize_gamma
    resolution = quantize_gamma(0.0, lsh_d=3)
    assert resolution.k == 1


# --- schedule: old path is a deprecated shim ---

def test_gamma_schedule_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.watermark.gamma_schedule", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.watermark.gamma_schedule")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.watermark.adaptive_gamma.schedule" in str(w.message)
        for w in caught
    )
    assert hasattr(module, "PiecewiseQuantileSchedule")
    assert hasattr(module, "GammaResolution")
    assert callable(module.quantize_gamma)


def test_gamma_schedule_old_and_new_paths_share_symbols():
    sys.modules.pop("wfcllm.watermark.gamma_schedule", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.watermark.gamma_schedule")
    new = importlib.import_module("wfcllm.watermark.adaptive_gamma.schedule")
    for name in ("GammaResolution", "PiecewiseQuantileSchedule", "quantize_gamma"):
        assert getattr(old, name) is getattr(new, name), name


# --- calibrate: new path works ---

def test_calibrate_new_path_writes_profile(tmp_path):
    from wfcllm.watermark.adaptive_gamma.calibrate import build_entropy_profile_from_log
    log_path = tmp_path / "wm.log"
    log_path.write_text(
        "\n".join(
            f"wfcllm.watermark.generator DEBUG entropy={value:.4f}"
            for value in (0.12, 0.24, 0.36, 0.48, 0.60)
        )
        + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "profile.json"
    build_entropy_profile_from_log(
        input_log=log_path,
        output=out_path,
        language="python",
        model_family="codet5-base",
    )
    import json
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["language"] == "python"
    assert payload["model_family"] == "codet5-base"
    assert payload["sample_count"] == 5
    assert payload["quantiles_units"]["p10"] == 1200
    assert payload["quantiles_units"]["p95"] == 6000


def test_calibrate_raises_when_log_has_no_entropy_lines(tmp_path):
    from wfcllm.watermark.adaptive_gamma.calibrate import build_entropy_profile_from_log
    log_path = tmp_path / "empty.log"
    log_path.write_text("nothing here\n", encoding="utf-8")
    out_path = tmp_path / "profile.json"
    with pytest.raises(ValueError, match="No entropy=<float> entries"):
        build_entropy_profile_from_log(
            input_log=log_path,
            output=out_path,
            language="python",
            model_family="codet5-base",
        )


def test_calibrate_persists_profile_id_when_provided(tmp_path):
    from wfcllm.watermark.adaptive_gamma.calibrate import build_entropy_profile_from_log
    log_path = tmp_path / "wm.log"
    log_path.write_text("entropy=0.5\n", encoding="utf-8")
    out_path = tmp_path / "profile.json"
    build_entropy_profile_from_log(
        input_log=log_path,
        output=out_path,
        language="python",
        model_family="codet5-base",
        profile_id="my-profile-v1",
    )
    import json
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["profile_id"] == "my-profile-v1"


def test_calibrate_loadable_by_entropy_profile(tmp_path):
    """The output JSON must be loadable by EntropyProfile.load."""
    from wfcllm.watermark.adaptive_gamma.calibrate import build_entropy_profile_from_log
    from wfcllm.watermark.adaptive_gamma.profile import EntropyProfile
    log_path = tmp_path / "wm.log"
    log_path.write_text(
        "\n".join(f"entropy={v:.4f}" for v in (0.1, 0.2, 0.3, 0.4, 0.5)) + "\n",
        encoding="utf-8",
    )
    out_path = tmp_path / "profile.json"
    build_entropy_profile_from_log(
        input_log=log_path,
        output=out_path,
        language="python",
        model_family="codet5-base",
    )
    profile = EntropyProfile.load(out_path)
    assert profile.language == "python"
    assert profile.quantile_units("p50") > 0
