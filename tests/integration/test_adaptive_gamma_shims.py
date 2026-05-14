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
