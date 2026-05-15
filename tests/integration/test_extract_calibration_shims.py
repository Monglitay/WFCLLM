"""Tests for wfcllm/extract/calibration/* migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import importlib
import sys
import warnings


# --- threshold: new path works ---

def test_threshold_new_path_importable():
    from wfcllm.extract.calibration.threshold import ThresholdCalibrator
    assert callable(ThresholdCalibrator)
