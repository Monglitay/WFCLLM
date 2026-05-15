"""Tests for wfcllm/extract/calibration/* migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import importlib
import sys
import warnings


# --- threshold: new path works ---

def test_threshold_new_path_importable():
    from wfcllm.extract.calibration.threshold import ThresholdCalibrator
    assert callable(ThresholdCalibrator)


# --- threshold: old path is a deprecated shim ---

def test_threshold_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.extract.calibrator", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.extract.calibrator")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.extract.calibration.threshold" in str(w.message)
        for w in caught
    )
    assert callable(module.ThresholdCalibrator)


def test_threshold_old_and_new_paths_share_symbol():
    sys.modules.pop("wfcllm.extract.calibrator", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.extract.calibrator")
    new = importlib.import_module("wfcllm.extract.calibration.threshold")
    assert old.ThresholdCalibrator is new.ThresholdCalibrator


# --- negative_corpus: new path works ---

def test_negative_corpus_new_path_importable():
    from wfcllm.extract.calibration.negative_corpus import (
        NegativeCorpusConfig,
        NegativeCorpusGenerator,
    )
    cfg = NegativeCorpusConfig(lm_model_path="/tmp/fake", output_path="/tmp/out.jsonl")
    assert cfg.dataset == "humaneval"
    assert callable(NegativeCorpusGenerator)


# --- negative_corpus: old path is a deprecated shim ---

def test_negative_corpus_old_path_emits_deprecation_warning():
    sys.modules.pop("wfcllm.extract.negative_corpus", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.extract.negative_corpus")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.extract.calibration.negative_corpus" in str(w.message)
        for w in caught
    )
    assert callable(module.NegativeCorpusGenerator)
    assert callable(module.NegativeCorpusConfig)


def test_negative_corpus_old_and_new_paths_share_symbols():
    sys.modules.pop("wfcllm.extract.negative_corpus", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.extract.negative_corpus")
    new = importlib.import_module("wfcllm.extract.calibration.negative_corpus")
    assert old.NegativeCorpusGenerator is new.NegativeCorpusGenerator
    assert old.NegativeCorpusConfig is new.NegativeCorpusConfig
