"""Tests for wfcllm/evaluation/* migration: new paths work, old paths still work via shim."""
from __future__ import annotations

import warnings

import pytest


# --- code_execution: new path works ---

def test_code_execution_new_path_importable_and_callable():
    from wfcllm.evaluation.code_execution import (
        compute_pass_at_k,
        compute_roc_auc,
        compute_tpr_at_fpr,
        load_jsonl_records,
        write_jsonl_records,
    )

    records = [
        {"task_id": "t1", "is_correct": True},
        {"task_id": "t1", "is_correct": False},
        {"task_id": "t2", "is_correct": True},
    ]
    assert compute_pass_at_k(records, k=1) == pytest.approx(0.75)


# --- code_execution: old path is deprecated shim ---

def test_offline_code_eval_old_path_emits_deprecation_warning():
    """Old wfcllm.common.offline_code_eval should warn but still expose public API."""
    # Force a fresh import so the warning fires deterministically.
    import importlib
    import sys
    sys.modules.pop("wfcllm.common.offline_code_eval", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.common.offline_code_eval")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.evaluation.code_execution" in str(w.message)
        for w in caught
    )
    # Public symbols are still resolvable from the old path.
    assert callable(module.compute_pass_at_k)
    assert callable(module.compute_roc_auc)
    assert callable(module.annotate_correctness_from_references)


def test_offline_code_eval_old_and_new_paths_share_symbols():
    """Symbols re-exported from the shim refer to the same callable as the new path."""
    import importlib
    import sys
    sys.modules.pop("wfcllm.common.offline_code_eval", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.common.offline_code_eval")
    new = importlib.import_module("wfcllm.evaluation.code_execution")
    for name in (
        "compute_pass_at_k",
        "compute_roc_auc",
        "annotate_correctness_from_references",
        "load_jsonl_records",
        "write_jsonl_records",
    ):
        assert getattr(old, name) is getattr(new, name), name


# --- detection_report: new path works ---

def test_detection_report_new_path_importable_and_callable():
    from wfcllm.evaluation.detection_report import (
        build_offline_regression_report,
        load_detail_artifact,
        load_summary_artifact,
        load_watermarked_artifact,
        write_offline_regression_report,
    )

    assert callable(build_offline_regression_report)
    assert callable(load_detail_artifact)


# --- detection_report: old path is deprecated shim ---

def test_offline_analysis_old_path_emits_deprecation_warning():
    import importlib
    import sys
    sys.modules.pop("wfcllm.extract.offline_analysis", None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        module = importlib.import_module("wfcllm.extract.offline_analysis")
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "wfcllm.evaluation.detection_report" in str(w.message)
        for w in caught
    )
    assert callable(module.build_offline_regression_report)
    assert callable(module.load_summary_artifact)


def test_offline_analysis_old_and_new_paths_share_symbols():
    import importlib
    import sys
    sys.modules.pop("wfcllm.extract.offline_analysis", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        old = importlib.import_module("wfcllm.extract.offline_analysis")
    new = importlib.import_module("wfcllm.evaluation.detection_report")
    for name in (
        "build_offline_regression_report",
        "load_summary_artifact",
        "load_detail_artifact",
        "load_watermarked_artifact",
        "write_offline_regression_report",
        "SummaryArtifact",
        "DetailArtifact",
        "WatermarkedArtifact",
        "DetailDelta",
        "DetailDeltaReport",
    ):
        assert getattr(old, name) is getattr(new, name), name
