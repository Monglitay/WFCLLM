from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "analyze_sawr_no_quality_gate_goal.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_sawr_no_quality_gate_goal",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_quality_policy_objective_ignores_pass_rate() -> None:
    module = _load_module()
    high_pass_low_evidence = {
        "pass": 120,
        "pass_rate": 0.73,
        "auroc": 0.62,
        "tp_at_8fp": 40,
        "fp": 8,
        "positive_insufficient": 70,
        "mean_proxy_windows": 1.0,
        "mean_scoreable_contexts": 1.0,
    }
    low_pass_high_evidence = {
        "pass": 80,
        "pass_rate": 0.49,
        "auroc": 0.84,
        "tp_at_8fp": 96,
        "fp": 8,
        "positive_insufficient": 10,
        "mean_proxy_windows": 4.0,
        "mean_scoreable_contexts": 3.0,
    }

    assert module._policy_objective_no_quality(  # noqa: SLF001
        low_pass_high_evidence,
    ) > module._policy_objective_no_quality(  # noqa: SLF001
        high_pass_low_evidence,
    )


def test_detector_input_integrity_rejects_private_or_score_fields() -> None:
    module = _load_module()
    rows = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def foo():\n",
            "final_code": "def foo():\n    return 1\n",
        },
        {
            "id": "HumanEval/1",
            "dataset": "humaneval",
            "prompt": "def bar():\n",
            "final_code": "def bar():\n    return 2\n",
            "detector_score": 0.9,
        },
    ]

    report = module.detector_input_integrity(rows, input_path="candidate.jsonl")

    assert report["allowed_fields"] == ["dataset", "final_code", "id", "prompt"]
    assert report["row_count"] == 2
    assert report["is_code_only"] is False
    assert report["forbidden_fields"] == ["detector_score"]


def test_final_summary_keeps_input_rows_out_of_metrics_json() -> None:
    module = _load_module()
    final = {
        "name": "evidence_only_selector_diagnostic",
        "target_achieved": False,
        "input_path": "inputs/final.jsonl",
        "input_rows": [{"id": "HumanEval/0", "final_code": "def f(): pass"}],
    }
    matrix = {"rows": []}
    model_negative = {"rows": []}
    compliance = {
        "summary": {
            "pass_test_correctness_proxy_used_for_generation_retry_selector_calibration": False,
            "strict_code_only_detector_required": True,
        }
    }

    summary = module.build_final_summary(
        final,
        matrix,
        model_negative,
        compliance,
    )

    assert summary["final_candidate"]["input_path"] == "inputs/final.jsonl"
    assert "input_rows" not in summary["final_candidate"]


def test_selector_compatible_candidate_data_excludes_ordinal_keying() -> None:
    module = _load_module()
    candidate_data = {
        "strict_code_only_baseline": {"details": {}},
        "evidence_retry_seed7x3": {"details": {}},
        "ordinal_anchor_seed8": {"details": {}},
    }

    compatible, excluded = module.selector_compatible_candidate_data(candidate_data)

    assert list(compatible) == [
        "strict_code_only_baseline",
        "evidence_retry_seed7x3",
    ]
    assert excluded == ["ordinal_anchor_seed8"]
