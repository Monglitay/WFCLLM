"""Tests for the benchmark evaluation module."""
from __future__ import annotations

import pytest

from wfcllm.datasets.loaders.local import load_test_cases


def test_load_test_cases_humaneval() -> None:
    cases = load_test_cases("humaneval", "data/datasets")
    assert len(cases) == 164
    case = cases["HumanEval/0"]
    assert case.task_id == "HumanEval/0"
    assert case.entry_point is not None
    assert "check" in case.test_code or "assert" in case.test_code


def test_load_test_cases_mbpp() -> None:
    cases = load_test_cases("mbpp", "data/datasets")
    assert len(cases) > 0
    first_key = next(iter(cases))
    case = cases[first_key]
    assert case.task_id.startswith("mbpp/")
    assert case.entry_point is None
    assert "assert" in case.test_code


def test_load_test_cases_invalid_dataset() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        load_test_cases("invalid", "data/datasets")


from wfcllm.evaluation.benchmark import TestExecutor


def test_executor_passing_code() -> None:
    executor = TestExecutor(timeout=5.0)
    code = "def add(a, b):\n    return a + b\n"
    test_code = "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
    assert executor.run_test(code, test_code) is True


def test_executor_failing_code() -> None:
    executor = TestExecutor(timeout=5.0)
    code = "def add(a, b):\n    return a - b\n"
    test_code = "assert add(1, 2) == 3\n"
    assert executor.run_test(code, test_code) is False


def test_executor_timeout() -> None:
    executor = TestExecutor(timeout=1.0)
    code = "import time\ndef slow():\n    time.sleep(10)\n"
    test_code = "slow()\n"
    assert executor.run_test(code, test_code) is False


def test_executor_syntax_error() -> None:
    executor = TestExecutor(timeout=5.0)
    code = "def broken(\n"
    test_code = "assert True\n"
    assert executor.run_test(code, test_code) is False


from wfcllm.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner


def test_benchmark_runner_pass_at_k() -> None:
    """BenchmarkRunner computes pass@k from execution results."""
    config = BenchmarkConfig(
        dataset="humaneval",
        config_path="configs/base_config.json",
        dataset_path="data/datasets",
        num_candidates=2,
        output_dir="/tmp/test_benchmark",
    )
    # Mock: 2 tasks, 2 candidates each. Task A: both pass. Task B: 1 pass, 1 fail.
    mock_records = [
        {"id": "HumanEval/0", "generated_code": "def f(): return 1", "candidate_index": 0},
        {"id": "HumanEval/0", "generated_code": "def f(): return 1", "candidate_index": 1},
        {"id": "HumanEval/1", "generated_code": "def g(): return 2", "candidate_index": 0},
        {"id": "HumanEval/1", "generated_code": "def g(): return 2", "candidate_index": 1},
    ]
    correctness = [True, True, True, False]

    runner = BenchmarkRunner(config)
    result = runner._compute_pass_at_k(mock_records, correctness)
    assert result["pass_at_1"] == pytest.approx(0.75)
    assert result["pass_at_2"] == pytest.approx(1.0)


def test_benchmark_runner_compute_auroc() -> None:
    """BenchmarkRunner computes AUROC from positive/negative details."""
    config = BenchmarkConfig(
        dataset="humaneval",
        config_path="configs/base_config.json",
        dataset_path="data/datasets",
        output_dir="/tmp/test_benchmark",
    )
    positive_details = [
        {"id": "HumanEval/0", "z_score": 3.0, "lexical_z_score": 2.5, "joint_score": 4.0},
        {"id": "HumanEval/1", "z_score": 2.5, "lexical_z_score": 2.0, "joint_score": 3.5},
    ]
    negative_details = [
        {"id": "HumanEval/0", "z_score": 0.5, "lexical_z_score": 0.3, "joint_score": 0.6},
        {"id": "HumanEval/1", "z_score": 0.2, "lexical_z_score": 0.1, "joint_score": 0.3},
    ]
    runner = BenchmarkRunner(config)
    result = runner._compute_detection_metrics(positive_details, negative_details)

    assert result["semantic"]["auroc"] == pytest.approx(1.0)
    assert result["lexical"]["auroc"] == pytest.approx(1.0)
    assert result["joint"]["auroc"] == pytest.approx(1.0)
    assert 0.0 <= result["semantic"]["tpr_at_1pct_fpr"] <= 1.0
