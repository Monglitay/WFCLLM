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
