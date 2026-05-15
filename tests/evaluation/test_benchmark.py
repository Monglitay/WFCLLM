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
