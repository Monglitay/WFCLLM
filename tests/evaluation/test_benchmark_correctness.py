"""Test that _evaluate_correctness prepends prompt to generated_code."""
import pytest
from unittest.mock import patch, MagicMock
from wfcllm.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner, TestExecutor
from wfcllm.datasets.loaders.local import TestCase


def test_evaluate_correctness_prepends_prompt():
    """The full function (prompt + body) must be passed to the executor."""
    config = BenchmarkConfig(dataset="humaneval", config_path="configs/base_config.json")
    runner = BenchmarkRunner(config)

    records = [{
        "id": "HumanEval/0",
        "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
        "generated_code": "    return a + b\n",
    }]
    test_cases = {
        "HumanEval/0": TestCase(
            task_id="HumanEval/0",
            entry_point="add",
            test_code="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        ),
    }
    executor = TestExecutor(timeout=5.0)
    results = runner._evaluate_correctness(records, test_cases, executor)
    assert results == [True], "prompt + generated_code should form a valid function"
