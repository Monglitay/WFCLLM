"""Tests for wfcllm.common.dataset_loader."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from wfcllm.datasets.loaders.local import (
    SUPPORTED_DATASETS,
    load_mbpp_generation_samples,
    load_prompts,
    load_reference_solutions,
)


class TestLoadPrompts:
    def test_supported_datasets_constant(self):
        assert "humaneval" in SUPPORTED_DATASETS
        assert "mbpp" in SUPPORTED_DATASETS

    def test_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            load_prompts("unknown", "data/datasets")

    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_humaneval_returns_id_and_prompt(self, mock_load):
        fake_split = [{"task_id": "HumanEval/0", "prompt": "def foo():"}]
        mock_ds = {"test": fake_split}
        mock_load.return_value = mock_ds

        prompts = load_prompts("humaneval", "data/datasets")

        mock_load.assert_called_once_with(
            "openai/openai_humaneval",
            cache_dir="data/datasets/humaneval",
            download_mode="reuse_cache_if_exists",
        )
        assert len(prompts) == 1
        assert prompts[0]["id"] == "HumanEval/0"
        assert prompts[0]["prompt"] == "def foo():"

    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_mbpp_returns_id_and_prompt(self, mock_load):
        fake_split = [{"task_id": 1, "text": "Write a function"}]
        mock_ds = {"train": fake_split}
        mock_load.return_value = mock_ds

        prompts = load_prompts("mbpp", "data/datasets")

        mock_load.assert_called_once_with(
            "google-research-datasets/mbpp",
            "full",
            cache_dir="data/datasets/mbpp",
            download_mode="reuse_cache_if_exists",
        )
        assert len(prompts) == 1
        assert prompts[0]["id"] == "mbpp/1"
        assert prompts[0]["prompt"] == "Write a function"

    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_load_prompts_applies_offset_and_limit(self, mock_load):
        fake_split = [
            {"task_id": "HumanEval/0", "prompt": "def a():\n"},
            {"task_id": "HumanEval/1", "prompt": "def b():\n"},
            {"task_id": "HumanEval/2", "prompt": "def c():\n"},
        ]
        mock_load.return_value = {"test": fake_split}

        prompts = load_prompts(
            "humaneval",
            "data/datasets",
            sample_offset=1,
            sample_limit=1,
        )

        assert prompts == [{"id": "HumanEval/1", "prompt": "def b():\n"}]

    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_load_prompts_rejects_negative_offset(self, mock_load):
        mock_load.return_value = {"test": []}

        with pytest.raises(ValueError, match="sample_offset must be non-negative"):
            load_prompts("humaneval", "data/datasets", sample_offset=-1)

    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_load_prompts_rejects_negative_limit(self, mock_load):
        mock_load.return_value = {"test": []}

        with pytest.raises(ValueError, match="sample_limit must be non-negative"):
            load_prompts("humaneval", "data/datasets", sample_limit=-1)


@patch("wfcllm.datasets.loaders.local.load_dataset")
def test_mbpp_generation_samples_keep_only_sanitized_interface_metadata(
    mock_load,
) -> None:
    mock_load.return_value = {
        "test": [
            {
                "task_id": 601,
                "text": "Find the maximum chain length from pairs.",
                "code": (
                    "def max_chain_length(pairs, n):\n"
                    "    secret_reference_value = 987654\n"
                    "    return secret_reference_value\n"
                ),
                "test_list": [
                    "assert max_chain_length([Pair(5, 24)], 4) == 3",
                ],
            }
        ]
    }

    rows = load_mbpp_generation_samples("data/datasets")

    assert len(rows) == 1
    assert rows[0]["id"] == "mbpp/601"
    assert rows[0]["interface_extraction_status"] == "interface_aware"
    assert rows[0]["interface_function_name"] == "max_chain_length"
    assert rows[0]["interface_positional_arities"] == [2]
    assert rows[0]["interface_parameter_names"] == ["pairs", "n"]
    assert rows[0]["interface_helper_classes"] == ["Pair"]
    assert rows[0]["prompt"].startswith("class Pair:")
    assert "def max_chain_length(pairs, n):" in rows[0]["prompt"]
    serialized = repr(rows[0])
    assert "assert max_chain_length" not in serialized
    assert "Pair(5, 24)" not in serialized
    assert "== 3" not in serialized
    assert "987654" not in rows[0]["prompt"]
    assert "secret_reference_value" not in rows[0]["prompt"]


class TestLoadReferenceSolutions:
    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_humaneval_returns_id_prompt_and_canonical_solution(self, mock_load):
        fake_split = [
            {
                "task_id": "HumanEval/0",
                "prompt": "def foo():\n",
                "canonical_solution": "    return 1\n",
            }
        ]
        mock_ds = {"test": fake_split}
        mock_load.return_value = mock_ds

        rows = load_reference_solutions("humaneval", "data/datasets")

        assert rows == [
            {
                "id": "HumanEval/0",
                "prompt": "def foo():\n",
                "generated_code": "    return 1\n",
            }
        ]

    @patch("wfcllm.datasets.loaders.local.load_dataset")
    def test_mbpp_returns_id_prompt_and_reference_code(self, mock_load):
        fake_split = [
            {
                "task_id": 1,
                "text": "Write a function",
                "code": "def f():\n    return 1\n",
            }
        ]
        mock_ds = {"train": fake_split}
        mock_load.return_value = mock_ds

        rows = load_reference_solutions("mbpp", "data/datasets")

        assert rows == [
            {
                "id": "mbpp/1",
                "prompt": "Write a function",
                "generated_code": "def f():\n    return 1\n",
            }
        ]
