"""Tests for wfcllm.common.dataset_loader."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts


class TestLoadPrompts:
    def test_supported_datasets_constant(self):
        assert "humaneval" in SUPPORTED_DATASETS
        assert "mbpp" in SUPPORTED_DATASETS

    def test_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            load_prompts("unknown", "data/datasets")

    @patch("wfcllm.common.dataset_loader.load_dataset")
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

    @patch("wfcllm.common.dataset_loader.load_dataset")
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
