"""Tests for wfcllm.extract.negative_corpus."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from wfcllm.extract.negative_corpus import NegativeCorpusConfig, NegativeCorpusGenerator


class TestNegativeCorpusConfig:
    def test_default_values(self):
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path="data/negative_corpus.jsonl",
        )
        assert cfg.dataset == "humaneval"
        assert cfg.dataset_path == "data/datasets"
        assert cfg.max_new_tokens == 512
        assert cfg.temperature == 0.8
        assert cfg.top_p == 0.95
        assert cfg.top_k == 50
        assert cfg.device == "cuda"
        assert cfg.limit is None

    def test_custom_values(self):
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path="data/out.jsonl",
            dataset="mbpp",
            temperature=1.0,
            limit=5,
        )
        assert cfg.dataset == "mbpp"
        assert cfg.temperature == 1.0
        assert cfg.limit == 5

    def test_unsupported_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be one of"):
            NegativeCorpusConfig(
                lm_model_path="x",
                output_path="y",
                dataset="unknown",
            )


class TestNegativeCorpusGeneratorGenerate:
    def test_generate_returns_string(self):
        """NegativeCorpusGenerator._generate() strips prompt tokens and decodes."""
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path="data/out.jsonl",
            device="cpu",
        )
        gen = NegativeCorpusGenerator.__new__(NegativeCorpusGenerator)
        gen._config = cfg

        mock_tokenizer = MagicMock()
        mock_tokenizer.eos_token_id = 0
        fake_inputs = {"input_ids": torch.zeros(1, 3, dtype=torch.long)}
        mock_tokenizer.return_value = fake_inputs
        mock_tokenizer.decode.return_value = "def foo(): pass"

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.zeros(1, 10, dtype=torch.long)

        gen._model = mock_model
        gen._tokenizer = mock_tokenizer
        gen._device = "cpu"

        result = gen._generate("def foo():")

        assert isinstance(result, str)
        assert result == "def foo(): pass"
        decoded_arg = mock_tokenizer.decode.call_args[0][0]
        assert decoded_arg.shape == (7,)


class TestNegativeCorpusGeneratorRun:
    def test_run_writes_jsonl(self, tmp_path):
        """run() writes one JSONL record per prompt with correct fields."""
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path=str(tmp_path / "out.jsonl"),
            device="cpu",
        )
        gen = NegativeCorpusGenerator.__new__(NegativeCorpusGenerator)
        gen._config = cfg
        gen._device = "cpu"
        gen._generate = MagicMock(side_effect=["def foo(): pass", "def bar(): pass"])

        prompts = [
            {"id": "HumanEval/0", "prompt": "def foo():"},
            {"id": "HumanEval/1", "prompt": "def bar():"},
        ]

        with patch("wfcllm.extract.negative_corpus.load_prompts", return_value=prompts), \
             patch("torch.cuda.is_available", return_value=False):
            out_path = gen.run()

        assert Path(out_path).exists()
        lines = Path(out_path).read_text().strip().splitlines()
        assert len(lines) == 2

        record = json.loads(lines[0])
        assert record["id"] == "HumanEval/0"
        assert record["generated_code"] == "def foo(): pass"
        assert record["dataset"] == "humaneval"
        assert "prompt" in record

    def test_run_respects_limit(self, tmp_path):
        """run() processes only first `limit` prompts when limit is set."""
        cfg = NegativeCorpusConfig(
            lm_model_path="data/models/my-model",
            output_path=str(tmp_path / "out.jsonl"),
            device="cpu",
            limit=1,
        )
        gen = NegativeCorpusGenerator.__new__(NegativeCorpusGenerator)
        gen._config = cfg
        gen._device = "cpu"
        gen._generate = MagicMock(return_value="def foo(): pass")

        prompts = [
            {"id": "HumanEval/0", "prompt": "def foo():"},
            {"id": "HumanEval/1", "prompt": "def bar():"},
        ]

        with patch("wfcllm.extract.negative_corpus.load_prompts", return_value=prompts), \
             patch("torch.cuda.is_available", return_value=False):
            gen.run()

        assert gen._generate.call_count == 1
