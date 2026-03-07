"""Tests for wfcllm.encoder.model."""

import pytest
import torch
from transformers import AutoTokenizer

from wfcllm.encoder.model import SemanticEncoder
from wfcllm.encoder.config import EncoderConfig


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Salesforce/codet5-base")


class TestSemanticEncoderFullFinetune:
    """Tests with LoRA disabled (full finetune, FP32)."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(use_lora=False, use_bf16=False, embed_dim=128)
        return SemanticEncoder(config=config)

    def test_output_shape(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 128)

    def test_output_normalized(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_batch_input(self, model, tokenizer):
        texts = ["x = 1", "y = 2", "for i in range(10):\n    print(i)"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (3, 128)

    def test_all_params_trainable(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == total


class TestSemanticEncoderLoRA:
    """Tests with LoRA enabled (default config)."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(use_lora=True, use_bf16=False, embed_dim=128)
        return SemanticEncoder(config=config)

    def test_output_shape(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 128)

    def test_output_normalized(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fewer_trainable_params(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total, "LoRA should freeze most parameters"
        # LoRA typically trains <5% of params
        ratio = trainable / total
        assert ratio < 0.10, f"Expected <10% trainable params, got {ratio:.2%}"

    def test_deterministic(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        model.eval()
        with torch.no_grad():
            out1 = model(inputs["input_ids"], inputs["attention_mask"])
            out2 = model(inputs["input_ids"], inputs["attention_mask"])
        assert torch.allclose(out1, out2)


class TestSemanticEncoderBF16:
    """Tests with BF16 enabled."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(use_lora=False, use_bf16=True, embed_dim=64)
        return SemanticEncoder(config=config)

    def test_encoder_dtype(self, model):
        # Encoder weights should be BF16
        param = next(model.encoder.parameters())
        assert param.dtype == torch.bfloat16

    def test_output_is_float32(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        # Output should be cast back to float32 for downstream use
        assert output.dtype == torch.float32

    def test_different_embed_dim(self, tokenizer):
        config = EncoderConfig(use_lora=False, use_bf16=False, embed_dim=64)
        model = SemanticEncoder(config=config)
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 64)
