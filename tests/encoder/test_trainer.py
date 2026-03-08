"""Tests for wfcllm.encoder.trainer."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from wfcllm.encoder.trainer import ContrastiveTrainer, triplet_cosine_loss
from wfcllm.encoder.config import EncoderConfig

LOCAL_MODEL = "data/models/codet5-base"


class TestTripletCosineLoss:
    def test_zero_loss_when_perfect(self):
        """Loss should be 0 when positive is identical and negative is orthogonal."""
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])  # cos = 1.0
        negative = torch.tensor([[0.0, 1.0]])  # cos = 0.0
        loss = triplet_cosine_loss(anchor, positive, negative, margin=0.3)
        assert loss.item() == 0.0

    def test_positive_loss_when_bad(self):
        """Loss > 0 when positive is far and negative is close."""
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.0, 1.0]])  # cos = 0.0
        negative = torch.tensor([[0.9, 0.1]])  # cos ≈ 0.9
        loss = triplet_cosine_loss(anchor, positive, negative, margin=0.3)
        assert loss.item() > 0.0

    def test_batch(self):
        anchor = torch.randn(8, 128)
        positive = torch.randn(8, 128)
        negative = torch.randn(8, 128)
        loss = triplet_cosine_loss(anchor, positive, negative, margin=0.3)
        assert loss.shape == ()  # scalar


class TestContrastiveTrainer:
    @pytest.fixture
    def dummy_setup(self):
        """Create minimal trainer with dummy data for smoke testing."""
        from wfcllm.encoder.model import SemanticEncoder
        config = EncoderConfig(
            model_name=LOCAL_MODEL,
            embed_dim=32, epochs=1, batch_size=2, lr=1e-4,
            use_lora=False, use_bf16=False,
            checkpoint_dir="/tmp/wfcllm_test_ckpt",
            results_dir="/tmp/wfcllm_test_results",
        )
        model = SemanticEncoder(config=config)

        # Create tiny synthetic dataset
        seq_len = 32
        n_samples = 4

        def make_loader():
            data = {}
            for prefix in ("anchor", "positive", "negative"):
                data[f"{prefix}_input_ids"] = torch.randint(0, 100, (n_samples, seq_len))
                data[f"{prefix}_attention_mask"] = torch.ones(n_samples, seq_len, dtype=torch.long)
            dataset = _DictDataset(data, n_samples)
            return DataLoader(dataset, batch_size=2)

        return model, config, make_loader(), make_loader()

    def test_train_epoch_returns_loss(self, dummy_setup):
        model, config, train_loader, val_loader = dummy_setup
        trainer = ContrastiveTrainer(model, train_loader, val_loader, config)
        metrics = trainer.train_epoch(epoch=0)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_validate_returns_metrics(self, dummy_setup):
        model, config, train_loader, val_loader = dummy_setup
        trainer = ContrastiveTrainer(model, train_loader, val_loader, config)
        metrics = trainer.validate(epoch=0)
        assert "val_loss" in metrics


class _DictDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, n: int):
        self.data = data
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
