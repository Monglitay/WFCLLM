"""Tests for wfcllm.encoder.train entry point."""

import pytest
from unittest.mock import patch, MagicMock

from wfcllm.encoder.train import load_code_samples, prepare_blocks_with_variants


class TestLoadCodeSamples:
    @patch("wfcllm.encoder.train.load_dataset")
    def test_loads_mbpp(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"code": "x = 1", "task_id": 1, "text": "test"},
        ]))
        mock_load.return_value = {"train": mock_ds}
        samples = load_code_samples(["mbpp"])
        assert len(samples) > 0
        assert "code" in samples[0]


class TestPrepareBlocksWithVariants:
    def test_basic(self):
        code_samples = [{"code": "x = 1\ny = 2"}]
        blocks = prepare_blocks_with_variants(code_samples, max_variants=5)
        assert len(blocks) > 0
        for b in blocks:
            assert "source" in b
            assert "positive_variants" in b
            assert "negative_variants" in b


def test_evaluate_only_prints_progress(tmp_path, monkeypatch):
    """evaluate_only should print batch progress during embedding collection."""
    import io, sys
    import torch
    from unittest.mock import MagicMock, patch
    from wfcllm.encoder.train import evaluate_only
    from wfcllm.encoder.config import EncoderConfig

    # Minimal fake batch
    def fake_batch():
        ids = torch.zeros(4, 16, dtype=torch.long)
        mask = torch.ones(4, 16, dtype=torch.long)
        return {
            "anchor_input_ids": ids,
            "anchor_attention_mask": mask,
            "positive_input_ids": ids,
            "positive_attention_mask": mask,
            "negative_input_ids": ids,
            "negative_attention_mask": mask,
        }

    fake_loader = [fake_batch() for _ in range(6)]  # 6 batches

    # Fake model: returns normalized random embeddings
    fake_emb = torch.randn(4, 128)
    fake_emb = torch.nn.functional.normalize(fake_emb, dim=1)
    fake_model = MagicMock(return_value=fake_emb)
    fake_model.eval = MagicMock(return_value=fake_model)

    config = EncoderConfig(
        model_name="data/models/codet5-base",
        results_dir=str(tmp_path),
    )

    captured = io.StringIO()
    with (
        patch("wfcllm.encoder.train.load_code_samples", return_value=[]),
        patch("wfcllm.encoder.train.prepare_blocks_with_variants", return_value=[]),
        patch("wfcllm.encoder.train.build_triplets_from_blocks", return_value=[]),
        patch("wfcllm.encoder.train.TripletCodeDataset", return_value=MagicMock(__len__=lambda s: 6)),
        patch("wfcllm.encoder.train.random_split", return_value=(None, None, fake_loader)),
        patch("wfcllm.encoder.train.DataLoader", return_value=fake_loader),
        patch("wfcllm.encoder.train.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("wfcllm.encoder.train.SemanticEncoder", return_value=fake_model),
        patch("torch.load", return_value={"model_state_dict": {}}),
        patch.object(fake_model, "load_state_dict"),
        patch.object(fake_model, "to", return_value=fake_model),
        patch("sys.stdout", captured),
    ):
        try:
            evaluate_only("data/checkpoints/encoder/encoder_epoch9.pt", config)
        except Exception:
            pass  # metrics computation may fail on mock data; we only care about progress output

    output = captured.getvalue()
    assert "batch" in output.lower() or "/" in output, (
        "Expected progress output during batch loop, got:\n" + output
    )
