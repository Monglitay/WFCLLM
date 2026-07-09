"""Tests for wfcllm.encoder.dataset."""

import torch

from wfcllm.encoder.dataset import TripletCodeDataset, build_triplets_from_blocks


class _FakeTokenizer:
    def __call__(
        self,
        text: str,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> dict[str, torch.Tensor]:
        assert padding == "max_length"
        assert truncation is True
        assert return_tensors == "pt"
        token_count = min(len(text.split()), max_length)
        input_ids = torch.zeros((1, max_length), dtype=torch.long)
        attention_mask = torch.zeros((1, max_length), dtype=torch.long)
        if token_count:
            input_ids[0, :token_count] = torch.arange(1, token_count + 1)
            attention_mask[0, :token_count] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class TestBuildTripletsFromBlocks:
    """Test triplet construction logic with synthetic data."""

    def _make_block(self, source, positive_variants=None, negative_variants=None):
        return {
            "source": source,
            "positive_variants": positive_variants or [],
            "negative_variants": negative_variants or [],
        }

    def test_basic_triplet(self):
        blocks = [
            self._make_block("x = 1", ["x = 2"], ["x = -1"]),
            self._make_block("y = 3", ["y = 4"], ["y = -3"]),
        ]
        triplets = build_triplets_from_blocks(blocks, negative_ratio=0.5, seed=42)
        assert len(triplets) > 0
        for t in triplets:
            assert "anchor" in t
            assert "positive" in t
            assert "negative" in t

    def test_negative_ratio_hard(self):
        """With ratio=1.0, all negatives should be hard (from negative_variants)."""
        blocks = [
            self._make_block("x = 1", ["x = 2"], ["x = -1"]),
        ]
        # Need other blocks for random negatives
        all_blocks_sources = ["x = 1", "y = 2"]
        triplets = build_triplets_from_blocks(
            blocks, negative_ratio=1.0, seed=42, all_sources=all_blocks_sources
        )
        for t in triplets:
            assert t["negative"] in ["x = -1"]  # only hard negatives

    def test_skip_blocks_without_positives(self):
        blocks = [
            self._make_block("x = 1", [], ["x = -1"]),  # no positives
        ]
        triplets = build_triplets_from_blocks(blocks, negative_ratio=0.5, seed=42)
        assert len(triplets) == 0

    def test_empty_blocks(self):
        triplets = build_triplets_from_blocks([], negative_ratio=0.5, seed=42)
        assert triplets == []


class TestTripletCodeDataset:
    def test_len(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
            {"anchor": "a = 1", "positive": "a = 2", "negative": "b = 3"},
        ]
        tokenizer = _FakeTokenizer()
        ds = TripletCodeDataset(triplets, tokenizer, max_length=64)
        assert len(ds) == 2

    def test_getitem_keys(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
        ]
        tokenizer = _FakeTokenizer()
        ds = TripletCodeDataset(triplets, tokenizer, max_length=64)
        item = ds[0]
        assert "anchor_input_ids" in item
        assert "anchor_attention_mask" in item
        assert "positive_input_ids" in item
        assert "positive_attention_mask" in item
        assert "negative_input_ids" in item
        assert "negative_attention_mask" in item

    def test_getitem_shapes(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
        ]
        tokenizer = _FakeTokenizer()
        max_len = 64
        ds = TripletCodeDataset(triplets, tokenizer, max_length=max_len)
        item = ds[0]
        assert item["anchor_input_ids"].shape[0] == max_len
        assert item["anchor_attention_mask"].shape[0] == max_len
