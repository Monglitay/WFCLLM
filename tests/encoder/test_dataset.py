"""Tests for wfcllm.encoder.dataset."""

import pytest
from unittest.mock import patch, MagicMock
from wfcllm.encoder.dataset import TripletCodeDataset, build_triplets_from_blocks


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
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        ds = TripletCodeDataset(triplets, tokenizer, max_length=64)
        assert len(ds) == 2

    def test_getitem_keys(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
        ]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
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
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        max_len = 64
        ds = TripletCodeDataset(triplets, tokenizer, max_length=max_len)
        item = ds[0]
        assert item["anchor_input_ids"].shape[0] == max_len
        assert item["anchor_attention_mask"].shape[0] == max_len
