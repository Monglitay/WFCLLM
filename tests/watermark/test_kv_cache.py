"""Tests for wfcllm.watermark.kv_cache."""

import torch
import pytest
from wfcllm.watermark.kv_cache import KVCacheManager, CacheSnapshot


class TestCacheSnapshot:
    def test_snapshot_stores_seq_len(self):
        snap = CacheSnapshot(seq_len=42)
        assert snap.seq_len == 42


class TestKVCacheManager:
    @pytest.fixture
    def manager(self):
        return KVCacheManager()

    def _make_kv_cache(self, num_layers: int, seq_len: int) -> tuple:
        """Create a mock past_key_values structure."""
        batch, heads, head_dim = 1, 4, 32
        return tuple(
            (
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim),
            )
            for _ in range(num_layers)
        )

    def test_snapshot_records_seq_len(self, manager):
        kv = self._make_kv_cache(num_layers=2, seq_len=50)
        snap = manager.snapshot(kv)
        assert snap.seq_len == 50

    def test_rollback_truncates(self, manager):
        # Create cache with 50 tokens, snapshot at 30, grow to 50, rollback
        kv_30 = self._make_kv_cache(num_layers=2, seq_len=30)
        snap = manager.snapshot(kv_30)

        kv_50 = self._make_kv_cache(num_layers=2, seq_len=50)
        rolled = manager.rollback(kv_50, snap)

        for k, v in rolled:
            assert k.shape[2] == 30
            assert v.shape[2] == 30

    def test_rollback_preserves_values(self, manager):
        kv = self._make_kv_cache(num_layers=2, seq_len=50)
        snap = manager.snapshot(kv)

        # "Grow" the cache by extending (simulate more tokens generated)
        extended = tuple(
            (
                torch.cat([k, torch.randn_like(k[:, :, :10, :])], dim=2),
                torch.cat([v, torch.randn_like(v[:, :, :10, :])], dim=2),
            )
            for k, v in kv
        )
        # extended has seq_len=60, snap was at 50
        rolled = manager.rollback(extended, snap)
        for (orig_k, orig_v), (roll_k, roll_v) in zip(kv, rolled):
            assert torch.allclose(orig_k, roll_k)
            assert torch.allclose(orig_v, roll_v)

    def test_rollback_structure_matches(self, manager):
        num_layers = 4
        kv = self._make_kv_cache(num_layers=num_layers, seq_len=20)
        snap = manager.snapshot(kv)
        rolled = manager.rollback(kv, snap)
        assert len(rolled) == num_layers
        assert all(len(layer) == 2 for layer in rolled)

    def test_snapshot_different_sizes(self, manager):
        kv10 = self._make_kv_cache(num_layers=2, seq_len=10)
        kv100 = self._make_kv_cache(num_layers=2, seq_len=100)
        snap10 = manager.snapshot(kv10)
        snap100 = manager.snapshot(kv100)
        assert snap10.seq_len == 10
        assert snap100.seq_len == 100

    def test_snapshot_at_computes_correct_seq_len(self, manager):
        """snapshot_at 应计算出语句块开始前的 seq_len。"""
        # 模拟：prompt=10 tokens，生成了 20 个 token，共 seq_len=30
        kv = self._make_kv_cache(num_layers=2, seq_len=30)
        # 语句块占最后 5 个生成 token；rollback_idx = 20 - 5 = 15
        snap = manager.snapshot_at(past_key_values=kv, rollback_idx=15, current_generated_count=20)
        # 期望 seq_len = 30 - (20 - 15) = 25
        assert snap.seq_len == 25

    def test_snapshot_at_zero_block_tokens(self, manager):
        """rollback_idx == current_generated_count 时，seq_len 不变。"""
        kv = self._make_kv_cache(num_layers=2, seq_len=30)
        snap = manager.snapshot_at(past_key_values=kv, rollback_idx=20, current_generated_count=20)
        assert snap.seq_len == 30

    def test_snapshot_at_clamps_to_zero(self, manager):
        """block_token_count 超过 current_seq_len 时，seq_len 不小于 0。"""
        kv = self._make_kv_cache(num_layers=2, seq_len=5)
        snap = manager.snapshot_at(past_key_values=kv, rollback_idx=0, current_generated_count=100)
        assert snap.seq_len == 0
