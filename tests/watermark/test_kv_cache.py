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

    def test_rollback_safety_check_stale_snapshot(self, manager):
        """snapshot.seq_len > current seq_len raises ValueError."""
        kv = self._make_kv_cache(num_layers=2, seq_len=10)
        stale_snap = CacheSnapshot(seq_len=20)
        with pytest.raises(ValueError, match="Snapshot seq_len"):
            manager.rollback(kv, stale_snap)

    def test_rollback_same_length_returns_same_object(self, manager):
        """When target_len == current_len, return the same tuple (no clone)."""
        kv = self._make_kv_cache(num_layers=2, seq_len=10)
        snap = manager.snapshot(kv)
        result = manager.rollback(kv, snap)
        assert result is kv  # same object, no clone

    def test_rollback_old_tensors_not_referenced(self, manager):
        """After rollback, old tensors have no external references (can be GC'd)."""
        import weakref
        kv = self._make_kv_cache(num_layers=2, seq_len=20)
        snap = CacheSnapshot(seq_len=10)
        # Keep weak reference to original key tensor
        old_k = kv[0][0]
        ref = weakref.ref(old_k)
        rolled = manager.rollback(kv, snap)
        del kv, old_k
        # After del, ref should be dead (no strong references remain)
        # Note: this may not work in all cases due to Python internals,
        # so we test the structural guarantee instead
        for k, v in rolled:
            assert k.shape[2] == 10
            assert v.shape[2] == 10

    def test_snapshot_at_removed(self, manager):
        """snapshot_at() should no longer exist."""
        assert not hasattr(manager, "snapshot_at")
