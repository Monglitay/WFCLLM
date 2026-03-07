"""Tests for wfcllm.watermark.keying."""

import torch
import pytest
from wfcllm.watermark.keying import WatermarkKeying


class TestWatermarkKeying:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-secret", embed_dim=128)

    def test_derive_returns_vector_and_bit(self, keying):
        v, t = keying.derive("module", "expression_statement")
        assert isinstance(v, torch.Tensor)
        assert v.shape == (128,)
        assert t in (0, 1)

    def test_vector_is_unit_normalized(self, keying):
        v, _ = keying.derive("module", "assignment")
        norm = torch.norm(v).item()
        assert abs(norm - 1.0) < 1e-5

    def test_deterministic(self, keying):
        v1, t1 = keying.derive("if_statement", "return_statement")
        v2, t2 = keying.derive("if_statement", "return_statement")
        assert torch.allclose(v1, v2)
        assert t1 == t2

    def test_different_topology_different_output(self, keying):
        v1, _ = keying.derive("module", "expression_statement")
        v2, _ = keying.derive("for_statement", "expression_statement")
        assert not torch.allclose(v1, v2)

    def test_different_key_different_output(self):
        k1 = WatermarkKeying(secret_key="key-a", embed_dim=128)
        k2 = WatermarkKeying(secret_key="key-b", embed_dim=128)
        v1, _ = k1.derive("module", "assignment")
        v2, _ = k2.derive("module", "assignment")
        assert not torch.allclose(v1, v2)

    def test_different_embed_dim(self):
        k64 = WatermarkKeying(secret_key="k", embed_dim=64)
        v, _ = k64.derive("module", "assignment")
        assert v.shape == (64,)

    def test_target_bit_distribution(self, keying):
        """Over many different inputs, t should be roughly 50/50."""
        bits = []
        for i in range(100):
            _, t = keying.derive(f"type_{i}", "expression_statement")
            bits.append(t)
        ratio = sum(bits) / len(bits)
        # Should be roughly balanced — allow 30-70% range
        assert 0.3 < ratio < 0.7
