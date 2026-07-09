"""Tests for LSHSpace."""
from __future__ import annotations

import torch
import pytest
from wfcllm.watermark.lsh_space import LSHSpace


class TestLSHSpace:
    @pytest.fixture
    def space(self):
        return LSHSpace(secret_key="test-secret", embed_dim=128, d=3)

    def test_hyperplanes_shape(self, space):
        """planes tensor should be (d, embed_dim)."""
        assert space._planes.shape == (3, 128)

    def test_hyperplanes_are_unit_normalized(self, space):
        """Each hyperplane normal vector should be L2-normalized."""
        norms = torch.norm(space._planes, dim=1)
        assert torch.allclose(norms, torch.ones(3), atol=1e-5)

    def test_sign_returns_tuple_of_length_d(self, space):
        u = torch.randn(128)
        sig = space.sign(u)
        assert isinstance(sig, tuple)
        assert len(sig) == 3
        assert all(b in (0, 1) for b in sig)

    def test_sign_deterministic(self, space):
        u = torch.randn(128)
        assert space.sign(u) == space.sign(u)

    def test_sign_opposite_vector_flips_all_bits(self, space):
        u = torch.randn(128)
        sig_u = space.sign(u)
        sig_neg = space.sign(-u)
        assert all(a != b for a, b in zip(sig_u, sig_neg))

    def test_min_margin_returns_float(self, space):
        u = torch.randn(128)
        m = space.min_margin(u)
        assert isinstance(m, float)
        assert 0.0 <= m <= 1.0

    def test_min_margin_is_min_of_abs_cosines(self, space):
        """min_margin should equal the smallest |cos(u, n_i)| across all planes."""
        import torch.nn.functional as F
        u = torch.randn(128)
        u_norm = F.normalize(u.unsqueeze(0), dim=1).squeeze(0)
        expected = min(
            abs(F.cosine_similarity(u_norm.unsqueeze(0), space._planes[i].unsqueeze(0)).item())
            for i in range(3)
        )
        assert abs(space.min_margin(u) - expected) < 1e-5

    def test_same_key_same_planes(self):
        """Same key and d -> identical hyperplanes."""
        s1 = LSHSpace("key", 64, 2)
        s2 = LSHSpace("key", 64, 2)
        assert torch.allclose(s1._planes, s2._planes)

    def test_different_key_different_planes(self):
        s1 = LSHSpace("key-a", 64, 2)
        s2 = LSHSpace("key-b", 64, 2)
        assert not torch.allclose(s1._planes, s2._planes)
