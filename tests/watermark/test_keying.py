"""Tests for wfcllm.watermark.keying (LSH version)."""

from __future__ import annotations

import pytest
from wfcllm.watermark.keying import WatermarkKeying


class TestWatermarkKeying:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-secret", d=3, gamma=0.5)

    def test_derive_returns_frozenset(self, keying):
        G = keying.derive("module")
        assert isinstance(G, frozenset)

    def test_derive_set_size_matches_gamma(self, keying):
        """With d=3 and gamma=0.5, G should have round(0.5 * 8) = 4 elements."""
        G = keying.derive("module")
        assert len(G) == 4

    def test_derive_elements_are_d_tuples(self, keying):
        G = keying.derive("module")
        for sig in G:
            assert isinstance(sig, tuple)
            assert len(sig) == 3
            assert all(b in (0, 1) for b in sig)

    def test_derive_deterministic(self, keying):
        G1 = keying.derive("module")
        G2 = keying.derive("module")
        assert G1 == G2

    def test_different_parent_different_G(self, keying):
        G1 = keying.derive("module")
        G2 = keying.derive("for_statement")
        assert G1 != G2

    def test_different_key_different_G(self):
        k1 = WatermarkKeying(secret_key="key-a", d=3, gamma=0.5)
        k2 = WatermarkKeying(secret_key="key-b", d=3, gamma=0.5)
        G1 = k1.derive("module")
        G2 = k2.derive("module")
        assert G1 != G2

    def test_gamma_controls_set_size(self):
        k_25 = WatermarkKeying(secret_key="k", d=3, gamma=0.25)
        k_75 = WatermarkKeying(secret_key="k", d=3, gamma=0.75)
        assert len(k_25.derive("module")) == 2  # round(0.25 * 8)
        assert len(k_75.derive("module")) == 6  # round(0.75 * 8)

    def test_G_is_subset_of_all_signatures(self, keying):
        """G must only contain valid d-bit signatures."""
        all_sigs = {
            tuple(int(b) for b in format(i, f"0{3}b"))
            for i in range(2 ** 3)
        }
        G = keying.derive("module")
        assert G.issubset(all_sigs)
