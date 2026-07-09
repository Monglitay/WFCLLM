"""Tests for wfcllm.watermark.keying (LSH version)."""

from __future__ import annotations

import pytest
from wfcllm.watermark.keying import WatermarkKeying


class TestWatermarkKeying:
    @pytest.fixture
    def keying(self):
        return WatermarkKeying(secret_key="test-secret", d=3)

    def test_derive_returns_frozenset(self, keying):
        G = keying.derive("module", 3)
        assert isinstance(G, frozenset)

    def test_derive_set_size_matches_k(self, keying):
        G = keying.derive("module", 3)
        assert len(G) == 3

    def test_derive_elements_are_d_tuples(self, keying):
        G = keying.derive("module", 3)
        for sig in G:
            assert isinstance(sig, tuple)
            assert len(sig) == 3
            assert all(b in (0, 1) for b in sig)

    def test_derive_deterministic(self, keying):
        G1 = keying.derive("module", 3)
        G2 = keying.derive("module", 3)
        assert G1 == G2

    def test_different_parent_different_G(self, keying):
        G1 = keying.derive("module", 3)
        G2 = keying.derive("for_statement", 3)
        assert G1 != G2

    def test_different_key_different_G(self):
        k1 = WatermarkKeying(secret_key="key-a", d=3)
        k2 = WatermarkKeying(secret_key="key-b", d=3)
        G1 = k1.derive("module", 3)
        G2 = k2.derive("module", 3)
        assert G1 != G2

    @pytest.mark.parametrize("k", [1, 2, 7])
    def test_derive_supports_valid_k_range(self, keying, k):
        G = keying.derive("module", k)
        assert len(G) == k

    def test_G_is_subset_of_all_signatures(self, keying):
        """G must only contain valid d-bit signatures."""
        all_sigs = {
            tuple(int(b) for b in format(i, f"0{3}b"))
            for i in range(2 ** 3)
        }
        G = keying.derive("module", 3)
        assert G.issubset(all_sigs)

    @pytest.mark.parametrize("k", [0, -1, 8, 9])
    def test_derive_rejects_invalid_k(self, keying, k):
        with pytest.raises(ValueError, match="1 <= k < 2\\*\\*d"):
            keying.derive("module", k)

    def test_legacy_constructor_and_derive_without_k(self):
        keying = WatermarkKeying(secret_key="legacy", d=3, gamma=0.5)
        G = keying.derive("module")
        assert isinstance(G, frozenset)
        assert len(G) == 4

    @pytest.mark.parametrize("k", [1.5, "2", None, True, False])
    def test_derive_rejects_non_int_or_bool_k(self, keying, k):
        with pytest.raises(TypeError, match="k must be an int"):
            keying.derive("module", k)  # type: ignore[arg-type]
