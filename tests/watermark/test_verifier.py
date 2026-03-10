"""Tests for wfcllm.watermark.verifier (LSH version)."""

from __future__ import annotations

import torch
import pytest
from unittest.mock import MagicMock

from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier, VerifyResult


def _make_lsh_space(d: int = 3, embed_dim: int = 128) -> LSHSpace:
    return LSHSpace(secret_key="test-secret", embed_dim=embed_dim, d=d)


def _make_encoder_returning(vec: torch.Tensor):
    """Return mock encoder that always outputs vec (shape (1, embed_dim))."""
    encoder = MagicMock()
    encoder.return_value = vec.unsqueeze(0)
    return encoder


def _make_tokenizer():
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(1, 10, dtype=torch.long),
        "attention_mask": torch.ones(1, 10, dtype=torch.long),
    }
    return tokenizer


class TestVerifyResult:
    def test_passed_true(self):
        r = VerifyResult(passed=True, min_margin=0.5, lsh_signature=(1, 0, 1))
        assert r.passed is True

    def test_passed_false(self):
        r = VerifyResult(passed=False, min_margin=0.05, lsh_signature=(0, 1, 0))
        assert r.passed is False


class TestProjectionVerifier:
    def test_verify_pass_when_sign_in_valid_set_and_margin_ok(self):
        """verify passes when sign ∈ valid_set and min_margin > margin."""
        lsh = _make_lsh_space(d=3)
        # Build a vector u, find its sign, put that sign in valid_set
        u = torch.randn(128)
        sig = lsh.sign(u)
        valid_set = frozenset([sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        # margin=0.0 to always pass the margin check
        result = verifier.verify("x = 1", valid_set, margin=0.0)
        assert result.passed is True

    def test_verify_fail_when_sign_not_in_valid_set(self):
        """verify fails when sign ∉ valid_set."""
        lsh = _make_lsh_space(d=3)
        u = torch.randn(128)
        sig = lsh.sign(u)
        # Flip one bit to get a different signature
        wrong_sig = tuple(1 - b for b in sig)
        valid_set = frozenset([wrong_sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        result = verifier.verify("x = 1", valid_set, margin=0.0)
        assert result.passed is False

    def test_verify_fail_when_margin_not_satisfied(self):
        """verify fails when min_margin <= margin threshold."""
        lsh = _make_lsh_space(d=3)
        u = torch.randn(128)
        sig = lsh.sign(u)
        valid_set = frozenset([sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        # margin=1.0 is impossible to satisfy
        result = verifier.verify("x = 1", valid_set, margin=1.0)
        assert result.passed is False

    def test_verify_result_has_min_margin_field(self):
        lsh = _make_lsh_space(d=3)
        u = torch.randn(128)
        sig = lsh.sign(u)
        valid_set = frozenset([sig])

        verifier = ProjectionVerifier(
            _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
        )
        result = verifier.verify("x = 1", valid_set, margin=0.0)
        assert isinstance(result.min_margin, float)
        assert 0.0 <= result.min_margin <= 1.0


def test_verify_result_contains_lsh_signature():
    """VerifyResult should expose the LSH signature used in the decision."""
    lsh = _make_lsh_space(d=3)
    u = torch.randn(128)
    sig = lsh.sign(u)
    valid_set = frozenset([sig])

    verifier = ProjectionVerifier(
        _make_encoder_returning(u), _make_tokenizer(), lsh_space=lsh, device="cpu"
    )
    result = verifier.verify("x = 1", valid_set, margin=0.0)

    assert hasattr(result, "lsh_signature")
    assert isinstance(result.lsh_signature, tuple)
    assert len(result.lsh_signature) == lsh._d  # same d used in construction
    assert all(b in (0, 1) for b in result.lsh_signature)
    assert result.lsh_signature == sig  # must match what lsh.sign(u) returns
