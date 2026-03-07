"""Tests for wfcllm.watermark.verifier."""

import torch
import pytest
from unittest.mock import MagicMock
from wfcllm.watermark.verifier import ProjectionVerifier, VerifyResult


class TestVerifyResult:
    def test_passed_true(self):
        r = VerifyResult(passed=True, projection=0.5, target_sign=1, margin=0.1)
        assert r.passed is True

    def test_passed_false(self):
        r = VerifyResult(passed=False, projection=-0.05, target_sign=1, margin=0.1)
        assert r.passed is False


class TestProjectionVerifier:
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder that returns a fixed vector."""
        encoder = MagicMock()
        # Return a normalized vector pointing in positive direction
        fixed_vec = torch.randn(1, 128)
        fixed_vec = fixed_vec / fixed_vec.norm()
        encoder.return_value = fixed_vec
        encoder.eval = MagicMock(return_value=encoder)
        encoder.config = MagicMock()
        encoder.config.model_name = "Salesforce/codet5-base"
        return encoder, fixed_vec.squeeze(0)

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        return tokenizer

    def test_verify_pass_positive_projection(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Direction = same as encoder output -> cos ~ 1.0
        v = fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=1, margin=0.1)
        assert result.passed is True
        assert result.projection > 0

    def test_verify_pass_negative_projection(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Direction = negated -> cos ~ -1.0, target t=0 -> t*=-1
        v = -fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=0, margin=0.1)
        # cos(u, -u) = -1.0, sign = -1, t* = -1 -> match
        assert result.passed is True

    def test_verify_fail_wrong_sign(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Direction same as output -> cos > 0, but target t=0 -> t*=-1
        v = fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=0, margin=0.1)
        assert result.passed is False

    def test_verify_fail_below_margin(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        # Use nearly orthogonal direction -> small |cos|
        v = torch.zeros(128)
        v[0] = 1.0  # Arbitrary direction likely not aligned
        # With random encoder output, projection could be small
        result = verifier.verify("x = 1", v, t=1, margin=0.99)
        # Margin 0.99 is very strict — almost certainly fails
        assert result.passed is False

    def test_verify_result_contains_values(self, mock_encoder, mock_tokenizer):
        encoder, fixed_vec = mock_encoder
        verifier = ProjectionVerifier(encoder, mock_tokenizer, device="cpu")
        v = fixed_vec.clone()
        result = verifier.verify("x = 1", v, t=1, margin=0.1)
        assert isinstance(result.projection, float)
        assert result.target_sign in (-1, 1)
        assert isinstance(result.margin, float)
