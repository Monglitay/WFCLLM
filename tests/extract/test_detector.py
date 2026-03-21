"""Tests for WatermarkDetector."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector


def _contract(*, entropy_units: int) -> dict:
    return {
        "ordinal": 0,
        "block_id": "0",
        "node_type": "expression_statement",
        "parent_node_type": "module",
        "block_text_hash": "hash-0",
        "start_line": 1,
        "end_line": 1,
        "entropy_units": entropy_units,
        "gamma_target": 0.0,
        "k": 0,
        "gamma_effective": 0.0,
    }


class TestWatermarkDetector:
    @pytest.fixture
    def config(self):
        return ExtractConfig(secret_key="test-key", embed_dim=128, fpr_threshold=3.0)

    @pytest.fixture
    def mock_encoder(self):
        encoder = MagicMock()
        vec = torch.randn(1, 128)
        vec = vec / vec.norm()
        encoder.return_value = vec
        encoder.eval = MagicMock(return_value=encoder)
        return encoder

    @pytest.fixture
    def mock_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "attention_mask": torch.ones(1, 10, dtype=torch.long),
        }
        return tokenizer

    def test_detect_returns_detection_result(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        # Use compound+simple input to verify compound block is excluded
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)
        assert isinstance(result, DetectionResult)
        # Only simple blocks counted (x = i + 1, y = i * 2), not the for compound block
        assert result.total_blocks == 2
        assert result.independent_blocks == result.total_blocks  # all simple blocks selected
        assert isinstance(result.z_score, float)
        assert isinstance(result.p_value, float)

    def test_detect_empty_code(self, config, mock_encoder, mock_tokenizer):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        result = detector.detect("")

        assert result.is_watermarked is False
        assert result.total_blocks == 0
        assert result.independent_blocks == 0

    def test_block_details_include_simple_blocks(
        self, config, mock_encoder, mock_tokenizer
    ):
        """block_details should contain only simple blocks, not compound blocks."""
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)
        assert len(result.block_details) == result.total_blocks
        # total_blocks should be 2 (only the simple blocks), not 3
        assert result.total_blocks == 2

    def test_selected_flag_set(self, config, mock_encoder, mock_tokenizer):
        """All simple blocks should have selected=True."""
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
        result = detector.detect(code)
        # All simple blocks are selected (no DP filtering)
        assert all(s.selected for s in result.block_details)
        assert result.independent_blocks == result.total_blocks

    def test_detect_surfaces_contract_alignment_when_metadata_present(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "x = 1\n"
        embedded_contract = _contract(entropy_units=101)

        with patch(
            "wfcllm.extract.detector.rebuild_block_contracts",
            return_value=[_contract(entropy_units=100)],
        ):
            result = detector.detect(
                code,
                watermark_metadata={
                    "blocks": [embedded_contract],
                    "adaptive_mode": "piecewise",
                    "profile_id": "entropy-profile-v1",
                },
            )

        assert result.alignment_report is not None
        assert result.alignment_report.structure_mismatch is False
        assert result.alignment_report.numeric_mismatch is True
        assert result.alignment_report.status == "numeric_mismatch"
        assert result.contract_valid is False

    def test_detect_without_metadata_keeps_legacy_path_unchanged(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        with patch("wfcllm.extract.detector.compare_block_contracts") as compare_contracts:
            result = detector.detect("x = 1\n")

        compare_contracts.assert_not_called()
        assert result.alignment_report is None
        assert result.contract_valid is None

    def test_detect_compares_fixed_mode_metadata_when_blocks_present(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "x = 1\n"
        embedded_contract = _contract(entropy_units=100)

        with patch(
            "wfcllm.extract.detector.rebuild_block_contracts",
            return_value=[_contract(entropy_units=100)],
        ), patch(
            "wfcllm.extract.detector.compare_block_contracts",
            wraps=__import__("wfcllm.extract.alignment", fromlist=["compare_block_contracts"]).compare_block_contracts,
        ) as compare_contracts:
            result = detector.detect(
                code,
                watermark_metadata={
                    "blocks": [embedded_contract],
                    "adaptive_mode": "fixed",
                    "profile_id": None,
                },
            )

        compare_contracts.assert_called_once()
        assert result.alignment_report is not None
        assert result.alignment_report.status == "aligned"
        assert result.contract_valid is True

    def test_detect_wires_adaptive_mode_and_block_gamma_from_metadata(
        self, config, mock_encoder, mock_tokenizer
    ):
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        code = "x = 1\ny = 2\n"
        scored_blocks = [
            BlockScore(block_id="0", score=1, min_margin=0.5, gamma_effective=0.5),
            BlockScore(block_id="1", score=0, min_margin=0.1, gamma_effective=0.5),
        ]

        with patch.object(detector._scorer, "score_all", return_value=scored_blocks):
            result = detector.detect(
                code,
                watermark_metadata={
                    "blocks": [
                        {**_contract(entropy_units=100), "gamma_effective": 0.2},
                        {
                            **_contract(entropy_units=100),
                            "ordinal": 1,
                            "block_id": "1",
                            "block_text_hash": "hash-1",
                            "start_line": 2,
                            "end_line": 2,
                            "gamma_effective": 0.8,
                        },
                    ],
                    "adaptive_mode": "piecewise",
                    "profile_id": "entropy-profile-v1",
                },
            )

        assert result.hypothesis_mode == "adaptive"
        assert result.expected_hits == pytest.approx(1.0)
        assert result.variance == pytest.approx((0.2 * 0.8) + (0.8 * 0.2))
        assert [score.gamma_effective for score in result.block_details] == pytest.approx([0.2, 0.8])

    def test_detect_scores_with_block_specific_k_from_metadata(
        self, config, mock_encoder, mock_tokenizer
    ):
        config.adaptive_detection.require_block_contract_check = False
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
        detector._scorer._keying.derive = MagicMock(return_value=frozenset())
        detector._scorer._verifier.verify = MagicMock(
            return_value=MagicMock(passed=True, min_margin=0.1)
        )

        code = "x = 1\ny = 2\n"
        contracts = [build_block_contracts(code)[0], build_block_contracts(code)[1]]
        metadata = {
            "blocks": [
                {
                    **contracts[0].__dict__,
                    "k": 3,
                    "gamma_effective": 0.1875,
                },
                {
                    **contracts[1].__dict__,
                    "k": 9,
                    "gamma_effective": 0.5625,
                },
            ],
            "adaptive_mode": "piecewise_quantile",
        }

        detector.detect(code, watermark_metadata=metadata)

        derive_calls = detector._scorer._keying.derive.call_args_list
        assert [call.kwargs["k"] for call in derive_calls] == [3, 9]
