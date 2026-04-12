"""Tests for WatermarkDetector."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.extract.config import BlockScore, DetectionResult, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.extract.hypothesis import JointDetectionResult
from wfcllm.extract.hypothesis import LexicalDetectionResult


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

    def test_detection_result_exposes_typed_channel_fields_by_default(self):
        result = DetectionResult(
            is_watermarked=False,
            z_score=0.0,
            p_value=1.0,
            total_blocks=0,
            independent_blocks=0,
            hit_blocks=0,
        )

        assert result.semantic_result is None
        assert result.lexical_result is None
        assert result.joint_result is None

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

    def test_detect_attaches_lexical_and_joint_results(self, config, mock_encoder, mock_tokenizer):
        config.token_channel.enabled = True
        config.token_channel.mode = "dual-channel"
        config.token_channel.joint.lexical_full_weight_min_positions = 4
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        lexical_result = LexicalDetectionResult(
            num_positions_scored=4,
            num_green_hits=3,
            green_fraction=0.75,
            lexical_z_score=2.0,
            lexical_p_value=0.1,
        )

        with patch.object(detector, "_detect_lexical", return_value=lexical_result):
            result = detector.detect("x = 1\n")

        assert result.lexical_result is lexical_result
        assert result.joint_result.joint_score == pytest.approx(result.z_score + (0.75 * 2.0))
        assert result.semantic_result.z_score == result.z_score

    def test_detect_uses_lexical_only_mode_without_semantic_scores(
        self, config, mock_encoder, mock_tokenizer
    ):
        config.token_channel.enabled = True
        config.token_channel.mode = "lexical-only"
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        lexical_result = LexicalDetectionResult(
            num_positions_scored=10,
            num_green_hits=8,
            green_fraction=0.8,
            lexical_z_score=4.5,
            lexical_p_value=0.001,
        )

        with patch.object(detector, "_detect_lexical", return_value=lexical_result), patch.object(
            detector._scorer,
            "score_all",
        ) as score_all:
            result = detector.detect("x = 1\n")

        score_all.assert_not_called()
        assert result.semantic_result is None
        assert result.lexical_result is lexical_result
        assert result.joint_result == lexical_result.to_joint_equivalent(
            threshold=config.token_channel.joint_threshold
        )
        assert result.is_watermarked is True
        assert result.z_score == pytest.approx(4.5)

    def test_detect_rejects_token_altering_postprocess_metadata(
        self, config, mock_encoder, mock_tokenizer
    ):
        config.token_channel.enabled = True
        config.token_channel.mode = "dual-channel"
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        with pytest.raises(ValueError, match="tokenizer-visible final code"):
            detector.detect(
                "x = 1\n",
                watermark_metadata={
                    "token_channel": {
                        "enabled": True,
                        "token_altering_postprocess": True,
                    }
                },
            )

    def test_detect_rejects_mismatched_pinned_token_channel_artifact(
        self,
        tmp_path,
        config,
        mock_encoder,
        mock_tokenizer,
    ):
        config.token_channel.enabled = True
        config.token_channel.mode = "dual-channel"
        config.token_channel.context_width = 64
        config.token_channel.ignore_repeated_ngrams = True
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        model_path = tmp_path / "model.pt"
        metadata_path = tmp_path / "metadata.json"
        model_path.write_text("current-model", encoding="utf-8")
        metadata_path.write_text('{"schema_version": "token-channel/v1"}', encoding="utf-8")
        artifact = SimpleNamespace(
            model=object(),
            metadata=SimpleNamespace(
                schema_version="token-channel/v1",
                tokenizer_name="demo-tokenizer",
                tokenizer_vocab_size=8,
                context_width=64,
                feature_version="token-channel-features/v1",
                training_config={"dropout": 0.1},
            ),
            model_path=Path(model_path),
            metadata_path=Path(metadata_path),
        )

        with patch("wfcllm.extract.detector.load_token_channel_artifact", return_value=artifact):
            with pytest.raises(ValueError, match="token-channel artifact/config mismatch"):
                detector.detect(
                    "x = 1\n",
                    watermark_metadata={
                        "token_channel": {
                            "enabled": True,
                            "mode": "dual-channel",
                            "context_width": 64,
                            "switch_threshold": 0.0,
                            "delta": 2.0,
                            "ignore_repeated_ngrams": False,
                            "ignore_repeated_prefixes": False,
                            "token_altering_postprocess": False,
                            "artifact_metadata": {
                                "schema_version": "token-channel/v1",
                                "tokenizer_name": "demo-tokenizer",
                                "tokenizer_vocab_size": 8,
                                "context_width": 64,
                                "feature_version": "token-channel-features/v1",
                                "training_config": {"dropout": 0.1},
                            },
                            "artifact_fingerprints": {
                                "model_sha256": "different-model",
                                "metadata_sha256": "different-metadata",
                            },
                        }
                    },
                )

    def test_detect_allows_token_altering_postprocess_in_semantic_only_mode(
        self, config, mock_encoder, mock_tokenizer
    ):
        config.token_channel.enabled = True
        config.token_channel.mode = "semantic-only"
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        result = detector.detect(
            "x = 1\n",
            watermark_metadata={
                "token_channel": {
                    "enabled": True,
                    "token_altering_postprocess": True,
                }
            },
        )

        assert isinstance(result, DetectionResult)
        assert result.semantic_result is not None
        assert result.lexical_result is None
        assert result.joint_result is None

    def test_detect_lazily_initializes_real_lexical_detector_path(
        self, config, mock_encoder, mock_tokenizer
    ):
        config.token_channel.enabled = True
        config.token_channel.mode = "dual-channel"
        detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")

        lexical_result = LexicalDetectionResult(
            num_positions_scored=5,
            num_green_hits=4,
            green_fraction=0.8,
            lexical_z_score=2.5,
            lexical_p_value=0.05,
        )
        artifact = MagicMock(model=object(), metadata=object())
        replay_instance = MagicMock()
        replay_instance.detect.return_value = lexical_result

        with patch.object(detector._scorer, "score_all", return_value=[]), patch(
            "wfcllm.extract.detector.load_token_channel_artifact",
            return_value=artifact,
        ) as load_artifact, patch(
            "wfcllm.extract.detector.TokenChannelRuntime",
            return_value="runtime",
        ) as runtime_cls, patch(
            "wfcllm.extract.detector.ReplayTokenChannelDetector",
            return_value=replay_instance,
        ) as replay_cls:
            first = detector.detect("x = 1\n")
            second = detector.detect("x = 1\n")

        load_artifact.assert_called_once_with(config.token_channel.model_path)
        runtime_cls.assert_called_once_with(
            model=artifact.model,
            config=config.token_channel,
            artifact_metadata=artifact.metadata,
            tokenizer=mock_tokenizer,
            secret_key=config.secret_key,
        )
        replay_cls.assert_called_once_with(
            runtime="runtime",
            tokenizer=mock_tokenizer,
            config=config.token_channel,
        )
        assert replay_instance.detect.call_count == 2
        assert detector._lexical_detector is replay_instance
        assert first.lexical_result is lexical_result
        assert second.lexical_result is lexical_result
