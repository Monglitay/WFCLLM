"""Adaptive metadata round-trip checks across detector alignment logic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from unittest.mock import MagicMock, patch

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.extract.config import BlockScore, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector


def _adaptive_metadata(code: str) -> dict:
    contracts = [asdict(contract) for contract in build_block_contracts(code)]
    return {
        "blocks": contracts,
        "adaptive_mode": "piecewise_quantile",
        "profile_id": "entropy-profile-v1",
    }


def test_adaptive_roundtrip_preserves_contracts():
    code = "x = 1\n"
    metadata = _adaptive_metadata(code)
    detector = WatermarkDetector(
        ExtractConfig(secret_key="test-key"),
        MagicMock(),
        MagicMock(),
        device="cpu",
    )
    scored_blocks = [
        BlockScore(
            block_id=metadata["blocks"][0]["block_id"],
            score=1,
            min_margin=0.1,
        )
    ]

    with patch.object(detector._scorer, "score_all", return_value=scored_blocks):
        result = detector.detect(code, watermark_metadata=metadata)

    assert result.mode == "adaptive"
    assert result.alignment_ok is True
    assert result.contract_valid is True


def test_tampered_adaptive_metadata_is_marked_invalid():
    code = "x = 1\n"
    metadata = _adaptive_metadata(code)
    tampered = deepcopy(metadata)
    tampered["blocks"][0]["entropy_units"] += 1

    detector = WatermarkDetector(
        ExtractConfig(secret_key="test-key"),
        MagicMock(),
        MagicMock(),
        device="cpu",
    )
    scored_blocks = [
        BlockScore(
            block_id=tampered["blocks"][0]["block_id"],
            score=1,
            min_margin=0.1,
        )
    ]

    with patch.object(detector._scorer, "score_all", return_value=scored_blocks):
        result = detector.detect(code, watermark_metadata=tampered)

    assert result.mode == "adaptive"
    assert result.alignment_ok is False
    assert result.contract_valid is False
    assert result.alignment_report is not None
    assert result.alignment_report.status == "numeric_mismatch"
