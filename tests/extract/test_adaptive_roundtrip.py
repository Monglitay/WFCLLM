"""Adaptive metadata round-trip checks across detector alignment logic."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import json
from unittest.mock import MagicMock, patch

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.extract.config import BlockScore, ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.watermark.config import AdaptiveGammaConfig
from wfcllm.watermark.entropy_profile import EntropyProfile
from wfcllm.watermark.gamma_schedule import PiecewiseQuantileSchedule


def _adaptive_metadata(code: str, tmp_path) -> tuple[dict, AdaptiveGammaConfig]:
    entropy_units = build_block_contracts(code)[0].entropy_units
    profile_payload = {
        "language": "python",
        "model_family": "demo-model",
        "quantiles_units": {
            "p10": max(0, entropy_units - 2),
            "p50": max(0, entropy_units - 1),
            "p75": entropy_units,
            "p90": entropy_units + 1,
            "p95": entropy_units + 2,
        },
        "strategy": "piecewise_quantile",
    }
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(profile_payload), encoding="utf-8")

    anchors = {
        "p10": 0.95,
        "p50": 0.75,
        "p75": 0.55,
        "p90": 0.35,
        "p95": 0.25,
    }
    schedule = PiecewiseQuantileSchedule(
        profile=EntropyProfile.load(profile_path),
        anchor_quantiles=tuple(anchors.keys()),
        anchor_gammas=tuple(anchors.values()),
    )
    contracts = [
        asdict(contract)
        for contract in build_block_contracts(
            code,
            gamma_resolver=lambda units: schedule.resolve(units, 4),
        )
    ]
    metadata = {
        "blocks": contracts,
        "adaptive_mode": "piecewise_quantile",
        "profile_id": "entropy-profile-v1",
        "watermark_params": {
            "lsh_d": 4,
            "adaptive_gamma": {
                "strategy": "piecewise_quantile",
                "profile_id": "entropy-profile-v1",
                "anchors": anchors,
                "profile": profile_payload,
            },
        },
    }
    adaptive_config = AdaptiveGammaConfig(
        enabled=True,
        strategy="piecewise_quantile",
        profile_path=str(profile_path),
        profile_id="entropy-profile-v1",
        anchors=anchors,
    )
    return metadata, adaptive_config


def test_adaptive_roundtrip_preserves_contracts(tmp_path):
    code = "x = 1\n"
    metadata, _ = _adaptive_metadata(code, tmp_path)
    detector = WatermarkDetector(
        ExtractConfig(secret_key="test-key", lsh_d=4),
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


def test_saved_historical_best_anchor_region_roundtrips_without_numeric_mismatch(tmp_path):
    code = "x = 1\n"
    historical_anchors = {
        "p10": 0.75,
        "p50": 0.75,
        "p75": 0.50,
        "p90": 0.50,
        "p95": 0.25,
    }
    metadata, _ = _adaptive_metadata(code, tmp_path)
    profile_payload = metadata["watermark_params"]["adaptive_gamma"]["profile"]
    schedule = PiecewiseQuantileSchedule(
        profile=EntropyProfile(
            language=profile_payload["language"],
            model_family=profile_payload["model_family"],
            quantiles_units_map=profile_payload["quantiles_units"],
            strategy=profile_payload["strategy"],
        ),
        anchor_quantiles=tuple(historical_anchors.keys()),
        anchor_gammas=tuple(historical_anchors.values()),
    )
    metadata["watermark_params"]["lsh_d"] = 4
    metadata["watermark_params"]["adaptive_gamma"]["anchors"] = historical_anchors
    metadata["blocks"] = [
        asdict(contract)
        for contract in build_block_contracts(
            code,
            gamma_resolver=lambda units: schedule.resolve(units, 4),
        )
    ]

    detector = WatermarkDetector(
        ExtractConfig(secret_key="test-key", lsh_d=4),
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


def test_tampered_adaptive_metadata_is_marked_invalid(tmp_path):
    code = "x = 1\n"
    metadata, _ = _adaptive_metadata(code, tmp_path)
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


def test_adaptive_roundtrip_uses_extract_config_when_schedule_metadata_missing(tmp_path):
    code = "x = 1\n"
    metadata, adaptive_config = _adaptive_metadata(code, tmp_path)
    metadata["watermark_params"] = {"lsh_d": 4}

    config = ExtractConfig(secret_key="test-key", lsh_d=4)
    config.adaptive_gamma = adaptive_config
    detector = WatermarkDetector(
        config,
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

    assert result.alignment_ok is True
    assert result.contract_valid is True
