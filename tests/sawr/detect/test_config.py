from __future__ import annotations

import json
from decimal import Decimal
from fractions import Fraction

import pytest

import wfcllm.sawr as sawr
import wfcllm.sawr.detect as detect
from wfcllm.sawr.detect.config import (
    DETECTOR_MODE,
    BucketEdges,
    SawrDetectionConfig,
    bucket_label,
)


def test_detection_config_quantizes_gamma_to_k() -> None:
    config = SawrDetectionConfig(secret_key="1010", lsh_d=4, gamma=0.75)

    assert config.k == 12
    assert config.gamma_effective == pytest.approx(0.75)
    assert config.detector_mode == DETECTOR_MODE


def test_detection_config_exports_secret_free_metadata() -> None:
    config = SawrDetectionConfig(secret_key="1010", lsh_d=4, gamma=0.75)
    payload = config.to_public_dict()

    assert payload["secret_key_sha256"]
    assert payload["secret_key_sha256"] != "1010"
    assert "secret_key" not in payload
    json.dumps(payload, allow_nan=False)


def test_detection_config_public_metadata_accepts_json_native_numbers() -> None:
    config = SawrDetectionConfig(
        secret_key="1010",
        lsh_d=4,
        gamma=1,
        semantic_margin=0.0,
        target_fpr=0.5,
    )

    json.dumps(config.to_public_dict(), allow_nan=False)


def test_detection_config_exports_proxy_penalty_alpha() -> None:
    config = SawrDetectionConfig(
        secret_key="1010",
        statistic="calibrated_context_mean_proxy_penalized",
        proxy_penalty_alpha=0.4,
    )

    payload = config.to_public_dict()

    assert payload["statistic"] == "calibrated_context_mean_proxy_penalized"
    assert payload["proxy_penalty_alpha"] == pytest.approx(0.4)
    json.dumps(payload, allow_nan=False)


def test_detection_config_exports_code_length_adjustment() -> None:
    config = SawrDetectionConfig(
        secret_key="1010",
        statistic="calibrated_context_mean_proxy_length_adjusted",
        proxy_penalty_alpha=0.34,
        code_length_adjustment_beta=0.3,
        code_length_reference_chars=700,
    )

    payload = config.to_public_dict()

    assert payload["statistic"] == "calibrated_context_mean_proxy_length_adjusted"
    assert payload["proxy_penalty_alpha"] == pytest.approx(0.34)
    assert payload["code_length_adjustment_beta"] == pytest.approx(0.3)
    assert payload["code_length_reference_chars"] == 700
    json.dumps(payload, allow_nan=False)


def test_bucket_label_uses_frozen_edges() -> None:
    assert bucket_label(0, (1, 2, 4, 8)) == "0"
    assert bucket_label(1, (1, 2, 4, 8)) == "1"
    assert bucket_label(3, (1, 2, 4, 8)) == "2-3"
    assert bucket_label(7, (1, 2, 4, 8)) == "4-7"
    assert bucket_label(9, (1, 2, 4, 8)) == "8+"


def test_bucket_label_uses_zero_based_exclusive_upper_edges() -> None:
    edges = (2, 6, 16, 32, 64)

    assert bucket_label(0, edges) == "0-1"
    assert bucket_label(1, edges) == "0-1"
    assert bucket_label(2, edges) == "2-5"
    assert bucket_label(5, edges) == "2-5"
    assert bucket_label(6, edges) == "6-15"
    assert bucket_label(64, edges) == "64+"


def test_bucket_edges_are_json_serializable() -> None:
    edges = BucketEdges(
        window_count=(1, 2, 4, 8, 16),
        statement_count=(1, 2, 4, 8, 16),
        sample_window_count=(2, 6, 16, 32, 64),
    )

    assert edges.to_dict() == {
        "window_count": [1, 2, 4, 8, 16],
        "statement_count": [1, 2, 4, 8, 16],
        "sample_window_count": [2, 6, 16, 32, 64],
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"window_count": (0, 1, 2)}, "window_count edges must be positive"),
        ({"window_count": (-1, 1, 2)}, "window_count edges must be positive"),
        ({"statement_count": (1, 0, 2)}, "statement_count edges must be positive"),
        (
            {"sample_window_count": (0, 2, 4)},
            "sample_window_count edges must be positive",
        ),
    ],
)
def test_bucket_edges_reject_non_positive_edges(
    kwargs: dict[str, tuple[int, ...]],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        BucketEdges(**kwargs)


def test_detector_config_package_exports() -> None:
    assert detect.DETECTOR_MODE == DETECTOR_MODE
    assert detect.BucketEdges is BucketEdges
    assert detect.SawrDetectionConfig is SawrDetectionConfig
    assert detect.bucket_label is bucket_label


def test_sawr_root_reexports_detector_config_names() -> None:
    assert sawr.DETECTOR_MODE == DETECTOR_MODE
    assert sawr.BucketEdges is BucketEdges
    assert sawr.SawrDetectionConfig is SawrDetectionConfig


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"secret_key": ""}, "secret_key must be non-empty"),
        ({"lsh_d": 0}, "lsh_d must be >= 1"),
        ({"gamma": "0.75"}, "gamma must be a real number"),
        ({"gamma": True}, "gamma must be a real number"),
        ({"gamma": Fraction(3, 4)}, "gamma must be .*finite JSON number"),
        ({"gamma": Decimal("0.75")}, "gamma must be .*finite JSON number"),
        ({"gamma": float("nan")}, "gamma must be .*finite JSON number"),
        ({"gamma": -0.1}, "gamma must be in \\[0, 1\\]"),
        ({"gamma": 1.1}, "gamma must be in \\[0, 1\\]"),
        ({"semantic_margin": "0.0"}, "semantic_margin must be a real number"),
        ({"semantic_margin": False}, "semantic_margin must be a real number"),
        (
            {"semantic_margin": float("inf")},
            "semantic_margin must be .*finite JSON number",
        ),
        ({"semantic_margin": -0.01}, "semantic_margin must be non-negative"),
        ({"max_group_statements": 0}, "max_group_statements must be positive"),
        ({"min_scoreable_contexts": 0}, "min_scoreable_contexts must be positive"),
        ({"min_proxy_windows": 0}, "min_proxy_windows must be positive"),
        ({"target_fpr": "0.05"}, "target_fpr must be a real number"),
        ({"target_fpr": True}, "target_fpr must be a real number"),
        ({"target_fpr": Decimal("0.05")}, "target_fpr must be .*finite JSON number"),
        ({"target_fpr": float("nan")}, "target_fpr must be .*finite JSON number"),
        ({"target_fpr": 0.0}, "target_fpr must be in \\(0, 1\\)"),
        ({"target_fpr": 1.0}, "target_fpr must be in \\(0, 1\\)"),
        ({"use_ordinal_keying": 1}, "use_ordinal_keying must be bool"),
        ({"evidence_mode": "raw"}, "evidence_mode must be one of"),
        ({"statistic": "sum"}, "statistic must be one of"),
        ({"proxy_penalty_alpha": True}, "proxy_penalty_alpha must be a real number"),
        ({"proxy_penalty_alpha": "0.4"}, "proxy_penalty_alpha must be a real number"),
        (
            {"proxy_penalty_alpha": Decimal("0.4")},
            "proxy_penalty_alpha must be .*finite JSON number",
        ),
        (
            {"proxy_penalty_alpha": float("nan")},
            "proxy_penalty_alpha must be .*finite JSON number",
        ),
        ({"proxy_penalty_alpha": -0.1}, "proxy_penalty_alpha must be non-negative"),
        (
            {"code_length_adjustment_beta": True},
            "code_length_adjustment_beta must be a real number",
        ),
        (
            {"code_length_adjustment_beta": "0.3"},
            "code_length_adjustment_beta must be a real number",
        ),
        (
            {"code_length_adjustment_beta": float("nan")},
            "code_length_adjustment_beta must be .*finite JSON number",
        ),
        (
            {"code_length_reference_chars": 0},
            "code_length_reference_chars must be positive",
        ),
        (
            {"code_length_reference_chars": 1.5},
            "code_length_reference_chars must be positive",
        ),
        ({"structure_aware": "yes"}, "structure_aware must be bool"),
        ({"detector_mode": "bad"}, "detector_mode must be"),
    ],
)
def test_detection_config_rejects_invalid_values(
    kwargs: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {"secret_key": "1010"}
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        SawrDetectionConfig(**values)
