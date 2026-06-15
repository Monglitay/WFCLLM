from __future__ import annotations

import json

import pytest

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


def test_bucket_label_uses_frozen_edges() -> None:
    assert bucket_label(0, (1, 2, 4, 8)) == "0"
    assert bucket_label(1, (1, 2, 4, 8)) == "1"
    assert bucket_label(3, (1, 2, 4, 8)) == "2-3"
    assert bucket_label(7, (1, 2, 4, 8)) == "4-7"
    assert bucket_label(9, (1, 2, 4, 8)) == "8+"


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
        ({"secret_key": ""}, "secret_key must be non-empty"),
        ({"lsh_d": 0}, "lsh_d must be >= 1"),
        ({"gamma": -0.1}, "gamma must be in \\[0, 1\\]"),
        ({"gamma": 1.1}, "gamma must be in \\[0, 1\\]"),
        ({"semantic_margin": -0.01}, "semantic_margin must be non-negative"),
        ({"max_group_statements": 0}, "max_group_statements must be positive"),
        ({"min_scoreable_contexts": 0}, "min_scoreable_contexts must be positive"),
        ({"min_proxy_windows": 0}, "min_proxy_windows must be positive"),
        ({"target_fpr": 0.0}, "target_fpr must be in \\(0, 1\\)"),
        ({"target_fpr": 1.0}, "target_fpr must be in \\(0, 1\\)"),
        ({"evidence_mode": "raw"}, "evidence_mode must be one of"),
        ({"statistic": "sum"}, "statistic must be one of"),
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
