"""Tests for wfcllm.watermark.entropy_profile."""

from __future__ import annotations

import json

from wfcllm.watermark.entropy_profile import EntropyProfile


def test_load_reads_quantile_units_from_json(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "language": "python",
                "model_family": "codellama",
                "quantiles_units": {
                    "p10": 1200,
                    "p50": 4200,
                    "p75": 6800,
                    "p90": 9000,
                    "p95": 9800,
                },
            }
        ),
        encoding="utf-8",
    )

    profile = EntropyProfile.load(path)

    assert profile.language == "python"
    assert profile.model_family == "codellama"
    assert profile.quantiles_units_map["p90"] == 9000


def test_quantile_units_returns_exact_value(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "language": "python",
                "model_family": "starcoder",
                "quantiles_units": {
                    "p10": 1000,
                    "p50": 3000,
                    "p75": 5000,
                    "p90": 7000,
                    "p95": 9000,
                },
            }
        ),
        encoding="utf-8",
    )

    profile = EntropyProfile.load(path)

    assert profile.quantile_units("p75") == 5000


def test_load_rejects_non_object_json_root(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text("[]", encoding="utf-8")

    try:
        EntropyProfile.load(path)
    except ValueError as exc:
        assert str(exc) == "profile JSON root must be an object"
    else:
        raise AssertionError("Expected ValueError for non-object JSON root")


def test_load_rejects_bool_quantile_value(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "language": "python",
                "model_family": "starcoder",
                "quantiles_units": {
                    "p10": 1000,
                    "p50": 3000,
                    "p75": True,
                    "p90": 7000,
                    "p95": 9000,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        EntropyProfile.load(path)
    except ValueError as exc:
        assert str(exc) == "quantiles_units[p75] must be an integer"
    else:
        raise AssertionError("Expected ValueError for bool quantile units")


def test_load_rejects_non_monotonic_quantiles(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            {
                "language": "python",
                "model_family": "starcoder",
                "quantiles_units": {
                    "p10": 1000,
                    "p50": 3000,
                    "p75": 2500,
                    "p90": 7000,
                    "p95": 9000,
                },
            }
        ),
        encoding="utf-8",
    )

    try:
        EntropyProfile.load(path)
    except ValueError as exc:
        assert str(exc) == "quantiles_units must be monotonic: p10 <= p50 <= p75 <= p90 <= p95"
    else:
        raise AssertionError("Expected ValueError for non-monotonic quantiles")
