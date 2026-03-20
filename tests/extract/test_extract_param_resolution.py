import run
import pytest
from wfcllm.extract.config import ExtractConfig


def test_extract_prefers_embedded_metadata_over_stale_config():
    try:
        resolve_extract_lsh_params = run.resolve_extract_lsh_params
    except AttributeError as exc:
        raise AssertionError(
            "run.py must provide resolve_extract_lsh_params helper"
        ) from exc

    record = {"watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75}}
    ext_cfg = {"lsh_d": 3, "lsh_gamma": 0.5}
    resolved = resolve_extract_lsh_params(record, ext_cfg)
    assert resolved == (4, 0.75)


def test_extract_uses_config_defaults_when_metadata_missing():
    resolved = run.resolve_extract_lsh_params({}, {"lsh_d": 3, "lsh_gamma": 0.5})
    assert resolved == (3, 0.5)


def test_extract_config_accepts_explicit_lsh_fields():
    cfg = ExtractConfig(secret_key="k", lsh_d=4, lsh_gamma=0.75)
    assert cfg.lsh_d == 4
    assert cfg.lsh_gamma == 0.75


def test_extract_rejects_invalid_lsh_values():
    with pytest.raises(ValueError):
        run.resolve_extract_lsh_params(
            {"watermark_params": {"lsh_d": "bad", "lsh_gamma": 0.75}},
            {"lsh_d": 3, "lsh_gamma": 0.5},
        )
