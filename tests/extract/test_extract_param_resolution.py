import pytest
from wfcllm.cli.config_resolver import resolve_extract_lsh_params
from wfcllm.extract.config import ExtractConfig


def test_extract_prefers_embedded_metadata_over_stale_config():
    record = {"watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75}}
    ext_cfg = {"lsh_d": 3, "lsh_gamma": 0.5}
    resolved = resolve_extract_lsh_params(record, ext_cfg)
    assert resolved == (4, 0.75)


def test_extract_uses_config_defaults_when_metadata_missing():
    resolved = resolve_extract_lsh_params({}, {"lsh_d": 3, "lsh_gamma": 0.5})
    assert resolved == (3, 0.5)


def test_extract_config_accepts_explicit_lsh_fields():
    cfg = ExtractConfig(secret_key="k", lsh_d=4, lsh_gamma=0.75)
    assert cfg.lsh_d == 4
    assert cfg.lsh_gamma == 0.75


def test_extract_rejects_invalid_lsh_values():
    with pytest.raises(ValueError):
        resolve_extract_lsh_params(
            {"watermark_params": {"lsh_d": "bad", "lsh_gamma": 0.75}},
            {"lsh_d": 3, "lsh_gamma": 0.5},
        )
