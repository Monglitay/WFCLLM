from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.batch_invariant_v4.config import load_public_config


def test_frozen_public_config_has_no_secret_confirmation_metadata() -> None:
    config = load_public_config("configs/wfcllm/v4_batch_invariant_public.json")
    text = repr(config).lower()
    assert "secret_key" not in text
    assert "key_fingerprint" not in text
    assert "key_sha256" not in text
    assert config.channel.bit_count_per_unit == 32
    assert config.channel.minimum_independent_units == 3
    assert config.runtime.encoder_used is False


def test_public_config_rejects_unknown_fields_and_secret_fields(tmp_path: Path) -> None:
    payload = json.loads(
        Path("configs/wfcllm/v4_batch_invariant_public.json").read_text()
    )
    payload["secret_key"] = "forbidden"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown|secret"):
        load_public_config(path)


def test_cli_surface_has_key_file_but_no_raw_key_argument() -> None:
    source = Path("scripts/wfcllm_v4_detect.py")
    assert source.exists()
    text = source.read_text(encoding="utf-8")
    assert "--key-file" in text
    assert "--secret-key" not in text
    assert "--raw-key" not in text
