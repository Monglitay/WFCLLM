from __future__ import annotations

import os
from pathlib import Path

import pytest

from wfcllm.dynamic_semantic.keying import (
    SecretKey,
    derive_bytes,
    derive_wrong_control_key,
    load_secret_key,
)


def _write_key(path: Path, material: bytes, mode: int = 0o600) -> None:
    path.write_bytes(material)
    os.chmod(path, mode)


def test_load_secret_key_requires_mode_0600(tmp_path: Path) -> None:
    path = tmp_path / "v3.key"
    _write_key(path, b"a" * 32, mode=0o640)

    with pytest.raises(ValueError, match="0600"):
        load_secret_key(path)


@pytest.mark.parametrize("material", [b"", b"short"])
def test_load_secret_key_rejects_short_material(
    tmp_path: Path,
    material: bytes,
) -> None:
    path = tmp_path / "v3.key"
    _write_key(path, material)

    with pytest.raises(ValueError, match="at least 32"):
        load_secret_key(path)


def test_secret_key_repr_and_api_do_not_expose_material_or_fingerprint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "v3.key"
    material = b"sensitive-private-material-00001"
    _write_key(path, material)

    key = load_secret_key(path)

    assert material.decode("ascii") not in repr(key)
    assert not hasattr(key, "fingerprint")
    assert not hasattr(key, "hex")


def test_hmac_derivation_is_deterministic_and_domain_separated() -> None:
    key = SecretKey.from_material_for_test(b"k" * 32)

    first = derive_bytes(key, domain="projection", message=b"unit", length=48)
    repeated = derive_bytes(key, domain="projection", message=b"unit", length=48)
    other_domain = derive_bytes(key, domain="target", message=b"unit", length=48)
    other_message = derive_bytes(key, domain="projection", message=b"other", length=48)

    assert first == repeated
    assert first != other_domain
    assert first != other_message
    assert len(first) == 48


def test_wrong_control_key_is_domain_separated_and_stable() -> None:
    key = SecretKey.from_material_for_test(b"p" * 32)

    wrong_a = derive_wrong_control_key(key)
    wrong_b = derive_wrong_control_key(key)

    probe = b"public-probe"
    assert derive_bytes(wrong_a, "target", probe, 32) == derive_bytes(
        wrong_b, "target", probe, 32
    )
    assert derive_bytes(wrong_a, "target", probe, 32) != derive_bytes(
        key, "target", probe, 32
    )
