from __future__ import annotations

from pathlib import Path

import pytest

from wfcllm.batch_invariant_v4.keying import (
    V4SecretKey,
    derive_bytes,
    derive_wrong_control_key,
    load_secret_key,
)


def test_key_file_requires_exact_0600_and_repr_is_redacted(tmp_path: Path) -> None:
    path = tmp_path / "v4.key"
    path.write_bytes(b"k" * 32)
    path.chmod(0o600)
    key = load_secret_key(path)

    assert repr(key) == "V4SecretKey(<redacted>)"
    path.chmod(0o640)
    with pytest.raises(ValueError, match="0600"):
        load_secret_key(path)


def test_formal_projection_target_and_wrong_key_domains_are_distinct() -> None:
    key = V4SecretKey.from_material_for_test(b"a" * 32)
    message = b"same-message"

    projection = derive_bytes(
        key,
        domain="v4-formal/structural-signature",
        message=message,
        length=32,
    )
    target = derive_bytes(
        key,
        domain="v4-formal/unit-target",
        message=message,
        length=32,
    )
    wrong = derive_wrong_control_key(key)

    assert projection != target
    assert derive_bytes(
        wrong,
        domain="v4-formal/structural-signature",
        message=message,
        length=32,
    ) != projection
    assert repr(wrong) == "V4SecretKey(<redacted>)"


def test_key_api_rejects_raw_string_material() -> None:
    with pytest.raises(ValueError, match="bytes"):
        V4SecretKey("raw-key-is-forbidden")  # type: ignore[arg-type]
