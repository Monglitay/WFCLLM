from __future__ import annotations

import hmac
import os
import stat
from hashlib import sha256
from pathlib import Path


_DERIVATION_HEADER = b"WFCLLM_DYNAMIC_SEMANTIC_V3\0"
_WRONG_KEY_LABEL = b"wfcllm-v3-wrong-key-control"


class SecretKey:
    """Opaque in-process private key with deliberately redacted representation."""

    __slots__ = ("_material",)

    def __init__(self, material: bytes) -> None:
        if len(material) < 32:
            raise ValueError("secret key must contain at least 32 bytes")
        self._material = bytes(material)

    @classmethod
    def from_material_for_test(cls, material: bytes) -> SecretKey:
        return cls(material)

    def __repr__(self) -> str:
        return "SecretKey(<redacted>)"


def load_secret_key(path: str | Path) -> SecretKey:
    key_path = Path(path)
    try:
        file_stat = key_path.stat()
    except OSError as exc:
        raise ValueError(f"failed to stat private key file: {key_path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise ValueError("private key path must be a regular file")
    if stat.S_IMODE(file_stat.st_mode) != 0o600:
        raise ValueError("private key file mode must be exactly 0600")
    try:
        material = key_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"failed to read private key file: {key_path}") from exc
    return SecretKey(material)


def derive_bytes(
    key: SecretKey,
    domain: str,
    message: bytes,
    length: int,
) -> bytes:
    if not isinstance(key, SecretKey):
        raise ValueError("key must be SecretKey")
    if not isinstance(domain, str) or not domain or "\0" in domain:
        raise ValueError("domain must be a non-empty NUL-free string")
    if not isinstance(message, bytes):
        raise ValueError("message must be bytes")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")

    prefix = _DERIVATION_HEADER + domain.encode("utf-8") + b"\0" + message
    output = bytearray()
    counter = 0
    while len(output) < length:
        counter += 1
        output.extend(
            hmac.new(
                key._material,
                prefix + counter.to_bytes(4, "big"),
                sha256,
            ).digest()
        )
    return bytes(output[:length])


def derive_wrong_control_key(key: SecretKey) -> SecretKey:
    material = derive_bytes(
        key,
        domain="wrong-key-control",
        message=_WRONG_KEY_LABEL,
        length=32,
    )
    return SecretKey(material)
