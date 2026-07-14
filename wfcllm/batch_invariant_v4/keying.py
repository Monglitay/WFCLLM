from __future__ import annotations

import hmac
import stat
from hashlib import sha256
from pathlib import Path


_HEADER = b"WFCLLM_BATCH_INVARIANT_SEMANTIC_V4_FORMAL\0"
_WRONG_LABEL = b"wfcllm-v4-independent-wrong-key-control"
_PUBLIC_CALIBRATION_DOMAIN = b"v4-public/calibration-null-reference"


class V4SecretKey:
    """Opaque formal V4 secret with no public confirmation identifier."""

    __slots__ = ("_material",)

    def __init__(self, material: bytes) -> None:
        if not isinstance(material, bytes):
            raise ValueError("secret key material must be bytes")
        if len(material) < 32:
            raise ValueError("secret key must contain at least 32 bytes")
        self._material = bytes(material)

    @classmethod
    def from_material_for_test(cls, material: bytes) -> V4SecretKey:
        return cls(material)

    def __repr__(self) -> str:
        return "V4SecretKey(<redacted>)"


def load_secret_key(path: str | Path) -> V4SecretKey:
    key_path = Path(path)
    try:
        info = key_path.stat()
    except OSError as exc:
        raise ValueError(f"failed to stat key file: {key_path}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("key file must be a regular file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("key file mode must be exactly 0600")
    try:
        material = key_path.read_bytes()
    except OSError as exc:
        raise ValueError(f"failed to read key file: {key_path}") from exc
    return V4SecretKey(material)


def derive_bytes(
    key: V4SecretKey,
    *,
    domain: str,
    message: bytes,
    length: int,
) -> bytes:
    if not isinstance(key, V4SecretKey):
        raise ValueError("key must be V4SecretKey")
    if not isinstance(domain, str) or not domain or "\0" in domain:
        raise ValueError("domain must be a non-empty NUL-free string")
    if not isinstance(message, bytes):
        raise ValueError("message must be bytes")
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("length must be a positive integer")
    prefix = _HEADER + domain.encode("utf-8") + b"\0" + message
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


def derive_wrong_control_key(key: V4SecretKey) -> V4SecretKey:
    return V4SecretKey(
        derive_bytes(
            key,
            domain="v4-formal/wrong-key-control",
            message=_WRONG_LABEL,
            length=32,
        )
    )


def public_calibration_reference_key() -> V4SecretKey:
    """Return the public null-reference material, never the primary V4 key."""

    return V4SecretKey(sha256(_PUBLIC_CALIBRATION_DOMAIN).digest())
