"""Descriptor-bound watermark key derivation for semantic windows."""

from __future__ import annotations

import hashlib
import hmac
import json

_REGION_ID_CONTRACT = "semantic-window-region/v1"
_REGION_ID_DOMAIN = b"semantic-window-region-id\0"


class WatermarkKeying:
    """Derive keyed LSH regions from versioned window descriptors."""

    def __init__(self, secret_key: str, d: int):
        self._key = secret_key.encode("utf-8")
        self._d = d

    def derive_descriptor(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
    ) -> frozenset[tuple[int, ...]]:
        """Derive valid regions from a versioned semantic-window descriptor."""
        _validate_descriptor_component("contract_version", contract_version)
        _validate_descriptor_component("parent_descriptor", parent_descriptor)
        payload = (
            f"window-descriptor\0{contract_version}\0{parent_descriptor}"
        ).encode("utf-8")
        return self._derive_from_message(payload, k)

    def descriptor_region_id(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
        allowed: frozenset[tuple[int, ...]],
    ) -> str:
        """Return a keyed opaque identifier for descriptor-derived regions."""
        _validate_descriptor_component("contract_version", contract_version)
        _validate_descriptor_component("parent_descriptor", parent_descriptor)
        self._validate_k(k)
        self._validate_allowed(allowed, k=k)
        canonical = json.dumps(
            {
                "allowed_signatures": sorted(allowed),
                "contract_version": contract_version,
                "d": self._d,
                "k": k,
                "parent_descriptor": parent_descriptor,
                "region_id_contract": _REGION_ID_CONTRACT,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = hmac.new(
            self._key,
            _REGION_ID_DOMAIN + canonical,
            hashlib.sha256,
        ).hexdigest()
        return f"{_REGION_ID_CONTRACT}:hmac-sha256:{digest}"

    def _derive_from_message(
        self,
        message: bytes,
        k: int,
    ) -> frozenset[tuple[int, ...]]:
        """Shuffle the signature universe deterministically for ``message``."""
        self._validate_k(k)
        digest = hmac.new(self._key, message, hashlib.sha256).digest()

        seed = int.from_bytes(digest[:8], "big")

        # Enumerate all 2^d possible signatures
        all_sigs = [
            tuple(int(b) for b in format(i, f"0{self._d}b"))
            for i in range(2 ** self._d)
        ]

        # Deterministic Fisher-Yates shuffle using seed
        import random
        rng = random.Random(seed)
        shuffled = list(all_sigs)
        rng.shuffle(shuffled)

        return frozenset(shuffled[:k])

    def _validate_k(self, k: int) -> None:
        max_regions = 2 ** self._d
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if not (1 <= k < max_regions):
            raise ValueError("k must satisfy 1 <= k < 2**d")

    def _validate_allowed(
        self,
        allowed: object,
        *,
        k: int,
    ) -> None:
        if not isinstance(allowed, frozenset) or len(allowed) != k:
            raise ValueError("allowed must be a frozenset containing k signatures")
        for signature in allowed:
            if (
                not isinstance(signature, tuple)
                or len(signature) != self._d
                or any(
                    type(bit) is not int or bit not in (0, 1)
                    for bit in signature
                )
            ):
                raise ValueError("allowed signatures must be d-bit tuples")


def _validate_descriptor_component(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if "\0" in value:
        raise ValueError(f"{name} must not contain NUL")
