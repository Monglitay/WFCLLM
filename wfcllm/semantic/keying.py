"""Descriptor-bound watermark key derivation for semantic windows."""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from typing import Any

_REGION_ID_CONTRACT = "semantic-window-region/v1"
_REGION_ID_DOMAIN = b"semantic-window-region-id\0"
_LAZY_SIGNATURE_SPACE_THRESHOLD = 1_000_000
_AFFINE_SELECTION_RULE = "affine-permutation-prefix/v1"


@dataclass(frozen=True)
class AffinePrefixSignatureSet:
    """O(1) representation of a keyed prefix of a 2^d signature permutation."""

    dimension: int
    size: int
    multiplier: int
    offset: int
    selection_rule: str = _AFFINE_SELECTION_RULE

    def __post_init__(self) -> None:
        if type(self.dimension) is not int or self.dimension < 1:
            raise ValueError("dimension must be a positive integer")
        max_regions = 2 ** self.dimension
        if type(self.size) is not int or not 1 <= self.size < max_regions:
            raise ValueError("size must satisfy 1 <= size < 2**dimension")
        if (
            type(self.multiplier) is not int
            or not 0 < self.multiplier < max_regions
            or self.multiplier % 2 != 1
        ):
            raise ValueError("multiplier must be an odd integer modulo 2**dimension")
        if type(self.offset) is not int or not 0 <= self.offset < max_regions:
            raise ValueError("offset must be an integer modulo 2**dimension")
        if self.selection_rule != _AFFINE_SELECTION_RULE:
            raise ValueError("unsupported lazy signature selection rule")

    def __len__(self) -> int:
        return self.size

    def __contains__(self, signature: object) -> bool:
        if (
            not isinstance(signature, tuple)
            or len(signature) != self.dimension
            or any(type(bit) is not int or bit not in (0, 1) for bit in signature)
        ):
            return False
        value = 0
        for bit in signature:
            value = (value << 1) | bit
        mask = (1 << self.dimension) - 1
        return ((self.multiplier * value + self.offset) & mask) < self.size

    def canonical_payload(self) -> dict[str, int | str]:
        return {
            "d": self.dimension,
            "k": self.size,
            "multiplier": self.multiplier,
            "offset": self.offset,
            "selection_rule": self.selection_rule,
        }


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
    ) -> frozenset[tuple[int, ...]] | AffinePrefixSignatureSet:
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
        allowed: frozenset[tuple[int, ...]] | AffinePrefixSignatureSet,
    ) -> str:
        """Return a keyed opaque identifier for descriptor-derived regions."""
        _validate_descriptor_component("contract_version", contract_version)
        _validate_descriptor_component("parent_descriptor", parent_descriptor)
        self._validate_k(k)
        self._validate_allowed(allowed, k=k)
        if isinstance(allowed, AffinePrefixSignatureSet):
            allowed_payload: dict[str, Any] = {
                "allowed_selection": allowed.canonical_payload()
            }
        else:
            allowed_payload = {"allowed_signatures": sorted(allowed)}
        canonical = json.dumps(
            {
                **allowed_payload,
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
    ) -> frozenset[tuple[int, ...]] | AffinePrefixSignatureSet:
        """Shuffle the signature universe deterministically for ``message``."""
        self._validate_k(k)
        digest = hmac.new(self._key, message, hashlib.sha256).digest()

        seed = int.from_bytes(digest[:8], "big")
        max_regions = 2 ** self._d
        if max_regions > _LAZY_SIGNATURE_SPACE_THRESHOLD:
            affine_digest = hmac.new(
                self._key,
                b"semantic-window-affine-prefix\0" + message,
                hashlib.sha256,
            ).digest()
            mask = max_regions - 1
            multiplier = int.from_bytes(affine_digest[:16], "big") & mask
            multiplier |= 1
            offset = int.from_bytes(affine_digest[16:], "big") & mask
            return AffinePrefixSignatureSet(
                dimension=self._d,
                size=k,
                multiplier=multiplier,
                offset=offset,
            )

        # Enumerate all 2^d possible signatures
        all_sigs = [
            tuple(int(b) for b in format(i, f"0{self._d}b"))
            for i in range(max_regions)
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
        if isinstance(allowed, AffinePrefixSignatureSet):
            if allowed.dimension != self._d or len(allowed) != k:
                raise ValueError("lazy allowed signature set shape mismatch")
            return
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
