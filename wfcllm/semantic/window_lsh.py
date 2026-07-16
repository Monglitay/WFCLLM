"""Versioned semantic-LSH scoring for complete statement windows."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Protocol

from wfcllm.windowing.normalization import normalize_unit_text

_REGION_ID_PREFIX = "semantic-window-region/v1:hmac-sha256:"
_REGION_ID_PATTERN = re.compile(
    rf"{re.escape(_REGION_ID_PREFIX)}[0-9a-f]{{64}}"
)


class WindowVerifyResult(Protocol):
    min_margin: float
    lsh_signature: tuple[int, ...]


class WindowVerifier(Protocol):
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> WindowVerifyResult:
        ...


class DescriptorKeying(Protocol):
    def derive_descriptor(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
    ) -> frozenset[tuple[int, ...]]:
        ...

    def descriptor_region_id(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
        allowed: frozenset[tuple[int, ...]],
    ) -> str:
        ...


@dataclass(frozen=True)
class SemanticWindowEvidence:
    """Validated three-state evidence for one complete semantic window."""

    signature: tuple[int, ...]
    allowed_region_id: str
    hit: bool
    margin: float
    stable: bool

    def __post_init__(self) -> None:
        _validate_signature(self.signature, field_name="signature")
        if (
            not isinstance(self.allowed_region_id, str)
            or _REGION_ID_PATTERN.fullmatch(self.allowed_region_id) is None
        ):
            raise ValueError("allowed_region_id has an invalid format")
        if not isinstance(self.hit, bool):
            raise ValueError("hit must be a bool")
        if not isinstance(self.stable, bool):
            raise ValueError("stable must be a bool")
        _validate_margin(self.margin, field_name="margin")
        if self.hit and not self.stable:
            raise ValueError("hit requires stable evidence")


class SemanticWindowScorer:
    """Score one normalized, complete window against descriptor-derived regions."""

    def __init__(
        self,
        *,
        verifier: WindowVerifier,
        keying: DescriptorKeying,
        contract_version: str,
        k: int,
        margin: float,
    ) -> None:
        _validate_descriptor_component("contract_version", contract_version)
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError("k must be >= 1")
        _validate_margin(margin, field_name="margin")
        self._verifier = verifier
        self._keying = keying
        self._contract_version = contract_version
        self._k = k
        self._margin = margin

    def score(
        self,
        *,
        window_text: str,
        parent_descriptor: str,
    ) -> SemanticWindowEvidence:
        if not isinstance(window_text, str):
            raise ValueError("window_text must be a string")
        _validate_descriptor_component("parent_descriptor", parent_descriptor)
        normalized_text = normalize_unit_text(window_text)
        if not normalized_text:
            raise ValueError("window_text is empty after normalization")

        allowed = self._keying.derive_descriptor(
            contract_version=self._contract_version,
            parent_descriptor=parent_descriptor,
            k=self._k,
        )
        dimension = _validate_allowed_set(allowed, expected_size=self._k)
        allowed_region_id = self._keying.descriptor_region_id(
            contract_version=self._contract_version,
            parent_descriptor=parent_descriptor,
            k=self._k,
            allowed=allowed,
        )
        result = self._verifier.verify(
            normalized_text,
            allowed,
            self._margin,
        )

        signature = _require_result_attribute(result, "lsh_signature")
        _validate_signature(signature, field_name="lsh_signature")
        if len(signature) != dimension:
            raise ValueError("lsh_signature dimension does not match allowed set")

        min_margin = _require_result_attribute(result, "min_margin")
        _validate_margin(min_margin, field_name="min_margin")
        derived_membership = signature in allowed
        in_valid_set = _optional_result_attribute(result, "in_valid_set")
        if in_valid_set is not None:
            if not isinstance(in_valid_set, bool):
                raise ValueError("in_valid_set must be a bool when provided")
            if in_valid_set is not derived_membership:
                raise ValueError("in_valid_set contradicts lsh_signature membership")

        stable = min_margin > self._margin
        hit = stable and derived_membership
        passed = _optional_result_attribute(result, "passed")
        if passed is not None:
            if not isinstance(passed, bool):
                raise ValueError("passed must be a bool when provided")
            if passed is not hit:
                raise ValueError("passed contradicts derived window hit")

        return SemanticWindowEvidence(
            signature=signature,
            allowed_region_id=allowed_region_id,
            hit=hit,
            margin=min_margin,
            stable=stable,
        )


def _validate_signature(signature: object, *, field_name: str) -> None:
    if not isinstance(signature, tuple) or not signature:
        raise ValueError(f"{field_name} must be a non-empty tuple of bits")
    if any(type(bit) is not int or bit not in (0, 1) for bit in signature):
        raise ValueError(f"{field_name} must contain only integer bits")


def _validate_margin(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite non-negative number")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative")


def _validate_allowed_set(
    allowed: object,
    *,
    expected_size: int,
) -> int:
    if not isinstance(allowed, frozenset):
        raise ValueError("derive_descriptor must return a frozenset")
    if len(allowed) != expected_size:
        raise ValueError("derived allowed set size does not match k")
    dimensions: set[int] = set()
    for signature in allowed:
        _validate_signature(signature, field_name="allowed signature")
        dimensions.add(len(signature))
    if len(dimensions) != 1:
        raise ValueError("allowed signatures must have one dimension")
    return dimensions.pop()


def _require_result_attribute(result: object, name: str) -> Any:
    try:
        return getattr(result, name)
    except AttributeError as exc:
        raise ValueError(f"verifier result must define {name}") from exc


def _optional_result_attribute(result: object, name: str) -> Any | None:
    return getattr(result, name, None)


def _validate_descriptor_component(name: str, value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    if "\0" in value:
        raise ValueError(f"{name} must not contain NUL")
