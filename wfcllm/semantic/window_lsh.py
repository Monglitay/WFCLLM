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


def canonical_semantic_window_text(window_text: str) -> str:
    """Return the sole encoder input representation for a statement window."""

    if not isinstance(window_text, str):
        raise ValueError("semantic window text must be a string")
    normalized = normalize_unit_text(window_text)
    if not normalized:
        raise ValueError("window_text is empty after semantic normalization")
    return normalized


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

    def semantic_reference_cosine(
        self, reference_text: str, candidate_text: str
    ) -> float: ...


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


@dataclass(frozen=True)
class SemanticPreservationEvidence:
    """Key-independent evidence comparing one rewrite to its original window."""

    cosine: float
    threshold: float
    passed: bool

    def __post_init__(self) -> None:
        _validate_cosine(self.cosine, field_name="cosine")
        _validate_cosine(self.threshold, field_name="threshold")
        if not isinstance(self.passed, bool):
            raise ValueError("passed must be a bool")
        if self.passed is not (self.cosine >= self.threshold):
            raise ValueError("passed must equal cosine >= threshold")


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
        semantic_preservation_threshold: float = 0.9,
    ) -> None:
        _validate_descriptor_component("contract_version", contract_version)
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an int")
        if k < 1:
            raise ValueError("k must be >= 1")
        _validate_margin(margin, field_name="margin")
        _validate_cosine(
            semantic_preservation_threshold,
            field_name="semantic_preservation_threshold",
        )
        self._verifier = verifier
        self._keying = keying
        self._contract_version = contract_version
        self._k = k
        self._margin = margin
        self._semantic_preservation_threshold = float(
            semantic_preservation_threshold
        )

    def compare_semantics(
        self, *, reference_text: str, candidate_text: str
    ) -> SemanticPreservationEvidence:
        canonical_reference = canonical_semantic_window_text(reference_text)
        canonical_candidate = canonical_semantic_window_text(candidate_text)
        compare = getattr(self._verifier, "semantic_reference_cosine", None)
        if not callable(compare):
            raise ValueError(
                "semantic verifier must expose semantic_reference_cosine"
            )
        cosine = compare(canonical_reference, canonical_candidate)
        _validate_cosine(cosine, field_name="semantic reference cosine")
        return SemanticPreservationEvidence(
            cosine=float(cosine),
            threshold=self._semantic_preservation_threshold,
            passed=bool(cosine >= self._semantic_preservation_threshold),
        )

    def score(
        self,
        *,
        window_text: str,
        parent_descriptor: str,
    ) -> SemanticWindowEvidence:
        _validate_descriptor_component("parent_descriptor", parent_descriptor)
        normalized_text = canonical_semantic_window_text(window_text)

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
        verify_modes = getattr(self._verifier, "verify_modes", None)
        result = (
            verify_modes(normalized_text, allowed, self._margin)
            if callable(verify_modes)
            else self._verifier.verify(
                normalized_text,
                allowed,
                self._margin,
            )
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

        precision_stable = _optional_result_attribute(
            result, "stable_across_precision_modes"
        )
        batch_stable = _optional_result_attribute(
            result, "stable_across_batch_modes"
        )
        if precision_stable is not None and not isinstance(precision_stable, bool):
            raise ValueError("precision stability must be a bool")
        if batch_stable is not None and not isinstance(batch_stable, bool):
            raise ValueError("batch stability must be a bool")
        stable = (
            min_margin > self._margin
            and precision_stable is not False
            and batch_stable is not False
        )
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

    def score_channels(
        self,
        *,
        window_text: str,
        parent_descriptor: str,
        channel_count: int,
    ) -> tuple[SemanticWindowEvidence, ...]:
        """Score domain-separated keyed regions from one public embedding."""

        if (
            isinstance(channel_count, bool)
            or not isinstance(channel_count, int)
            or not 1 <= channel_count <= 4
        ):
            raise ValueError("channel_count must be an integer in [1, 4]")
        first = self.score(
            window_text=window_text,
            parent_descriptor=parent_descriptor,
        )
        output = [first]
        for channel in range(1, channel_count):
            descriptor = (
                f"{parent_descriptor}|wfcllm-evidence-channel={channel}"
            )
            allowed = self._keying.derive_descriptor(
                contract_version=self._contract_version,
                parent_descriptor=descriptor,
                k=self._k,
            )
            dimension = _validate_allowed_set(
                allowed, expected_size=self._k
            )
            if len(first.signature) != dimension:
                raise ValueError(
                    "lsh_signature dimension does not match allowed set"
                )
            region_id = self._keying.descriptor_region_id(
                contract_version=self._contract_version,
                parent_descriptor=descriptor,
                k=self._k,
                allowed=allowed,
            )
            output.append(
                SemanticWindowEvidence(
                    signature=first.signature,
                    allowed_region_id=region_id,
                    hit=first.stable and first.signature in allowed,
                    margin=first.margin,
                    stable=first.stable,
                )
            )
        return tuple(output)


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


def _validate_cosine(value: object, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number in [-1, 1]")
    if not math.isfinite(value) or not -1.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must be in [-1, 1]")


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
