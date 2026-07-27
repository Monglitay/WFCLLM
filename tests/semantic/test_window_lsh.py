from __future__ import annotations

import re
from dataclasses import dataclass

import pytest
import torch

from wfcllm.semantic import SemanticWindowEvidence, SemanticWindowScorer
from wfcllm.semantic.keying import WatermarkKeying


@dataclass
class _FakeVerifyResult:
    passed: bool
    min_margin: float
    lsh_signature: tuple[int, ...]
    in_valid_set: bool


class _FakeVerifier:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[
            tuple[str, frozenset[tuple[int, ...]], float]
        ] = []

    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> object:
        self.calls.append((code_text, valid_set, margin))
        return self.result


class _FakeKeying:
    def __init__(self, allowed: frozenset[tuple[int, ...]]) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, str, int]] = []
        self.region_calls: list[
            tuple[str, str, int, frozenset[tuple[int, ...]]]
        ] = []
        self.region_id = "semantic-window-region/v1:hmac-sha256:" + "a" * 64

    def derive_descriptor(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
    ) -> frozenset[tuple[int, ...]]:
        self.calls.append((contract_version, parent_descriptor, k))
        return self.allowed

    def descriptor_region_id(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
        allowed: frozenset[tuple[int, ...]],
    ) -> str:
        self.region_calls.append(
            (contract_version, parent_descriptor, k, allowed)
        )
        return self.region_id


def _scorer(
    result: object,
    *,
    allowed: frozenset[tuple[int, ...]] = frozenset({(1, 0, 1, 0)}),
    margin: float = 0.1,
) -> tuple[SemanticWindowScorer, _FakeVerifier, _FakeKeying]:
    verifier = _FakeVerifier(result)
    keying = _FakeKeying(allowed)
    scorer = SemanticWindowScorer(
        verifier=verifier,
        keying=keying,
        contract_version="python-statement-window/v1",
        k=len(allowed),
        margin=margin,
    )
    return scorer, verifier, keying


def test_complete_normalized_window_is_verified_once_as_one_embedding() -> None:
    descriptor = (
        "python-statement-window/v1|module/function_definition|"
        "parent=block|ordinal=1|role=body"
    )
    scorer, verifier, keying = _scorer(
        _FakeVerifyResult(
            passed=True,
            min_margin=0.42,
            lsh_signature=(1, 0, 1, 0),
            in_valid_set=True,
        )
    )

    evidence = scorer.score(
        window_text="\n  x = 1  \r\n  return x \t\n",
        parent_descriptor=descriptor,
    )

    assert verifier.calls == [
        ("x = 1\n  return x", frozenset({(1, 0, 1, 0)}), 0.1)
    ]
    assert keying.calls == [
        ("python-statement-window/v1", descriptor, 1)
    ]
    assert keying.region_calls == [
        (
            "python-statement-window/v1",
            descriptor,
            1,
            frozenset({(1, 0, 1, 0)}),
        )
    ]
    assert evidence.allowed_region_id == keying.region_id
    assert evidence.signature == (1, 0, 1, 0)
    assert evidence.hit is True
    assert evidence.stable is True
    assert evidence.margin == pytest.approx(0.42)


class _ChannelKeying(_FakeKeying):
    def derive_descriptor(
        self,
        *,
        contract_version: str,
        parent_descriptor: str,
        k: int,
    ) -> frozenset[tuple[int, ...]]:
        self.calls.append((contract_version, parent_descriptor, k))
        if parent_descriptor.endswith("wfcllm-evidence-channel=1"):
            return frozenset({(0, 0, 0, 1)})
        return frozenset({(1, 0, 1, 0)})


def test_channel_scoring_reuses_one_embedding_with_domain_separated_regions() -> None:
    verifier = _FakeVerifier(
        _FakeVerifyResult(
            passed=True,
            min_margin=0.42,
            lsh_signature=(1, 0, 1, 0),
            in_valid_set=True,
        )
    )
    keying = _ChannelKeying(frozenset({(1, 0, 1, 0)}))
    scorer = SemanticWindowScorer(
        verifier=verifier,
        keying=keying,
        contract_version="python-statement-window/v1",
        k=1,
        margin=0.1,
    )

    evidence = scorer.score_channels(
        window_text="return x",
        parent_descriptor="descriptor",
        channel_count=2,
    )

    assert len(verifier.calls) == 1
    assert [item.hit for item in evidence] == [True, False]
    assert keying.calls == [
        ("python-statement-window/v1", "descriptor", 1),
        (
            "python-statement-window/v1",
            "descriptor|wfcllm-evidence-channel=1",
            1,
        ),
    ]


def test_stable_signature_outside_allowed_set_is_a_stable_miss() -> None:
    scorer, _, _ = _scorer(
        _FakeVerifyResult(
            passed=False,
            min_margin=0.42,
            lsh_signature=(0, 0, 0, 1),
            in_valid_set=False,
        )
    )

    evidence = scorer.score(window_text="return x", parent_descriptor="descriptor")

    assert evidence.hit is False
    assert evidence.stable is True


@pytest.mark.parametrize("min_margin", [0.1, 0.099])
def test_margin_at_or_below_threshold_is_uncertain(min_margin: float) -> None:
    scorer, _, _ = _scorer(
        _FakeVerifyResult(
            passed=False,
            min_margin=min_margin,
            lsh_signature=(1, 0, 1, 0),
            in_valid_set=True,
        )
    )

    evidence = scorer.score(window_text="return x", parent_descriptor="descriptor")

    assert evidence.hit is False
    assert evidence.stable is False


class _MembershipVerifier:
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> _FakeVerifyResult:
        signature = min(valid_set)
        return _FakeVerifyResult(
            passed=True,
            min_margin=margin + 1.0,
            lsh_signature=signature,
            in_valid_set=True,
        )


def test_allowed_region_id_is_stable_opaque_and_descriptor_specific() -> None:
    secret = "deployment-secret"
    first_descriptor = "python-statement-window/v1|module|parent=block|ordinal=1|role=body"
    second_descriptor = "python-statement-window/v1|module|parent=block|ordinal=2|role=body"
    scorer = SemanticWindowScorer(
        verifier=_MembershipVerifier(),
        keying=WatermarkKeying(secret, d=4),
        contract_version="python-statement-window/v1",
        k=4,
        margin=0.1,
    )

    first = scorer.score(window_text="return x", parent_descriptor=first_descriptor)
    repeated = scorer.score(window_text="return x", parent_descriptor=first_descriptor)
    second = scorer.score(window_text="return x", parent_descriptor=second_descriptor)

    assert first.allowed_region_id == repeated.allowed_region_id
    assert first.allowed_region_id != second.allowed_region_id
    assert re.fullmatch(
        r"semantic-window-region/v1:hmac-sha256:[0-9a-f]{64}",
        first.allowed_region_id,
    )
    assert secret not in first.allowed_region_id
    assert first_descriptor not in first.allowed_region_id


def test_semantic_inputs_use_one_canonical_text_across_layout_variants() -> None:
    from wfcllm.semantic.window_lsh import canonical_semantic_window_text

    variants = (
        "x = 1",
        "x = 1\n",
        "x = 1\r\n",
        "x = 1\n\n",
        "x = 1   \n",
    )
    assert {canonical_semantic_window_text(value) for value in variants} == {"x = 1"}


@dataclass
class _MinimalVerifyResult:
    lsh_signature: tuple[int, ...]
    min_margin: float


def test_minimal_verifier_result_needs_only_signature_and_margin() -> None:
    scorer, _, _ = _scorer(
        _MinimalVerifyResult(lsh_signature=(1, 0, 1, 0), min_margin=0.42)
    )

    evidence = scorer.score(window_text="return x", parent_descriptor="descriptor")

    assert evidence.hit is True
    assert evidence.stable is True


@dataclass
class _OptionalNoneVerifyResult:
    lsh_signature: tuple[int, ...]
    min_margin: float
    in_valid_set: None = None
    passed: None = None


def test_none_optional_verifier_fields_are_treated_as_unavailable() -> None:
    scorer, _, _ = _scorer(
        _OptionalNoneVerifyResult(
            lsh_signature=(0, 0, 0, 1),
            min_margin=0.42,
        )
    )

    evidence = scorer.score(window_text="return x", parent_descriptor="descriptor")

    assert evidence.hit is False
    assert evidence.stable is True


@pytest.mark.parametrize("window_text", ["", "   ", "\t", "\r\n", " \t\r\n \n"])
def test_empty_normalized_window_is_rejected_before_downstream_calls(
    window_text: str,
) -> None:
    scorer, verifier, keying = _scorer(
        _MinimalVerifyResult(lsh_signature=(1, 0, 1, 0), min_margin=0.42)
    )

    with pytest.raises(ValueError, match="window_text.*empty"):
        scorer.score(window_text=window_text, parent_descriptor="descriptor")

    assert keying.calls == []
    assert keying.region_calls == []
    assert verifier.calls == []


@pytest.mark.parametrize("contract_version", [None, 1, "", "contract\0v1"])
def test_scorer_rejects_invalid_contract_version(
    contract_version: object,
) -> None:
    with pytest.raises(ValueError, match="contract_version"):
        SemanticWindowScorer(
            verifier=_FakeVerifier(
                _MinimalVerifyResult((1, 0, 1, 0), 0.42)
            ),
            keying=_FakeKeying(frozenset({(1, 0, 1, 0)})),
            contract_version=contract_version,
            k=1,
            margin=0.1,
        )


@pytest.mark.parametrize("parent_descriptor", [None, 1, "", "descriptor\0tail"])
def test_scorer_rejects_invalid_parent_descriptor(
    parent_descriptor: object,
) -> None:
    scorer, _, _ = _scorer(
        _MinimalVerifyResult(lsh_signature=(1, 0, 1, 0), min_margin=0.42)
    )

    with pytest.raises(ValueError, match="parent_descriptor"):
        scorer.score(
            window_text="return x",
            parent_descriptor=parent_descriptor,
        )


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (_FakeVerifyResult(False, float("nan"), (1, 0, 1, 0), True), "finite"),
        (_FakeVerifyResult(False, -0.1, (1, 0, 1, 0), True), "non-negative"),
        (_FakeVerifyResult(False, 0.42, (1, 2, 1, 0), False), "bits"),
        (_FakeVerifyResult(False, 0.42, (1, 0), False), "dimension"),
        (_FakeVerifyResult(False, 0.42, (1, 0, 1, 0), False), "in_valid_set"),
        (_FakeVerifyResult(False, 0.42, (1, 0, 1, 0), True), "passed"),
        (_FakeVerifyResult(False, 0.42, (0, 0, 0, 1), "no"), "in_valid_set"),
        (_FakeVerifyResult("yes", 0.42, (1, 0, 1, 0), True), "passed"),
    ],
)
def test_scorer_rejects_malformed_or_contradictory_verifier_results(
    result: _FakeVerifyResult,
    message: str,
) -> None:
    scorer, _, _ = _scorer(result)

    with pytest.raises(ValueError, match=message):
        scorer.score(window_text="return x", parent_descriptor="descriptor")


@pytest.mark.parametrize(
    "changes",
    [
        {"signature": (1, 0, 2, 0)},
        {"allowed_region_id": "not-an-id"},
        {"hit": True, "stable": False},
        {"margin": float("inf")},
        {"margin": -0.1},
    ],
)
def test_window_evidence_validates_public_contract(changes: dict[str, object]) -> None:
    values: dict[str, object] = {
        "signature": (1, 0, 1, 0),
        "allowed_region_id": (
            "semantic-window-region/v1:hmac-sha256:" + "a" * 64
        ),
        "hit": False,
        "margin": 0.42,
        "stable": True,
    }
    values.update(changes)

    with pytest.raises(ValueError):
        SemanticWindowEvidence(**values)


def test_scorer_applies_fixed_key_independent_preservation_threshold() -> None:
    result = _FakeVerifyResult(False, 0.4, (1, 0, 1, 0), False)
    scorer, verifier, _keying = _scorer(result)
    verifier.semantic_reference_cosine = lambda reference, candidate: 0.89
    scorer = SemanticWindowScorer(
        verifier=verifier,
        keying=_FakeKeying(frozenset({(1, 0, 1, 0)})),
        contract_version="python-statement-window/v1",
        k=1,
        margin=0.0,
        semantic_preservation_threshold=0.9,
    )

    evidence = scorer.compare_semantics(
        reference_text="x = y + 1", candidate_text="x = 1 + y"
    )

    assert evidence.cosine == pytest.approx(0.89)
    assert evidence.threshold == pytest.approx(0.9)
    assert evidence.passed is False


def test_scorer_rejects_real_precision_or_batch_instability() -> None:
    class ModeVerifier(_FakeVerifier):
        def verify_modes(self, code_text, valid_set, margin):
            self.calls.append((code_text, valid_set, margin))
            return type(
                "ModeResult",
                (),
                {
                    "lsh_signature": (1, 0, 1, 0),
                    "min_margin": 0.4,
                    "in_valid_set": True,
                    "passed": False,
                    "stable_across_precision_modes": False,
                    "stable_across_batch_modes": True,
                },
            )()

    verifier = ModeVerifier(object())
    scorer = SemanticWindowScorer(
        verifier=verifier,
        keying=_FakeKeying(frozenset({(1, 0, 1, 0)})),
        contract_version="python-statement-window/v1",
        k=1,
        margin=0.0,
    )

    evidence = scorer.score(
        window_text="x = 1", parent_descriptor="module/body"
    )

    assert evidence.stable is False
    assert evidence.hit is False
