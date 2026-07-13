from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from wfcllm.detection.canonical_v2 import extract_canonical_units
from wfcllm.detection.signature_v2 import (
    CanonicalSignatureScorer,
    derive_target_signature,
    robust_unit_mean,
    standardized_bit_sum,
)


class _FakeVerifier:
    def __init__(self, signatures: dict[str, tuple[int, ...]]) -> None:
        self._signatures = signatures

    def verify(self, code_text, valid_set, margin):
        signature = self._signatures[code_text]
        return SimpleNamespace(
            passed=signature in valid_set,
            lsh_signature=signature,
            min_margin=1.0,
            in_valid_set=signature in valid_set,
        )


def _unit():
    return extract_canonical_units("def target(x):\n    return x + 1\n").units[0]


def test_target_signature_is_deterministic_and_key_conditioned() -> None:
    unit = _unit()

    left = derive_target_signature(unit, secret_key="left", bits=16)
    left_again = derive_target_signature(unit, secret_key="left", bits=16)
    right = derive_target_signature(unit, secret_key="right", bits=16)

    assert left == left_again
    assert left != right
    assert len(left) == len(right) == 16
    assert set(left) <= {0, 1}


def test_canonical_signature_scorer_reports_exact_bit_agreement() -> None:
    code = "def target(x):\n    return x + 1\n"
    unit = extract_canonical_units(code).units[0]
    target = derive_target_signature(unit, secret_key="key", bits=8)
    actual = tuple([*target[:6], 1 - target[6], 1 - target[7]])
    scorer = CanonicalSignatureScorer(
        verifier=_FakeVerifier({unit.normalized_text: actual}),
        secret_key="key",
        signature_bits=8,
    )

    score = scorer.score_code(code)

    assert score.raw_score == pytest.approx(0.75)
    assert score.matched_bits == 6
    assert score.total_bits == 8
    assert score.unit_count == 1
    assert score.duplicate_count == 0
    assert score.unit_evidence[0].actual_signature == actual
    assert score.unit_evidence[0].target_signature == target


def test_robust_unit_mean_trims_one_value_per_tail_from_five_units() -> None:
    assert robust_unit_mean([0.0, 0.25, 0.5, 0.75]) == pytest.approx(0.375)
    assert robust_unit_mean([0.0, 0.25, 0.5, 0.75, 1.0]) == pytest.approx(0.5)
    assert robust_unit_mean([]) == 0.0


def test_standardized_bit_sum_accounts_for_effective_bit_count() -> None:
    assert standardized_bit_sum(matched_bits=6, total_bits=8) == pytest.approx(
        math.sqrt(2.0)
    )
    assert standardized_bit_sum(matched_bits=12, total_bits=16) == pytest.approx(2.0)
    assert standardized_bit_sum(matched_bits=0, total_bits=0) == 0.0

    with pytest.raises(ValueError, match="matched_bits"):
        standardized_bit_sum(matched_bits=9, total_bits=8)


def test_scorer_can_use_standardized_bit_sum_aggregation() -> None:
    code = "def target(x):\n    return x + 1\n"
    unit = extract_canonical_units(code).units[0]
    target = derive_target_signature(unit, secret_key="key", bits=8)
    actual = tuple([*target[:6], 1 - target[6], 1 - target[7]])
    scorer = CanonicalSignatureScorer(
        verifier=_FakeVerifier({unit.normalized_text: actual}),
        secret_key="key",
        signature_bits=8,
        aggregation="standardized_bit_sum",
    )

    score = scorer.score_code(code)

    assert score.matched_bits == 6
    assert score.total_bits == 8
    assert score.raw_score == pytest.approx(math.sqrt(2.0))


def test_scorer_deduplicates_units_before_aggregation() -> None:
    code = '''
def target(x):
    if x:
        value = 1
    else:
        value = 1
    return value
'''
    extraction = extract_canonical_units(code)
    signatures = {}
    for unit in extraction.units:
        signatures[unit.normalized_text] = derive_target_signature(
            unit,
            secret_key="key",
            bits=8,
        )
    scorer = CanonicalSignatureScorer(
        verifier=_FakeVerifier(signatures),
        secret_key="key",
        signature_bits=8,
    )

    score = scorer.score_code(code)

    assert score.unit_count == 2
    assert score.duplicate_count == 1
    assert score.raw_score == 1.0


@pytest.mark.parametrize("bits", [0, 1, 129])
def test_signature_bits_must_be_between_two_and_128(bits: int) -> None:
    with pytest.raises(ValueError, match="signature_bits"):
        CanonicalSignatureScorer(
            verifier=_FakeVerifier({}),
            secret_key="key",
            signature_bits=bits,
        )
