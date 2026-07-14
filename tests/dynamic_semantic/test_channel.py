from __future__ import annotations

import pytest

from wfcllm.dynamic_semantic.channel import (
    SemanticChannel,
    UnitEvidence,
    aggregate_unit_evidence,
    hamming_7_4_encode,
    hamming_7_4_syndrome,
    quantize_vector,
)
from wfcllm.dynamic_semantic.config import ChannelConfig
from wfcllm.dynamic_semantic.keying import SecretKey


def _channel(material: bytes = b"a" * 32) -> SemanticChannel:
    return SemanticChannel(
        SecretKey.from_material_for_test(material),
        ChannelConfig(
            whitening_dimensions=4,
            quantization_scale=16,
            projection_rows=7,
            target_data_bits=4,
            minimum_independent_units=1,
        ),
    )


def test_hamming_7_4_known_codeword_and_syndrome() -> None:
    codeword = hamming_7_4_encode((1, 0, 1, 1))

    assert codeword == (0, 1, 1, 0, 0, 1, 1)
    assert hamming_7_4_syndrome(codeword) == 0
    corrupted = list(codeword)
    corrupted[2] ^= 1
    assert hamming_7_4_syndrome(tuple(corrupted)) == 3


def test_quantization_is_symmetric_half_away_from_zero() -> None:
    assert quantize_vector((0.0, 0.5, -0.5, 0.03125, -0.03125), 16) == (
        0,
        8,
        -8,
        1,
        -1,
    )


def test_channel_is_deterministic_and_uses_post_embedding_secret() -> None:
    channel = _channel()
    vector = (0.25, -0.5, 0.75, 0.125)

    first = channel.score("unit-a", "context-a", vector)
    second = channel.score("unit-a", "context-a", vector)

    assert first == second
    assert first.quantized_embedding == (4, -8, 12, 2)
    assert len(first.signature_bits) == 7
    assert len(first.target_bits) == 7
    assert first.denominator == 7
    assert first.numerator == 2 * first.matches - 7


def test_target_is_content_addressed_but_not_context_addressed(monkeypatch) -> None:
    from wfcllm.dynamic_semantic import channel as channel_module

    real_derive = channel_module.derive_bytes
    target_messages: list[bytes] = []

    def recording_derive(key, domain: str, message: bytes, length: int) -> bytes:
        if domain == "unit-target":
            target_messages.append(message)
        return real_derive(key, domain, message, length)

    monkeypatch.setattr(channel_module, "derive_bytes", recording_derive)
    channel = _channel()
    vector = (0.25, -0.5, 0.75, 0.125)

    first = channel.score("same-unit", "context-a", vector)
    second = channel.score("same-unit", "context-b", vector)
    other = channel.score("other-unit", "context-a", vector)

    assert first.target_bits == second.target_bits
    assert first.signature_bits == second.signature_bits
    assert target_messages == [b"same-unit", b"same-unit", b"other-unit"]


def test_projection_rejects_wrong_embedding_dimensions() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        _channel().score("unit", "context", (0.1, 0.2))


def test_aggregate_statistic_is_exact_and_rejects_duplicate_units() -> None:
    evidence = (
        UnitEvidence.synthetic("a", matches=7),
        UnitEvidence.synthetic("b", matches=3),
        UnitEvidence.synthetic("c", matches=0),
    )

    result = aggregate_unit_evidence(evidence, minimum_independent_units=3)

    assert result.numerator == 7 - 1 - 7
    assert result.denominator == 21
    assert result.independent_units == 3
    assert result.score == pytest.approx(-1 / 21)
    assert result.eligible is True

    with pytest.raises(ValueError, match="duplicate"):
        aggregate_unit_evidence(
            (evidence[0], evidence[0]),
            minimum_independent_units=1,
        )


def test_aggregate_marks_sparse_code_ineligible_without_padding() -> None:
    result = aggregate_unit_evidence(
        (UnitEvidence.synthetic("a", matches=7),),
        minimum_independent_units=3,
    )

    assert result.numerator == 7
    assert result.denominator == 7
    assert result.independent_units == 1
    assert result.eligible is False
