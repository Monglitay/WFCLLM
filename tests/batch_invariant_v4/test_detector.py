from __future__ import annotations

from dataclasses import replace

import pytest

from wfcllm.batch_invariant_v4.calibration import CalibrationArtifact
from wfcllm.batch_invariant_v4.channel import StructuralChannel
from wfcllm.batch_invariant_v4.context import ContextConfig, StructuralContextExtractor
from wfcllm.batch_invariant_v4.detector import (
    DetectorPayload,
    V4Detector,
    exact_code_evidence_mismatches,
)
from wfcllm.batch_invariant_v4.keying import V4SecretKey


def _detector() -> V4Detector:
    calibration = CalibrationArtifact.from_scores_for_test(
        tuple(value / 20 for value in range(-20, 21)),
        target_fpr=0.05,
        minimum_independent_units=3,
    )
    return V4Detector(
        extractor=StructuralContextExtractor(ContextConfig()),
        channel=StructuralChannel(
            V4SecretKey.from_material_for_test(b"d" * 32),
            bit_count=32,
            minimum_independent_units=3,
        ),
        calibration=calibration,
    )


@pytest.mark.parametrize(
    "forbidden",
    [
        "prompt",
        "task_id",
        "attempt_index",
        "generation_ledger",
        "stored_embedding",
        "candidate_pool",
        "batch_metadata",
    ],
)
def test_r3_payload_rejects_every_forbidden_input(forbidden: str) -> None:
    with pytest.raises(ValueError, match="exactly final_code"):
        DetectorPayload.from_dict({"final_code": "def f():\n return 1\n", forbidden: 1})


def test_r3_payload_accepts_only_final_code() -> None:
    payload = DetectorPayload.from_dict({"final_code": "def f():\n return 1\n"})
    assert payload.final_code.startswith("def f")
    assert tuple(payload.__dict__) == ("final_code",)


def test_same_final_code_produces_all_exact_discrete_fields() -> None:
    code = (
        "def f(values):\n"
        "    total = 0\n"
        "    for value in values:\n"
        "        total += value\n"
        "    return total\n"
    )
    detector = _detector()

    generation = detector.detect(DetectorPayload(final_code=code))
    r3 = detector.detect(DetectorPayload.from_dict({"final_code": code}))

    assert exact_code_evidence_mismatches(generation, r3) == ()
    assert generation == r3
    assert generation.independent_units >= 3
    unit = generation.units[0]
    assert unit.representation == unit.quantized_values
    assert unit.erasure_mask == (False,) * 32
    assert len(unit.signature_bits) == len(unit.target_bits) == 32
    assert unit.matches == sum(
        left == right
        for left, right in zip(unit.signature_bits, unit.target_bits, strict=True)
    )
    assert unit.numerator == 2 * unit.matches - unit.denominator


def test_exact_comparator_names_representation_quantized_erasure_and_decision() -> None:
    detector = _detector()
    evidence = detector.detect(
        DetectorPayload(final_code="def f():\n    a = 1\n    b = 2\n    return a + b\n")
    )
    first = evidence.units[0]
    changed_unit = replace(
        first,
        representation=(first.representation[0] ^ 1,) + first.representation[1:],
        quantized_values=(first.quantized_values[0] ^ 1,) + first.quantized_values[1:],
        erasure_mask=(True,) + first.erasure_mask[1:],
    )
    changed = replace(evidence, units=(changed_unit,) + evidence.units[1:], decision=not evidence.decision)

    assert exact_code_evidence_mismatches(evidence, changed) == (
        "units[0].representation",
        "units[0].quantized_values",
        "units[0].erasure_mask",
        "decision",
    )


def test_unit_ids_are_sorted_and_no_float_tolerance_path_exists() -> None:
    evidence = _detector().detect(
        DetectorPayload(final_code="def f():\n    z = 3\n    a = 1\n    return z + a\n")
    )

    assert tuple(unit.unit_id for unit in evidence.units) == tuple(
        sorted(unit.unit_id for unit in evidence.units)
    )
    assert not hasattr(evidence, "cosine_similarity")
