from __future__ import annotations

import pytest

from wfcllm.batch_invariant_v4.calibration import CalibrationArtifact
from wfcllm.batch_invariant_v4.channel import StructuralChannel
from wfcllm.batch_invariant_v4.context import ContextConfig, StructuralContextExtractor
from wfcllm.batch_invariant_v4.detector import DetectorPayload, V4Detector
from wfcllm.batch_invariant_v4.keying import V4SecretKey


FAILURE_CODES = {
    "HumanEval/34-unit-3": (
        "def f(values):\n"
        "    unique_set = set(values)\n"
        "    unique_list = list(unique_set)\n"
        "    unique_list.sort()\n"
        "    return unique_list\n"
    ),
    "HumanEval/39-unit-6": (
        "def f(n):\n"
        "    i = 0\n"
        "    while True:\n"
        "        fib_num = fib(i)\n"
        "        if is_prime(fib_num):\n"
        "            i += 1\n"
        "        if i == n:\n"
        "            break\n"
        "    return fib_num\n"
    ),
    "HumanEval/71-unit-2": (
        "def f(a, b, c):\n"
        "    if a + b <= c or a + c <= b or b + c <= a:\n"
        "        return -1\n"
        "    s = (a + b + c) / 2\n"
        "    return (s * (s-a) * (s-b) * (s-c)) ** 0.5\n"
    ),
}


def _detector() -> V4Detector:
    return V4Detector(
        extractor=StructuralContextExtractor(ContextConfig()),
        channel=StructuralChannel(
            V4SecretKey.from_material_for_test(b"g" * 32),
            bit_count=32,
            minimum_independent_units=1,
        ),
        calibration=CalibrationArtifact.from_scores_for_test(
            (-1.0, 0.0, 1.0),
            target_fpr=0.05,
            minimum_independent_units=1,
        ),
    )


@pytest.mark.parametrize("case_id", tuple(FAILURE_CODES))
def test_v3_failure_context_is_exact_across_caller_batch_conditions(case_id: str) -> None:
    detector = _detector()
    target = FAILURE_CODES[case_id]
    reference = detector.detect(DetectorPayload(final_code=target))
    distractors = [
        "def short():\n return 1\n",
        "def long(values):\n    total = 0\n    for value in values:\n        total += value\n    return total\n",
        FAILURE_CODES["HumanEval/34-unit-3"],
        FAILURE_CODES["HumanEval/39-unit-6"],
        FAILURE_CODES["HumanEval/71-unit-2"],
    ]

    for batch_size in (1, 2, 4, 8, 16, 32):
        for ordered in (distractors, list(reversed(distractors))):
            for code in (ordered * batch_size)[:batch_size]:
                detector.detect(DetectorPayload(final_code=code))
            assert detector.detect(DetectorPayload(final_code=target)) == reference
