from __future__ import annotations

import pytest

from scripts.wfcllm_v4_pilot import (
    _bootstrap_ci,
    _mcnemar,
    _two_sided_binomial_p,
)


def test_bootstrap_constant_paired_delta_is_exact() -> None:
    assert _bootstrap_ci((1.0, 1.0, 1.0), seed=7, repetitions=100) == [1.0, 1.0]


def test_exact_binomial_and_mcnemar_handle_ties_and_discordance() -> None:
    assert _two_sided_binomial_p(0, 0) == 1.0
    assert _two_sided_binomial_p(0, 4) == pytest.approx(0.125)
    assert _mcnemar((True, False, True), (True, False, True)) == {
        "current_only": 0,
        "v4_only": 0,
        "both": 2,
        "neither": 1,
        "exact_two_sided_p": 1.0,
    }
