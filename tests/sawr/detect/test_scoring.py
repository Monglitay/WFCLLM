from __future__ import annotations

import pytest

from wfcllm.sawr.detect.config import SawrDetectionConfig
from wfcllm.sawr.detect.proxy_windows import ProxyWindow
from wfcllm.sawr.detect.scoring import SawrWindowScorer, WindowEvidence
from wfcllm.sawr.boundary import Candidate


class FakeVerifyResult:
    def __init__(
        self,
        *,
        passed: bool,
        signature: tuple[int, ...],
        min_margin: float,
        in_valid_set: bool,
    ) -> None:
        self.passed = passed
        self.lsh_signature = signature
        self.min_margin = min_margin
        self.in_valid_set = in_valid_set


class FakeVerifier:
    def __init__(self, result: FakeVerifyResult) -> None:
        self.result = result
        self.calls: list[tuple[str, frozenset[tuple[int, ...]], float]] = []

    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> FakeVerifyResult:
        self.calls.append((code_text, valid_set, margin))
        return self.result


class FakeKeying:
    def __init__(self) -> None:
        self.valid_set = frozenset({(1, 0, 1, 0)})
        self.calls: list[tuple[str, int, int | None]] = []

    def derive(
        self,
        parent_node_type: str,
        *,
        k: int,
        ordinal: int | None,
    ) -> frozenset[tuple[int, ...]]:
        self.calls.append((parent_node_type, k, ordinal))
        return self.valid_set


def _window() -> ProxyWindow:
    candidate = Candidate(
        text="return x",
        candidate_type="proxy_window_statement",
        node_type="return_statement",
        position_id="module.target.body",
        token_start_idx=0,
        token_count=0,
        parent_node_type="function_definition",
        ordinal=7,
    )
    return ProxyWindow(
        context_id="module.target.body",
        window_id="module.target.body.window:0",
        normalized_text="return x",
        candidates=(candidate,),
        parent_node_type="function_definition",
        structure_type="function_body",
        window_length=1,
        context_statement_count=1,
        context_window_count=1,
        ordinal=7,
        start_line=2,
        end_line=2,
    )


def test_window_scorer_uses_keyed_lsh_with_ordinal() -> None:
    verifier = FakeVerifier(
        FakeVerifyResult(
            passed=True,
            signature=(1, 0, 1, 0),
            min_margin=0.42,
            in_valid_set=True,
        )
    )
    keying = FakeKeying()
    config = SawrDetectionConfig(
        secret_key="1010",
        lsh_d=4,
        gamma=0.75,
        semantic_margin=0.03,
        use_ordinal_keying=True,
    )
    scorer = SawrWindowScorer(verifier=verifier, keying=keying, config=config)

    evidence = scorer.score_window(_window())

    assert evidence == WindowEvidence(
        window_id="module.target.body.window:0",
        context_id="module.target.body",
        in_valid_set=True,
        passed_margin=True,
        min_margin=0.42,
        lsh_signature=(1, 0, 1, 0),
        parent_node_type="function_definition",
        window_length=1,
        structure_type="function_body",
        context_window_count=1,
        context_statement_count=1,
        window_raw=0.42,
    )
    assert keying.calls == [("function_definition", 12, 7)]
    assert verifier.calls == [("return x", keying.valid_set, 0.03)]


def test_window_scorer_omits_ordinal_keying_by_default() -> None:
    verifier = FakeVerifier(
        FakeVerifyResult(
            passed=True,
            signature=(1, 0, 1, 0),
            min_margin=0.42,
            in_valid_set=True,
        )
    )
    keying = FakeKeying()
    scorer = SawrWindowScorer(
        verifier=verifier,
        keying=keying,
        config=SawrDetectionConfig(secret_key="1010"),
    )

    scorer.score_window(_window())

    assert keying.calls == [("function_definition", 12, None)]


def test_hit_only_evidence_uses_binary_hit() -> None:
    scorer = SawrWindowScorer(
        verifier=FakeVerifier(
            FakeVerifyResult(
                passed=True,
                signature=(1, 0, 1, 0),
                min_margin=0.02,
                in_valid_set=True,
            )
        ),
        keying=FakeKeying(),
        config=SawrDetectionConfig(
            secret_key="1010",
            evidence_mode="hit_only",
            semantic_margin=0.01,
        ),
    )

    assert scorer.score_window(_window()).window_raw == pytest.approx(1.0)


def test_window_scorer_zeroes_invalid_hits() -> None:
    scorer = SawrWindowScorer(
        verifier=FakeVerifier(
            FakeVerifyResult(
                passed=False,
                signature=(0, 0, 0, 0),
                min_margin=0.42,
                in_valid_set=False,
            )
        ),
        keying=FakeKeying(),
        config=SawrDetectionConfig(secret_key="1010", semantic_margin=0.01),
    )

    evidence = scorer.score_window(_window())

    assert evidence.in_valid_set is False
    assert evidence.passed_margin is True
    assert evidence.window_raw == pytest.approx(0.0)
