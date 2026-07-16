from __future__ import annotations

from wfcllm.gate.data import RewriteRequest, StructuralBoundary
from wfcllm.generation.window_rewriter import CausalWindowRewriter, RewriteGeneration


def _request() -> RewriteRequest:
    return RewriteRequest(
        prompt="def f():\n",
        completed_prefix="",
        original_window="x = 1\ny = 2\n",
        canonical_parent="python-statement-window/v1||parent=module|ordinal=0|role=body",
        window_start_unit_id="0",
        window_length=2,
        structural_boundary=StructuralBoundary(
            0, 12, 0, "module", ("0", "1"), False, False
        ),
    )


class Generator:
    def __init__(self) -> None:
        self.calls = []

    def generate_window(self, **kwargs):
        self.calls.append(kwargs)
        return RewriteGeneration((1, 2), "x = 3\ny = 4\n", "seed-1", "cfg-1")


def test_causal_rewriter_returns_complete_parser_checked_window() -> None:
    backend = Generator()
    result = CausalWindowRewriter(backend).rewrite(_request(), candidate_index=1)
    assert result.parse_status == "ok"
    assert result.unit_count == 2
    assert result.same_parent_scope is True
    assert result.code == "x = 3\ny = 4\n"
    assert set(backend.calls[0]) == {
        "prompt", "completed_prefix", "original_window", "candidate_index", "max_units"
    }


def test_gate_data_collects_all_six_without_early_stop() -> None:
    backend = Generator()
    rewriter = CausalWindowRewriter(backend)
    results = [rewriter.rewrite(_request(), candidate_index=index) for index in range(1, 7)]
    assert len(results) == 6
    assert len(backend.calls) == 6


def test_zero_and_four_unit_outputs_are_rejected() -> None:
    class Bad:
        def __init__(self, text: str):
            self.text = text

        def generate_window(self, **kwargs):
            return RewriteGeneration((1,), self.text, "seed", "cfg")

    assert CausalWindowRewriter(Bad("")).rewrite(_request(), candidate_index=1).parse_status == "parse_error"
    four = "a=1\nb=2\nc=3\nd=4\n"
    assert CausalWindowRewriter(Bad(four)).rewrite(_request(), candidate_index=1).unit_count == 4
    assert CausalWindowRewriter(Bad(four)).rewrite(_request(), candidate_index=1).parse_status == "unit_count_out_of_range"
