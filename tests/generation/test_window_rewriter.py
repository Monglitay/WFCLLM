from __future__ import annotations

from wfcllm.gate.data import RewriteRequest, StructuralBoundary
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    KeyBlindWhitespaceWindowRewriter,
    RewriteGeneration,
)


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


class BatchGenerator(Generator):
    def generate_windows(self, **kwargs):
        self.calls.append(kwargs)
        return tuple(
            RewriteGeneration(
                (index,),
                f"x = {index}\ny = {index + 1}\n",
                f"seed-{index}",
                "cfg-batch",
            )
            for index in kwargs["candidate_indices"]
        )


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


def test_causal_rewriter_batches_three_candidates_in_one_backend_call() -> None:
    backend = BatchGenerator()

    results = CausalWindowRewriter(backend).rewrite_many(
        _request(), candidate_indices=(1, 2, 3)
    )

    assert [result.code for result in results] == [
        "x = 1\ny = 2\n",
        "x = 2\ny = 3\n",
        "x = 3\ny = 4\n",
    ]
    assert len(backend.calls) == 1
    assert backend.calls[0]["candidate_indices"] == (1, 2, 3)


def test_causal_rewriter_keeps_requested_window_length_from_attempt_pool() -> None:
    class AttemptPoolBackend(Generator):
        def generate_windows(self, **kwargs):
            self.calls.append(kwargs)
            texts = (
                "",
                "x = 2\ny = 3\n",
                "a=1\nb=2\nc=3\nd=4\n",
                "x = 4\n",
                "if :\n",
                "x = 6\n",
            )
            return tuple(
                RewriteGeneration((index,), text, f"seed-{index}", "cfg-pool")
                for index, text in enumerate(texts, 1)
            )

    backend = AttemptPoolBackend()
    rewriter = CausalWindowRewriter(backend, generation_attempts=6)

    results = rewriter.rewrite_many(
        _request(), candidate_indices=(1, 2, 3)
    )

    assert results[0].code == "x = 2\ny = 3\n"
    assert all(result.unit_count == 2 for result in results)
    assert all(result.parse_status == "ok" for result in results)
    assert backend.calls[0]["candidate_indices"] == (1, 2, 3, 4, 5, 6)


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


def test_in_range_but_wrong_window_length_is_rejected() -> None:
    class ThreeUnits:
        def generate_window(self, **kwargs):
            return RewriteGeneration(
                (1,),
                "a = 1\nb = 2\nc = 3\n",
                "seed",
                "cfg",
            )

    result = CausalWindowRewriter(ThreeUnits()).rewrite(
        _request(), candidate_index=1
    )

    assert result.unit_count == 3
    assert result.parse_status == "unit_count_out_of_range"


def test_unique_structural_fallback_replaces_duplicate_model_outputs() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="",
        original_window="foo_bar = []\nprint(foo_bar)\n",
        canonical_parent="python-statement-window/v1||parent=module|ordinal=0|role=body",
        window_start_unit_id="0",
        window_length=2,
        structural_boundary=StructuralBoundary(
            0, 31, 0, "module", ("0", "1"), False, False
        ),
    )

    class DuplicatePool(Generator):
        def generate_windows(self, **kwargs):
            self.calls.append(kwargs)
            return tuple(
                RewriteGeneration((index,), "foo_bar = []\nprint(foo_bar)\n", f"seed-{index}", "cfg")
                for index in kwargs["candidate_indices"]
            )

    results = CausalWindowRewriter(
        DuplicatePool(),
        generation_attempts=6,
        unique_structural_fallback=True,
    ).rewrite_many(request, candidate_indices=(1, 2, 3))

    assert len({result.code for result in results}) == 3
    assert all(result.code != request.original_window for result in results)
    assert all(result.parse_status == "ok" for result in results)
    assert all("public-structural-fallback/v1" in result.rewrite_config_id for result in results)


def test_key_blind_whitespace_rewriter_emits_distinct_parser_stable_variants() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="",
        original_window="value = item + 1\n",
        canonical_parent="python-statement-window/v1||parent=module|ordinal=0|role=body",
        window_start_unit_id="0",
        window_length=1,
        structural_boundary=StructuralBoundary(
            0, 16, 0, "module", ("0",), False, False
        ),
    )
    rewriter = KeyBlindWhitespaceWindowRewriter()

    variants = tuple(
        rewriter.rewrite_window(request, candidate_index=index)
        for index in (1, 2, 3)
    )

    assert len({variant.text for variant in variants}) == 3
    assert all(variant.text != request.original_window for variant in variants)
    assert all(variant.text.endswith("\n") for variant in variants)
    assert all(
        "public-key-blind-whitespace/v1" in variant.rewrite_config_id
        for variant in variants
    )


def test_key_blind_whitespace_rewriter_preserves_bare_return_semantics() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="def f():\n",
        original_window="    return\n",
        canonical_parent=(
            "python-statement-window/v1|module/function_definition/block|"
            "parent=block|ordinal=0|role=body"
        ),
        window_start_unit_id="0",
        window_length=1,
        structural_boundary=StructuralBoundary(
            9, 19, 1, "block", ("0",), False, False
        ),
    )

    variant = KeyBlindWhitespaceWindowRewriter().rewrite_window(
        request, candidate_index=2
    )

    assert variant.text == "    return  None\n"
