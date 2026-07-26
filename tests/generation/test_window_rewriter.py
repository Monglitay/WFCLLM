from __future__ import annotations

from wfcllm.gate.data import RewriteRequest, StructuralBoundary
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    KeyBlindAstEquivalentWindowRewriter,
    KeyBlindWhitespaceWindowRewriter,
    RewriteGeneration,
    python_ast_equivalent,
    python_comprehension_alpha_equivalent,
    python_literal_equivalent,
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


def test_causal_rewriter_rejects_hidden_candidate_pool_size() -> None:
    import pytest

    with pytest.raises(ValueError, match="three public candidates"):
        CausalWindowRewriter(Generator(), generation_attempts=6)


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


def test_causal_rewriter_preserves_invalid_candidate_at_original_index() -> None:
    class AttemptPoolBackend(Generator):
        def generate_windows(self, **kwargs):
            self.calls.append(kwargs)
            texts = ("", "x = 2\ny = 3\n", "a=1\nb=2\nc=3\nd=4\n")
            return tuple(
                RewriteGeneration((index,), text, f"seed-{index}", "cfg-pool")
                for index, text in enumerate(texts, 1)
            )

    backend = AttemptPoolBackend()
    rewriter = CausalWindowRewriter(backend, generation_attempts=3)

    results = rewriter.rewrite_many(
        _request(), candidate_indices=(1, 2, 3)
    )

    assert [result.parse_status for result in results] == [
        "parse_error", "ok", "unit_count_out_of_range"
    ]
    assert backend.calls[0]["candidate_indices"] == (1, 2, 3)


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


def test_causal_rewriter_preserves_duplicate_outputs_without_replacement() -> None:
    class DuplicatePool(Generator):
        def generate_windows(self, **kwargs):
            self.calls.append(kwargs)
            return tuple(
                RewriteGeneration((index,), "foo_bar = []\nprint(foo_bar)\n", f"seed-{index}", "cfg")
                for index in kwargs["candidate_indices"]
            )

    results = CausalWindowRewriter(DuplicatePool()).rewrite_many(
        _request(), candidate_indices=(1, 2, 3)
    )

    assert [result.code for result in results] == [
        "foo_bar = []\nprint(foo_bar)\n",
        "foo_bar = []\nprint(foo_bar)\n",
        "foo_bar = []\nprint(foo_bar)\n",
    ]
    assert [result.generation_seed_id for result in results] == [
        "seed-1", "seed-2", "seed-3"
    ]


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


def test_ast_equivalent_rewriter_emits_three_distinct_certified_variants() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="def f():\n",
        original_window=(
            "    label = 'ready'\n"
            "    count = 10\n"
            "    return label, count\n"
        ),
        canonical_parent=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=0|role=body"
        ),
        window_start_unit_id="0",
        window_length=3,
        structural_boundary=StructuralBoundary(
            9, 64, 1, "block", ("0", "1", "2"), False, False
        ),
    )

    variants = KeyBlindAstEquivalentWindowRewriter().rewrite_windows(
        request, candidate_indices=(1, 2, 3)
    )

    assert len({variant.text for variant in variants}) == 3
    assert all(variant.text != request.original_window for variant in variants)
    assert all(variant.semantic_validation_rule == "python-ast-equivalent/v1" for variant in variants)
    assert all(variant.semantic_equivalence_certified is True for variant in variants)
    assert all(
        python_ast_equivalent(request.original_window, variant.text)
        for variant in variants
    )


def test_ast_equivalence_rejects_behavior_changes_seen_in_attempt14() -> None:
    assert python_ast_equivalent("arr.sort()\n", "arr.sort(reverse=True)\n") is False
    assert python_ast_equivalent("return y\n", "return y - 1\n") is False
    assert python_ast_equivalent(
        "for i in range(2, n):\n    total += i\n",
        "for i in range(2, n + 1):\n    total += i\n",
    ) is False


def test_ast_equivalent_rewriter_preserves_indentation_and_window_length() -> None:
    variants = KeyBlindAstEquivalentWindowRewriter().rewrite_windows(
        _request(), candidate_indices=(1, 2, 3)
    )

    assert all(variant.parse_status == "ok" for variant in variants)
    assert all(variant.unit_count == 2 for variant in variants)
    assert all(variant.same_parent_scope is True for variant in variants)


def test_ast_rewriter_handles_source_slice_with_first_indent_in_prefix() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="def f(nums):\n    ",
        original_window=(
            '"""Filter the even numbers.\n'
            "    Return a list.\n"
            '    """\n'
            "    return list(filter(lambda x: x % 2 == 0, nums))\n"
        ),
        canonical_parent=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=0|role=body"
        ),
        window_start_unit_id="0",
        window_length=2,
        structural_boundary=StructuralBoundary(
            17, 134, 1, "block", ("0", "1"), False, False
        ),
    )

    variants = KeyBlindAstEquivalentWindowRewriter().rewrite_windows(
        request,
        candidate_indices=(1, 2, 3),
    )

    assert len({variant.text for variant in variants}) == 3
    assert all(variant.text != request.original_window for variant in variants)
    assert all(variant.parse_status == "ok" for variant in variants)
    assert all(variant.unit_count == 2 for variant in variants)
    assert all(variant.same_parent_scope is True for variant in variants)
    assert all(variant.semantic_equivalence_certified is True for variant in variants)
    assert all(
        python_ast_equivalent(request.original_window, variant.text)
        for variant in variants
    )


def test_ast_rewriter_certifies_compound_header_with_synthetic_body() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="def f(nums):\n    ",
        original_window="for num in nums:",
        canonical_parent=(
            "python-statement-window/v1|module/function_definition|"
            "parent=block|ordinal=0|role=header"
        ),
        window_start_unit_id="0",
        window_length=1,
        structural_boundary=StructuralBoundary(
            17, 33, 1, "block", ("0",), False, False
        ),
    )

    variants = KeyBlindAstEquivalentWindowRewriter().rewrite_windows(
        request,
        candidate_indices=(1, 2, 3),
    )

    assert len({variant.text for variant in variants}) == 3
    assert all(variant.text != request.original_window for variant in variants)
    assert all(variant.parse_status == "ok" for variant in variants)
    assert all(variant.same_parent_scope is True for variant in variants)
    assert all(variant.semantic_equivalence_certified is True for variant in variants)
    assert all(
        python_ast_equivalent(request.original_window, variant.text)
        for variant in variants
    )


def test_ast_equivalent_rewriter_exposes_gate_data_candidate_interface() -> None:
    results = KeyBlindAstEquivalentWindowRewriter().rewrite_many(
        _request(), candidate_indices=(1, 2, 3)
    )

    assert len(results) == 3
    assert all(result.parse_status == "ok" for result in results)
    assert all(result.unit_count == 2 for result in results)
    assert all(result.same_parent_scope is True for result in results)


def test_ast_equivalent_rewriter_supports_bounded_online_retry_beyond_r3() -> None:
    rewriter = KeyBlindAstEquivalentWindowRewriter()

    variants = tuple(
        rewriter.rewrite_window(_request(), candidate_index=index)
        for index in range(1, 13)
    )

    assert len({variant.text for variant in variants}) == 12
    assert all(variant.semantic_equivalence_certified is True for variant in variants)
    assert variants[-1].rewrite_config_id.endswith(":12")
    assert all(variant.text.endswith("\n") for variant in variants)


def test_rewriter_uses_certified_comprehension_alpha_variants_after_r3() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="def f(items):\n",
        original_window=(
            "    positives = [item * item for item in items if item > 0]\n"
            "    return positives\n"
        ),
        canonical_parent=(
            "python-statement-window/v1|module/function_definition/block|"
            "parent=block|ordinal=0|role=body"
        ),
        window_start_unit_id="0",
        window_length=2,
        structural_boundary=StructuralBoundary(
            14, 93, 1, "block", ("0", "1"), False, False
        ),
    )
    rewriter = KeyBlindAstEquivalentWindowRewriter()

    fourth = rewriter.rewrite_window(request, candidate_index=4)
    fifth = rewriter.rewrite_window(request, candidate_index=5)

    assert fourth.text != fifth.text
    assert "_wfcllm_comp_4" in fourth.text
    assert "_wfcllm_comp_5" in fifth.text
    assert fourth.semantic_validation_rule == "python-comprehension-alpha-equivalent/v1"
    assert fourth.semantic_equivalence_certified is True
    assert fifth.semantic_equivalence_certified is True
    assert python_comprehension_alpha_equivalent(
        request.original_window, fourth.text
    )


def test_comprehension_alpha_prover_rejects_behavior_changes() -> None:
    reference = "return [item * item for item in items if item > 0]\n"

    assert python_comprehension_alpha_equivalent(
        reference,
        "return [renamed * renamed for renamed in items if renamed > 0]\n",
    )
    assert not python_comprehension_alpha_equivalent(
        reference,
        "return [renamed + renamed for renamed in items if renamed > 0]\n",
    )
    assert not python_comprehension_alpha_equivalent(
        reference,
        "return [renamed * renamed for renamed in reversed(items) if renamed > 0]\n",
    )


def test_literal_equivalence_prover_accepts_identities_and_rejects_changes() -> None:
    assert python_literal_equivalent("return 10\n", "return 10 + 0\n") is True
    assert python_literal_equivalent("return 'ok'\n", "return 'ok' + ''\n") is True
    assert python_literal_equivalent("return y\n", "return y - 1\n") is False
    assert python_literal_equivalent("'doc'\n", "'doc' + ''\n") is False


def test_literal_equivalent_rewriter_adds_distinct_semantic_variants_after_r12() -> None:
    rewriter = KeyBlindAstEquivalentWindowRewriter()
    variants = tuple(
        rewriter.rewrite_window(_request(), candidate_index=index)
        for index in range(13, 19)
    )

    assert len({variant.text for variant in variants}) == 6
    assert all(
        variant.semantic_validation_rule == "python-literal-equivalent/v1"
        for variant in variants
    )
    assert all(variant.semantic_equivalence_certified is True for variant in variants)
    assert all(
        python_literal_equivalent(_request().original_window, variant.text)
        for variant in variants
    )
    assert all(variant.text.endswith("\n") for variant in variants)


def test_literal_equivalent_trajectory_advances_to_next_literal_after_six_modes() -> None:
    rewriter = KeyBlindAstEquivalentWindowRewriter()
    first_literal = tuple(
        rewriter.rewrite_window(_request(), candidate_index=index)
        for index in range(13, 19)
    )
    second_literal = tuple(
        rewriter.rewrite_window(_request(), candidate_index=index)
        for index in range(19, 25)
    )

    assert len({variant.text for variant in second_literal}) == 6
    assert {variant.text for variant in first_literal}.isdisjoint(
        variant.text for variant in second_literal
    )
    assert all("x = 1\n" in variant.text for variant in second_literal)
    assert all(variant.semantic_equivalence_certified is True for variant in second_literal)


def test_literal_equivalent_rewriter_fails_closed_on_fstrings() -> None:
    request = RewriteRequest(
        prompt="",
        completed_prefix="",
        original_window='result = f"{value:02d}"\n',
        canonical_parent="python-statement-window/v1||parent=module|ordinal=0|role=body",
        window_start_unit_id="0",
        window_length=1,
        structural_boundary=StructuralBoundary(
            0, 24, 0, "module", ("0",), False, False
        ),
    )

    variant = KeyBlindAstEquivalentWindowRewriter().rewrite_window(
        request, candidate_index=13
    )

    assert variant.text == request.original_window
    assert variant.semantic_equivalence_certified is False
