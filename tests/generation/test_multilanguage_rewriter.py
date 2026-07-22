from __future__ import annotations

from wfcllm.gate.data import RewriteRequest, StructuralBoundary
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    RewriteGeneration,
)
from wfcllm.windowing import get_statement_unit_extractor


class _CppBackend:
    def generate_window(self, **kwargs) -> RewriteGeneration:
        return RewriteGeneration(
            token_ids=(1, 2, 3),
            text="int value = 2;",
            generation_seed_id="cpp-seed",
            rewrite_config_id="cpp-model-semantic/v1",
        )


def test_causal_rewriter_rebinds_parser_and_contract_for_cpp() -> None:
    extractor = get_statement_unit_extractor("cpp")
    unit = extractor.extract("int value = 1;")[0]
    canonical_parent = (
        "cpp-statement-window/v1||parent=translation_unit|ordinal=0|role=body"
    )
    request = RewriteRequest(
        prompt="rewrite",
        completed_prefix="",
        original_window=unit.text,
        canonical_parent=canonical_parent,
        window_start_unit_id=unit.unit_id,
        window_length=1,
        structural_boundary=StructuralBoundary(
            start_byte=unit.start_byte,
            end_byte=unit.end_byte,
            depth=unit.depth,
            direct_parent_type=unit.direct_parent_type,
            unit_ids=(unit.unit_id,),
            compound_singleton=False,
            hard_boundary_after=False,
        ),
    )
    base = CausalWindowRewriter(_CppBackend())
    rewriter = base.for_extractor(
        extractor,
        window_contract_version="cpp-statement-window/v1",
    )

    parsed = rewriter.rewrite_window(request, candidate_index=1)

    assert parsed.parse_status == "ok"
    assert parsed.same_parent_scope is True
    assert parsed.parent_descriptor == canonical_parent
    assert parsed.rewrite_config_id == "cpp-model-semantic/v1"
