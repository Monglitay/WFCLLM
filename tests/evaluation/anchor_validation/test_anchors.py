from __future__ import annotations

from wfcllm.evaluation.anchor_validation.anchors import (
    build_anchor_text,
    infer_semantic_role,
    mask_code_skeleton,
)
from wfcllm.evaluation.anchor_validation.schema import (
    AnchorMethod,
    CandidateBlock,
    CandidateContext,
)


def _context() -> CandidateContext:
    return CandidateContext(
        context_id="ctx",
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        function_signature="def f(x):",
        ast_path=("function_definition", "return_statement"),
        node_type="return_statement",
        parent_node_type="function_definition",
        block_ordinal=0,
        context_hash="ctxhash",
        context_before="def f(x):\n",
        context_after="",
        masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
        import_and_helper_signatures=("import math", "def helper(v):"),
        temperature=0.2,
        candidates=(CandidateBlock("c0", "return x + 1", 0),),
    )


def test_mask_code_skeleton_masks_identifiers_and_literals():
    assert mask_code_skeleton("total = x + 42") == "<NAME> = <NAME> + <NUMBER>"


def test_mask_code_skeleton_falls_back_on_indentation_errors():
    source = "if x:\n    y = 1\n  z = 2\n"

    assert mask_code_skeleton(source) == source.strip()


def test_slot_context_anchor_is_prompt_free_and_does_not_include_secret_key():
    text = build_anchor_text(
        AnchorMethod.SLOT_CONTEXT,
        _context(),
        _context().candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "HumanEval/0" in text
    assert "return x + 1" not in text
    assert "def f(<NAME>):" in text


def test_context_anchor_uses_surrounding_context_not_only_signature():
    slot = build_anchor_text(
        AnchorMethod.SLOT,
        _context(),
        _context().candidates[0],
    )
    context = build_anchor_text(
        AnchorMethod.CONTEXT,
        _context(),
        _context().candidates[0],
    )

    assert context != slot
    assert "<TARGET_BLOCK>" in context


def test_infer_semantic_role_from_node_type_and_block_text():
    assert infer_semantic_role("return total", "return_statement", "function_definition") == "return final value"
    assert infer_semantic_role("if n <= 1:", "if_statement", "function_definition") == "branch condition / guard"
    assert infer_semantic_role("total += item", "expression_statement", "for_statement") == "accumulator update"
    assert infer_semantic_role("items.append(x)", "expression_statement", "for_statement") == "function call side effect"
    assert infer_semantic_role("import math", "import_statement", "module") == "import/dependency"
    assert infer_semantic_role("assert value >= 0", "assert_statement", "function_definition") == "assertion/invariant check"
    assert infer_semantic_role("raise ValueError('bad')", "raise_statement", "except_clause") == "exception/error handling"


def test_role_aware_anchor_includes_role_and_structural_context_without_raw_candidate():
    text = build_anchor_text(
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT,
        _context(),
        _context().candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "role=return final value" in text
    assert "signature=def f(x):" in text
    assert "ast_path=function_definition/return_statement" in text
    assert "masked_parent=def f(<NAME>):" in text
    assert "return x + 1" not in text


def test_role_aware_skeleton_anchor_includes_masked_candidate_skeleton():
    text = build_anchor_text(
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT_SKELETON,
        _context(),
        _context().candidates[0],
    )

    assert "role=return final value" in text
    assert "skeleton=return <NAME> + <NUMBER>" in text
