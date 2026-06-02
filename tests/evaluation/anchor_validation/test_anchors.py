from __future__ import annotations

from wfcllm.evaluation.anchor_validation.anchors import build_anchor_text, mask_code_skeleton
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
