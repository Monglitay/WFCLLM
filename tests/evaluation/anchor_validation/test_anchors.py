from __future__ import annotations

import ast

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


def _context_for_node(
    node_type: str,
    block_text: str,
    *,
    context_before: str = "from math import sqrt\n\ndef f(x):\n",
    context_after: str = "    return x\n",
    parent_node_type: str = "function_definition",
    import_and_helper_signatures: tuple[str, ...] = ("from math import sqrt", "def helper(v):"),
) -> CandidateContext:
    return CandidateContext(
        context_id=f"ctx-{node_type}",
        dataset="humaneval",
        task_id="HumanEval/1",
        prompt="def f(x):\n",
        function_signature="def f(x):",
        ast_path=("function_definition", node_type),
        node_type=node_type,
        parent_node_type=parent_node_type,
        block_ordinal=3,
        context_hash="ctxhash-node",
        temperature=0.2,
        candidates=(CandidateBlock("c0", block_text, 0),),
        context_before=context_before,
        context_after=context_after,
        masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
        import_and_helper_signatures=import_and_helper_signatures,
    )


def _assert_parseable_anchor(text: str) -> None:
    ast.parse(text.replace("<extra_id_0>", "pass"))


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


def test_codet5_masked_code_anchor_is_code_like_deterministic_and_secret_free():
    context = _context_for_node("return_statement", "return x + 1")

    first = build_anchor_text(
        AnchorMethod.CODET5_MASKED_CODE,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )
    second = build_anchor_text(AnchorMethod.CODET5_MASKED_CODE, context, context.candidates[0])

    assert first == second
    assert "do-not-leak" not in first
    assert "from math import sqrt" in first
    assert "def helper(v):" in first
    assert "def f(x):" in first
    assert "<extra_id_0>" in first
    assert "return x + 1" not in first
    assert "Fill" not in first
    assert first.count("from math import sqrt") == 1
    assert first.count("def f(x):") == 1
    assert "def helper(v):\ndef f(x):" not in first
    _assert_parseable_anchor(first)


def test_codet5_valid_skeleton_anchor_handles_statement_node_types_without_secret_leakage():
    cases = [
        ("expression_statement", "total += x", "_ = None"),
        ("return_statement", "return x", "return None"),
        ("import_from_statement", "from math import sqrt", "from math import sqrt"),
    ]

    for node_type, block_text, expected in cases:
        context = _context_for_node(node_type, block_text)
        text = build_anchor_text(
            AnchorMethod.CODET5_VALID_SKELETON,
            context,
            context.candidates[0],
            secret_key="do-not-leak",
        )

        assert "do-not-leak" not in text
        assert expected in text
        assert "<extra_id_0>" not in text
        assert text.count("from math import sqrt") == 1
        assert text.count("def f(x):") == 1
        assert "def helper(v):\ndef f(x):" not in text
        _assert_parseable_anchor(text)


def test_codet5_comment_anchor_keeps_code_adjacent_compact_metadata():
    context = _context_for_node("return_statement", "return x")
    text = build_anchor_text(
        AnchorMethod.CODET5_COMMENT_ANCHOR,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "# wfcllm role=return" in text
    assert "slot=3" in text
    assert "ctx=ctxhash-node" in text
    assert "return None" in text
    assert text.index("# wfcllm") < text.index("return None")
    assert text.count("from math import sqrt") == 1
    assert text.count("def f(x):") == 1
    assert "def helper(v):\ndef f(x):" not in text
    _assert_parseable_anchor(text)


def test_codet5_comment_minimal_is_short_parseable_and_secret_free():
    context = _context_for_node("return_statement", "return x + 1")
    text = build_anchor_text(
        AnchorMethod.CODET5_COMMENT_MINIMAL,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "ctxhash" not in text
    assert "return x + 1" not in text
    assert "# wfcllm:" in text
    assert "return_statement" in text
    assert "ordinal_3" in text
    assert "return None" in text
    _assert_parseable_anchor(text)


def test_codet5_comment_contextual_keeps_context_after_parseable():
    context = _context_for_node(
        "expression_statement",
        "total += x",
        context_before="def f(x):\n    total = 0\n",
        context_after="    return total\n",
    )
    text = build_anchor_text(
        AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "total += x" not in text
    assert "def f(x):" in text
    assert "# wfcllm:" in text
    assert "_ = None" in text
    assert "return total" in text
    assert text.index("# wfcllm:") < text.index("_ = None")
    assert text.index("_ = None") < text.index("return total")
    _assert_parseable_anchor(text)


def test_new_codet5_comment_variants_handle_import_from_statement():
    context = _context_for_node(
        "import_from_statement",
        "from .utils import helper",
        context_before="",
        context_after="",
        parent_node_type="module",
    )

    for method in (
        AnchorMethod.CODET5_COMMENT_MINIMAL,
        AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
    ):
        text = build_anchor_text(method, context, context.candidates[0])
        assert "from .utils import helper" in text
        _assert_parseable_anchor(text)


def test_codet5_identifier_anchor_uses_identifier_shaped_tokens_for_metadata():
    context = _context_for_node("return_statement", "return x")
    text = build_anchor_text(
        AnchorMethod.CODET5_IDENTIFIER_ANCHOR,
        context,
        context.candidates[0],
        secret_key="do-not-leak",
    )

    assert "do-not-leak" not in text
    assert "_wfcllm_slot_return_3 = None" in text
    assert "_wfcllm_ctx_ctxhash_node = None" in text
    assert "return None" in text
    assert text.count("from math import sqrt") == 1
    assert text.count("def f(x):") == 1
    assert "def helper(v):\ndef f(x):" not in text
    _assert_parseable_anchor(text)


def test_codet5_valid_skeleton_preserves_relative_import_from_shape():
    cases = [
        ("from .utils import helper", "from .utils import helper"),
        ("from ..pkg import name", "from ..pkg import name"),
        ("from . import helper", "from . import helper"),
        ("from .. import name", "from .. import name"),
    ]

    for block_text, expected in cases:
        context = _context_for_node(
            "import_from_statement",
            block_text,
            context_before="",
            context_after="",
            parent_node_type="module",
        )
        text = build_anchor_text(
            AnchorMethod.CODET5_VALID_SKELETON,
            context,
            context.candidates[0],
        )

        assert expected in text


def test_codet5_anchors_do_not_prefix_orphan_pass_when_function_context_exists():
    context = _context_for_node(
        "return_statement",
        "return x",
        context_before="def f(x):\n",
        context_after="",
        import_and_helper_signatures=(),
    )
    methods_and_markers = [
        (AnchorMethod.CODET5_MASKED_CODE, "<extra_id_0>"),
        (AnchorMethod.CODET5_VALID_SKELETON, "return None"),
        (AnchorMethod.CODET5_COMMENT_ANCHOR, "# wfcllm"),
        (AnchorMethod.CODET5_COMMENT_MINIMAL, "# wfcllm:"),
        (AnchorMethod.CODET5_COMMENT_CONTEXTUAL, "# wfcllm:"),
        (AnchorMethod.CODET5_IDENTIFIER_ANCHOR, "_wfcllm_slot_return_3 = None"),
    ]

    for method, marker in methods_and_markers:
        text = build_anchor_text(method, context, context.candidates[0])

        assert not text.startswith("pass\n")
        assert text.index("def f(x):") < text.index(marker)
        _assert_parseable_anchor(text)
