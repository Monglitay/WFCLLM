from __future__ import annotations

import pytest

from wfcllm.evaluation.anchor_validation.pool_builder import build_candidate_contexts_from_records


def test_build_candidate_contexts_groups_repeated_task_blocks_by_ordinal():
    records = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    return x + 1\n",
            "candidate_index": 0,
            "temperature": 0.2,
        },
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    return 1 + x\n",
            "candidate_index": 1,
            "temperature": 0.2,
        },
    ]

    contexts = build_candidate_contexts_from_records(records, min_candidates=2)

    assert contexts
    first = contexts[0]
    assert first.dataset == "humaneval"
    assert first.task_id == "HumanEval/0"
    assert first.function_signature == "def add_one(x):"
    assert len(first.candidates) == 2
    assert first.parent_node_type == "function_definition"
    assert first.context_hash
    assert "<TARGET_BLOCK>" in first.masked_parent_context
    assert "return 1 + x" not in first.context_before
    assert "return 1 + x" not in first.context_after
    assert "return 1 + x" not in first.masked_parent_context


def test_build_candidate_contexts_does_not_mix_changed_surrounding_contexts():
    records = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    y = x + 1\n    return y\n",
            "candidate_index": 0,
        },
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def add_one(x):\n",
            "generated_code": "    debug = x\n    y = x + 1\n    return y\n",
            "candidate_index": 1,
        },
    ]

    with pytest.raises(ValueError, match="ambiguous whole-program candidate grouping"):
        build_candidate_contexts_from_records(records, min_candidates=2)


def test_build_candidate_contexts_rejects_mixed_valid_and_ambiguous_ordinals():
    records = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def f(x):\n",
            "generated_code": "    a = x + 1\n    b = x\n    return b\n",
            "candidate_index": 0,
        },
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def f(x):\n",
            "generated_code": "    a = x + 1\n    b = x + 1\n    return b\n",
            "candidate_index": 1,
        },
    ]

    with pytest.raises(ValueError, match="ambiguous whole-program candidate grouping"):
        build_candidate_contexts_from_records(records, min_candidates=2)


def test_build_candidate_contexts_accepts_explicit_per_block_rows():
    records = [
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c0",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x",
            "context_hash": "ctxhash",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 0,
        },
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c1",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x + 1",
            "context_hash": "ctxhash",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 1,
        },
    ]

    contexts = build_candidate_contexts_from_records(records, min_candidates=2)

    assert len(contexts) == 1
    assert contexts[0].context_id == "ctx-1"
    assert len(contexts[0].candidates) == 2


def test_build_candidate_contexts_rejects_conflicting_explicit_context_metadata():
    records = [
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c0",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x",
            "context_hash": "ctxhash-a",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 0,
        },
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c1",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x + 1",
            "context_hash": "ctxhash-b",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 1,
        },
    ]

    with pytest.raises(ValueError, match="conflicting explicit candidate context"):
        build_candidate_contexts_from_records(records, min_candidates=2)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", "1", "rank"),
        ("syntax_valid", "false", "syntax_valid"),
        ("parse_valid", 1, "parse_valid"),
    ],
)
def test_build_candidate_contexts_validates_explicit_candidate_types(
    field,
    value,
    message,
):
    records = [
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c0",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x",
            "context_hash": "ctxhash",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 0,
            "syntax_valid": True,
            "parse_valid": True,
        },
        {
            "candidate_context_id": "ctx-1",
            "candidate_id": "c1",
            "task_id": "HumanEval/0",
            "dataset": "humaneval",
            "block_text": "return x + 1",
            "context_hash": "ctxhash",
            "context_before": "def f(x):\n",
            "context_after": "",
            "masked_parent_context": "def f(<NAME>):\n    <TARGET_BLOCK>",
            "node_type": "return_statement",
            "parent_node_type": "function_definition",
            "block_ordinal": 0,
            "rank": 1,
            "syntax_valid": True,
            "parse_valid": True,
        },
    ]
    records[1][field] = value

    with pytest.raises(ValueError, match=message):
        build_candidate_contexts_from_records(records, min_candidates=2)


def test_build_candidate_contexts_skips_slots_below_min_candidates():
    records = [
        {
            "id": "HumanEval/0",
            "dataset": "humaneval",
            "prompt": "def f(x):\n",
            "generated_code": "    return x\n",
        }
    ]

    assert build_candidate_contexts_from_records(records, min_candidates=2) == []
