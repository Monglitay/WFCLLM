from __future__ import annotations

import json

import pytest

from wfcllm.evaluation.anchor_validation.io import (
    load_candidate_contexts,
    write_jsonl,
    write_candidate_contexts,
)
from wfcllm.evaluation.anchor_validation.schema import (
    CandidateBlock,
    CandidateContext,
    RegionMetricRow,
    SelectionSimulationRow,
)


def test_candidate_context_jsonl_roundtrip(tmp_path):
    path = tmp_path / "candidate_pools.jsonl"
    context = CandidateContext(
        context_id="humaneval:0:1",
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        function_signature="def f(x):",
        ast_path=("function_definition", "return_statement"),
        node_type="return_statement",
        parent_node_type="function_definition",
        block_ordinal=1,
        context_hash="ctxhash",
        context_before="def f(x):\n",
        context_after="",
        masked_parent_context="def f(x):\n    <TARGET_BLOCK>",
        import_and_helper_signatures=("import math", "def helper(v):"),
        temperature=0.2,
        candidates=(
            CandidateBlock(candidate_id="c0", block_text="return x + 1", rank=0),
            CandidateBlock(candidate_id="c1", block_text="return 1 + x", rank=1),
        ),
    )

    write_candidate_contexts(path, [context])
    loaded = load_candidate_contexts(path)

    assert loaded == [context]


def test_load_candidate_contexts_rejects_missing_required_fields(tmp_path):
    path = tmp_path / "candidate_pools.jsonl"
    payload = {
        "context_id": "ctx",
        "dataset": "humaneval",
        "task_id": "HumanEval/0",
        "ast_path": ["function_definition", "return_statement"],
        "node_type": "return_statement",
        "block_ordinal": 0,
        "temperature": 0.2,
        "candidates": [{"candidate_id": "c0", "block_text": "return x", "rank": 0}],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required fields"):
        load_candidate_contexts(path)


def _valid_context_payload() -> dict:
    return {
        "context_id": "ctx",
        "dataset": "humaneval",
        "task_id": "HumanEval/0",
        "prompt": "def f(x):\n",
        "function_signature": "def f(x):",
        "ast_path": ["function_definition", "return_statement"],
        "node_type": "return_statement",
        "parent_node_type": "function_definition",
        "block_ordinal": 0,
        "context_hash": "ctxhash",
        "temperature": 0.2,
        "candidates": [
            {
                "candidate_id": "c0",
                "block_text": "return x",
                "rank": 0,
                "syntax_valid": True,
                "parse_valid": True,
                "quality": {},
            }
        ],
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rank", "0", "rank"),
        ("rank", True, "rank"),
        ("syntax_valid", "false", "syntax_valid"),
        ("parse_valid", 1, "parse_valid"),
        ("quality", None, "quality"),
    ],
)
def test_load_candidate_contexts_validates_candidate_types(
    tmp_path,
    field,
    value,
    message,
):
    path = tmp_path / "candidate_pools.jsonl"
    payload = _valid_context_payload()
    payload["candidates"][0][field] = value
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_candidate_contexts(path)


def test_frozen_quality_mappings_are_defensive_copies():
    quality = {"score": 1, "nested": {"inner": 2}}
    block = CandidateBlock("c0", "return x", 0, quality=quality)

    quality["score"] = 2
    quality["nested"]["inner"] = 3

    assert block.quality["score"] == 1
    assert block.quality["nested"]["inner"] == 2
    with pytest.raises(TypeError):
        block.quality["score"] = 3
    with pytest.raises(TypeError):
        block.quality["nested"]["inner"] = 4


def test_write_jsonl_serializes_metric_and_simulation_rows(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(
        path,
        [
            RegionMetricRow(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                method="vanilla",
                projection_key_id="proj-00",
                key_id=None,
                gamma=None,
                candidate_count=2,
                normalized_entropy=1.0,
                collapse_ratio=0.0,
                effective_region_count=2.0,
                hamming_diversity=1.0,
            ),
            SelectionSimulationRow(
                context_id="ctx",
                method="vanilla",
                key_id="key-00",
                gamma=0.5,
                retry_budget=2,
                selected_candidate_id="c0",
                selected_rank=0,
                hit_acquired=True,
                fallback=False,
                z_proxy=1.0,
            ),
        ],
    )

    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert lines[0]["method"] == "vanilla"
    assert lines[1]["selected_candidate_id"] == "c0"
