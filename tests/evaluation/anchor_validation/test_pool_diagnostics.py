from __future__ import annotations

import pytest

from wfcllm.evaluation.anchor_validation.pool_diagnostics import (
    build_pool_quality_summary,
    enrich_pool_quality_with_embedding_diversity,
)
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext


def _context(
    context_id: str,
    node_type: str,
    parent_node_type: str,
    task_id: str,
    candidates: tuple[CandidateBlock, ...],
) -> CandidateContext:
    return CandidateContext(
        context_id=context_id,
        dataset="humaneval",
        task_id=task_id,
        prompt="def f(x):\n",
        function_signature="def f(x):",
        ast_path=("function_definition", node_type),
        node_type=node_type,
        parent_node_type=parent_node_type,
        block_ordinal=0,
        context_hash=f"{context_id}-hash",
        temperature=0.2,
        candidates=candidates,
    )


def test_pool_quality_summary_reports_candidate_and_distribution_stats():
    contexts = [
        _context(
            "ctx1",
            "return_statement",
            "function_definition",
            "HumanEval/0",
            (
                CandidateBlock("c0", "return x", 0, syntax_valid=True, parse_valid=True),
                CandidateBlock("c1", "return x + 1", 1, syntax_valid=True, parse_valid=False),
                CandidateBlock("c2", "return x", 2, syntax_valid=False, parse_valid=True),
            ),
        ),
        _context(
            "ctx2",
            "expression_statement",
            "for_statement",
            "HumanEval/1",
            (
                CandidateBlock("c3", "total += item", 0, syntax_valid=True, parse_valid=True),
                CandidateBlock("c4", "items.append(item)", 1, syntax_valid=True, parse_valid=True),
            ),
        ),
    ]

    summary = build_pool_quality_summary(contexts)

    assert summary["context_count"] == 2
    assert summary["total_candidates"] == 5
    assert summary["candidates_per_context"]["min"] == 2
    assert summary["candidates_per_context"]["median"] == pytest.approx(2.5)
    assert summary["unique_candidates_per_context"]["min"] == 2
    assert summary["unique_candidates_per_context"]["max"] == 2
    assert summary["node_type_distribution"] == {
        "expression_statement": 1,
        "return_statement": 1,
    }
    assert summary["parent_node_type_distribution"] == {
        "for_statement": 1,
        "function_definition": 1,
    }
    assert summary["task_distribution"] == {"HumanEval/0": 1, "HumanEval/1": 1}
    assert summary["parse_valid_rate"] == pytest.approx(0.8)
    assert summary["syntax_valid_rate"] == pytest.approx(0.8)
    assert summary["candidate_length"]["min"] == len("return x")


def test_embedding_diversity_enrichment_averages_context_pairwise_distances():
    summary = build_pool_quality_summary([])

    enriched = enrich_pool_quality_with_embedding_diversity(
        summary,
        {
            "ctx1": [("c0", (1.0, 0.0)), ("c1", (0.0, 1.0))],
            "ctx2": [("c2", (1.0, 0.0)), ("c3", (1.0, 0.0))],
        },
    )

    assert enriched["embedding_pairwise_diversity"]["context_count"] == 2
    assert enriched["embedding_pairwise_diversity"]["mean_cosine_distance"] == pytest.approx(0.5)
    assert enriched["embedding_pairwise_diversity"]["max_cosine_distance"] == pytest.approx(1.0)
