"""Tests for Aligner.align() — no model required."""
from __future__ import annotations

import pytest

from experiment.embed_extract_alignment.models import (
    EmbedEvent,
    PromptReport,
)
from experiment.embed_extract_alignment.aligner import Aligner
from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore


# ── helpers ──────────────────────────────────────────────────────────────────

def make_block(
    block_id: str,
    source: str,
    node_type: str = "expression_statement",
    block_type: str = "simple",
    start_line: int = 1,
    end_line: int = 1,
    parent_id: str | None = None,
) -> StatementBlock:
    return StatementBlock(
        block_id=block_id,
        block_type=block_type,
        node_type=node_type,
        source=source,
        start_line=start_line,
        end_line=end_line,
        depth=0,
        parent_id=parent_id,
    )


def make_embed(
    block_text: str,
    parent_node_type: str = "module",
    node_type: str = "expression_statement",
    passed: bool = True,
    path: str = "simple",
) -> EmbedEvent:
    return EmbedEvent(
        path=path,
        block_text=block_text,
        parent_node_type=parent_node_type,
        node_type=node_type,
        passed=passed,
    )


# ── tests ────────────────────────────────────────────────────────────────────

CODE = "x = 1\ny = 2\n"


def test_text_match_produces_aligned_pair():
    """Embed and extract with same text → aligned, text_match=True."""
    embed = make_embed("x = 1", passed=True)
    block = make_block("0", "x = 1", start_line=1, end_line=1)
    score = BlockScore(block_id="0", score=1, min_margin=0.5)

    report = Aligner.align(
        embed_events=[embed],
        simple_blocks=[block],
        all_blocks=[block],
        block_scores=[score],
        generated_code=CODE,
    )

    assert len(report.aligned_pairs) == 1
    pair = report.aligned_pairs[0]
    assert pair.text_match is True
    assert pair.embed_passed is True
    assert pair.extract_score == 1
    assert pair.score_agree is True


def test_whitespace_strip_still_matches():
    """Leading/trailing whitespace on embed side is ignored — pair is aligned, text_match=True."""
    embed = make_embed("  x = 1  ", passed=True)
    block = make_block("0", "x = 1", start_line=1, end_line=1)
    score = BlockScore(block_id="0", score=1, min_margin=0.5)

    report = Aligner.align(
        embed_events=[embed],
        simple_blocks=[block],
        all_blocks=[block],
        block_scores=[score],
        generated_code=CODE,
    )

    assert len(report.aligned_pairs) == 1
    # text_match is strip-equal comparison; both sides strip to "x = 1"
    assert report.aligned_pairs[0].text_match is True


def test_score_disagree_when_embed_passed_but_extract_missed():
    """Embed passed=True, extract score=0 → score_agree=False."""
    embed = make_embed("x = 1", passed=True)
    block = make_block("0", "x = 1")
    score = BlockScore(block_id="0", score=0, min_margin=0.1)

    report = Aligner.align(
        embed_events=[embed],
        simple_blocks=[block],
        all_blocks=[block],
        block_scores=[score],
        generated_code=CODE,
    )

    assert report.score_disagree_count == 1
    assert report.aligned_pairs[0].score_agree is False


def test_unmatched_embed_when_no_extract_block():
    """Embed event with text not found in any extract block → unmatched_embeds."""
    embed = make_embed("z = 99", passed=True)
    block = make_block("0", "x = 1")
    score = BlockScore(block_id="0", score=1, min_margin=0.5)

    report = Aligner.align(
        embed_events=[embed],
        simple_blocks=[block],
        all_blocks=[block],
        block_scores=[score],
        generated_code=CODE,
    )

    assert len(report.unmatched_embeds) == 1
    assert report.unmatched_embeds[0].block_text == "z = 99"
    assert report.embed_unmatched_count == 1


def test_unmatched_extract_when_no_embed_event():
    """Extract block with no matching embed event → unmatched_extracts."""
    block = make_block("0", "y = 2", start_line=2, end_line=2)
    score = BlockScore(block_id="0", score=1, min_margin=0.5)

    report = Aligner.align(
        embed_events=[],
        simple_blocks=[block],
        all_blocks=[block],
        block_scores=[score],
        generated_code=CODE,
    )

    assert len(report.unmatched_extracts) == 1
    assert report.extract_unmatched_count == 1


def test_fallback_embed_matches_compound_block():
    """Fallback embed event → matched against all_blocks (compound ok), extract_score=None."""
    embed = make_embed(
        "for i in range(10):\n    pass",
        path="fallback",
        node_type="for_statement",
        passed=True,
    )
    compound = make_block(
        "0",
        "for i in range(10):\n    pass",
        node_type="for_statement",
        block_type="compound",
    )
    # No BlockScore for compound blocks
    report = Aligner.align(
        embed_events=[embed],
        simple_blocks=[],
        all_blocks=[compound],
        block_scores=[],
        generated_code="for i in range(10):\n    pass\n",
    )

    assert len(report.aligned_pairs) == 1
    pair = report.aligned_pairs[0]
    assert pair.extract_score is None
    assert pair.score_agree is False
    assert report.compound_aligned_count == 1
    assert report.score_disagree_count == 0  # excluded because extract_score is None


def test_cascade_diagnostic_preserves_compound_alignment():
    embed = make_embed(
        "for i in range(3):\n    pass",
        path="cascade",
        node_type="for_statement",
        passed=False,
    )
    embed.is_diagnostic_compound_probe = True
    simple = make_block("0", "x = 1", start_line=1, end_line=1)
    compound = make_block(
        "1",
        "for i in range(3):\n    pass",
        node_type="for_statement",
        block_type="compound",
        start_line=2,
        end_line=3,
    )
    score = BlockScore(block_id="0", score=1, min_margin=0.5)

    report = Aligner.align(
        embed_events=[make_embed("x = 1", passed=True), embed],
        simple_blocks=[simple],
        all_blocks=[simple, compound],
        block_scores=[score],
        generated_code="x = 1\nfor i in range(3):\n    pass\n",
    )

    assert report.compound_aligned_count >= 1
    assert report.score_disagree_count == 0
    assert report.text_mismatch_simple_only == 0
    assert report.text_mismatch_compound_only == 0


def test_parent_mismatch_detection():
    """If embed.parent_node_type != resolved parent type → parent_match=False."""
    embed = make_embed("x = 1", parent_node_type="for_statement", passed=True)
    parent_block = make_block("1", "for i in range(10):\n    x = 1", node_type="function_definition", block_type="compound")
    child_block = make_block("0", "x = 1", parent_id="1")
    score = BlockScore(block_id="0", score=1, min_margin=0.5)

    report = Aligner.align(
        embed_events=[embed],
        simple_blocks=[child_block],
        all_blocks=[parent_block, child_block],
        block_scores=[score],
        generated_code="for i in range(10):\n    x = 1\n",
    )

    assert len(report.aligned_pairs) == 1
    assert report.aligned_pairs[0].parent_match is False
    assert report.parent_mismatch_count == 1


def test_summary_counts_are_consistent():
    """All embed_* counts in PromptReport sum consistently."""
    events = [
        make_embed("x = 1", passed=True, path="simple"),
        make_embed("y = 2", passed=False, path="simple"),
        make_embed("for i in range(10):\n    pass", passed=True, path="fallback", node_type="for_statement"),
        make_embed("while True:\n    pass", passed=False, path="fallback", node_type="while_statement"),
    ]
    report = Aligner.align(
        embed_events=events,
        simple_blocks=[],
        all_blocks=[],
        block_scores=[],
        generated_code="x = 1\ny = 2\nfor i in range(10):\n    pass\nwhile True:\n    pass\n",
    )

    assert report.embed_total == 4
    assert report.embed_simple_passed == 1
    assert report.embed_simple_failed == 1
    assert report.embed_fallback_passed == 1
    assert report.embed_fallback_failed == 1
    assert report.embed_compound_total == 2


class TestDiagnosticGeneratorNoFallback:
    def test_no_diag_try_passive_fallback(self):
        """DiagnosticGenerator 不应有 _diag_try_passive_fallback 方法。"""
        from experiment.embed_extract_alignment.diagnostic_generator import DiagnosticGenerator
        assert not hasattr(DiagnosticGenerator, "_diag_try_passive_fallback"), (
            "DiagnosticGenerator._diag_try_passive_fallback 应已删除（与主系统对齐）"
        )

    def test_embed_event_supports_diagnostic_compound_probe_flag(self):
        event = make_embed("for i in range(3):\n    pass", path="cascade")
        assert event.is_diagnostic_compound_probe is False
