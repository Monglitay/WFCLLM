"""Block alignment between embed-side events and extract-side StatementBlocks."""
from __future__ import annotations

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.config import BlockScore

from experiment.embed_extract_alignment.models import (
    AlignedPair,
    EmbedEvent,
    PromptReport,
)


class Aligner:
    """Align EmbedEvents from DiagnosticGenerator against StatementBlocks from detector."""

    @staticmethod
    def align(
        embed_events: list[EmbedEvent],
        simple_blocks: list[StatementBlock],
        all_blocks: list[StatementBlock],
        block_scores: list[BlockScore],
        generated_code: str,
        prompt_id: str = "",
        detect_z_score: float = 0.0,
        detect_is_watermarked: bool = False,
    ) -> PromptReport:
        score_map = {s.block_id: s.score for s in block_scores}
        block_by_id = {b.block_id: b for b in all_blocks}

        aligned_pairs: list[AlignedPair] = []
        unmatched_embeds: list[EmbedEvent] = []
        used_extract_ids: set[str] = set()

        for embed in embed_events:
            # Determine candidate pool: simple-only for "simple" path, all for compound
            candidates = (
                simple_blocks
                if embed.path == "simple"
                else all_blocks
            )

            matched = Aligner._match_embed(embed, candidates, generated_code, used_extract_ids)

            if matched is None:
                unmatched_embeds.append(embed)
                continue

            used_extract_ids.add(matched.block_id)

            # Resolve parent type on extract side
            resolved_parent = Aligner._resolve_parent_type(matched, block_by_id)
            text_match = embed.block_text.strip() == matched.source.strip()
            parent_match = embed.parent_node_type == resolved_parent
            extract_score = score_map.get(matched.block_id, None)
            score_agree = (
                extract_score is not None
                and int(embed.passed) == extract_score
            )

            aligned_pairs.append(AlignedPair(
                embed=embed,
                extract=matched,
                text_match=text_match,
                parent_match=parent_match,
                embed_passed=embed.passed,
                extract_score=extract_score,
                score_agree=score_agree,
            ))

        unmatched_extracts = [
            b for b in simple_blocks if b.block_id not in used_extract_ids
        ]

        return Aligner._build_report(
            prompt_id=prompt_id,
            generated_code=generated_code,
            embed_events=embed_events,
            aligned_pairs=aligned_pairs,
            unmatched_embeds=unmatched_embeds,
            unmatched_extracts=unmatched_extracts,
            simple_blocks=simple_blocks,
            detect_z_score=detect_z_score,
            detect_is_watermarked=detect_is_watermarked,
        )

    @staticmethod
    def _match_embed(
        embed: EmbedEvent,
        candidates: list[StatementBlock],
        generated_code: str,
        used_ids: set[str],
    ) -> StatementBlock | None:
        # Pass 1: text strip match
        for block in candidates:
            if block.block_id in used_ids:
                continue
            if embed.block_text.strip() == block.source.strip():
                return block

        # Pass 2: position match — find block_text in generated_code, compute line range
        idx = generated_code.find(embed.block_text)
        if idx == -1:
            return None

        start_line = generated_code[:idx].count("\n") + 1
        end_line = generated_code[:idx + len(embed.block_text)].count("\n") + 1

        for block in candidates:
            if block.block_id in used_ids:
                continue
            if block.start_line == start_line and block.end_line == end_line:
                return block

        return None

    @staticmethod
    def _resolve_parent_type(block: StatementBlock, block_by_id: dict[str, StatementBlock]) -> str:
        if block.parent_id is None:
            return "module"
        parent = block_by_id.get(block.parent_id)
        return parent.node_type if parent else "module"

    @staticmethod
    def _build_report(
        prompt_id: str,
        generated_code: str,
        embed_events: list[EmbedEvent],
        aligned_pairs: list[AlignedPair],
        unmatched_embeds: list[EmbedEvent],
        unmatched_extracts: list[StatementBlock],
        simple_blocks: list[StatementBlock],
        detect_z_score: float,
        detect_is_watermarked: bool,
    ) -> PromptReport:
        def count(path, passed):
            return sum(1 for e in embed_events if e.path == path and e.passed == passed)

        compound_total = sum(1 for e in embed_events if e.path in ("fallback", "cascade"))

        return PromptReport(
            prompt_id=prompt_id,
            generated_code=generated_code,
            embed_events=embed_events,
            aligned_pairs=aligned_pairs,
            unmatched_embeds=unmatched_embeds,
            unmatched_extracts=unmatched_extracts,
            detect_z_score=detect_z_score,
            detect_is_watermarked=detect_is_watermarked,
            embed_total=len(embed_events),
            embed_simple_passed=count("simple", True),
            embed_simple_failed=count("simple", False),
            embed_fallback_passed=count("fallback", True),
            embed_fallback_failed=count("fallback", False),
            embed_cascade_passed=count("cascade", True),
            embed_cascade_failed=count("cascade", False),
            embed_compound_total=compound_total,
            extract_simple_count=len(simple_blocks),
            embed_unmatched_count=len(unmatched_embeds),
            extract_unmatched_count=len(unmatched_extracts),
            compound_aligned_count=sum(1 for p in aligned_pairs if p.extract_score is None),
            text_mismatch_count=sum(1 for p in aligned_pairs if not p.text_match),
            parent_mismatch_count=sum(1 for p in aligned_pairs if not p.parent_match),
            score_disagree_count=sum(
                1 for p in aligned_pairs
                if p.extract_score is not None and not p.score_agree
            ),
        )
