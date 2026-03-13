"""Data structures for embed-extract alignment diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from wfcllm.common.ast_parser import StatementBlock


@dataclass
class EmbedEvent:
    """One watermark embedding attempt recorded during DiagnosticGenerator.generate()."""

    path: Literal["simple", "fallback", "cascade"]
    block_text: str          # 生成端捕获的原始文本
    parent_node_type: str    # 生成端推断的父节点类型
    node_type: str           # 当前节点类型
    passed: bool             # 该 block 的最终嵌入结果（含 retry 后结果）


@dataclass
class AlignedPair:
    """One matched (EmbedEvent, StatementBlock) pair."""

    embed: EmbedEvent
    extract: StatementBlock
    text_match: bool          # embed.block_text.strip() == extract.source.strip()
    parent_match: bool        # embed.parent_node_type == resolved parent type
    embed_passed: bool        # embed.passed
    extract_score: int | None # BlockScore.score; None for compound blocks
    score_agree: bool         # extract_score is not None and int(embed_passed) == extract_score


@dataclass
class PromptReport:
    """Full alignment diagnostics for one HumanEval prompt."""

    prompt_id: str
    generated_code: str
    embed_events: list[EmbedEvent]
    aligned_pairs: list[AlignedPair]
    unmatched_embeds: list[EmbedEvent]         # 生成端有、提取端找不到
    unmatched_extracts: list[StatementBlock]   # 提取端有、生成端没记录
    detect_z_score: float
    detect_is_watermarked: bool

    # Derived summary counts
    embed_total: int
    embed_simple_passed: int
    embed_simple_failed: int
    embed_fallback_passed: int
    embed_fallback_failed: int
    embed_cascade_passed: int
    embed_cascade_failed: int
    embed_compound_total: int      # fallback+cascade 总数（怀疑三核心指标）
    extract_simple_count: int
    embed_unmatched_count: int
    extract_unmatched_count: int
    compound_aligned_count: int    # aligned_pairs 中 extract_score is None 的数量
    text_mismatch_count: int
    parent_mismatch_count: int
    score_disagree_count: int      # excludes pairs where extract_score is None


@dataclass
class SummaryReport:
    """Aggregated diagnostics across all prompts."""

    n_prompts: int
    total_embed_events: int
    compound_only_events: int   # sum of all fallback+cascade events
    compound_ratio: float       # compound_only_events / total_embed_events
    text_mismatch_total: int
    parent_mismatch_total: int
    score_disagree_total: int
    avg_embed_rate: float       # mean(simple_passed / embed_total) for reports with embed_total > 0
    avg_detect_z: float
