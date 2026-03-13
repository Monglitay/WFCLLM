# Embed-Extract Alignment Diagnostic Experiment Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 编写诊断实验脚本，对比 WatermarkGenerator 嵌入端与 WatermarkDetector 提取端的 block 集合差异，定位水印命中率低的根因。

**Architecture:** 在 `experiment/embed_extract_alignment/` 下新建五个模块：数据结构定义（models.py）、生成器子类（diagnostic_generator.py）、对齐器（aligner.py）、报告格式化（report.py）、主入口（run.py）。子类化 WatermarkGenerator 复制主循环并在六个关键判断点插入 EmbedEvent 记录，不修改 `wfcllm/` 任何代码。

**Tech Stack:** Python 3.10+, tree-sitter-python, dataclasses, HuggingFace datasets, pytest

**Spec:** `docs/superpowers/specs/2026-03-13-embed-extract-alignment-design.md`

---

## File Structure

```
experiment/embed_extract_alignment/
├── __init__.py
├── models.py                # EmbedEvent, AlignedPair, PromptReport, SummaryReport
├── diagnostic_generator.py  # DiagnosticGenerator(WatermarkGenerator)
├── aligner.py               # Aligner.align() — pure alignment logic
├── report.py                # console summary + JSON serialization
└── run.py                   # CLI entry point

tests/experiment/
├── __init__.py
└── embed_extract_alignment/
    ├── __init__.py
    └── test_aligner.py      # unit tests for Aligner (no model needed)
```

**参考代码（只读，禁止 import）：**
- `wfcllm/watermark/generator.py` — DiagnosticGenerator 复制 generate() 的来源
- `wfcllm/watermark/config.py` — WatermarkConfig
- `wfcllm/extract/detector.py` / `wfcllm/extract/config.py` — WatermarkDetector, BlockScore
- `wfcllm/common/ast_parser.py` — extract_statement_blocks, StatementBlock
- `wfcllm/common/dataset_loader.py` — load_prompts

---

## Chunk 1: 数据结构 + Aligner

### Task 1: models.py — 定义所有数据结构

**Files:**
- Create: `experiment/embed_extract_alignment/__init__.py`
- Create: `experiment/embed_extract_alignment/models.py`

- [ ] **Step 1: 创建 `experiment/embed_extract_alignment/__init__.py`（空文件）**

```python
```

- [ ] **Step 2: 写 models.py**

```python
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
```

- [ ] **Step 3: Commit**

```bash
git add experiment/embed_extract_alignment/__init__.py experiment/embed_extract_alignment/models.py
git commit -m "feat: add embed-extract alignment models"
```

---

### Task 2: tests/experiment/embed_extract_alignment/test_aligner.py — 写失败测试

**Files:**
- Create: `tests/experiment/__init__.py`
- Create: `tests/experiment/embed_extract_alignment/__init__.py`
- Create: `tests/experiment/embed_extract_alignment/test_aligner.py`

- [ ] **Step 1: 创建 test package init 文件**

```python
# tests/experiment/__init__.py  (空)
# tests/experiment/embed_extract_alignment/__init__.py  (空)
```

- [ ] **Step 2: 写测试文件**

```python
"""Tests for Aligner.align() — no model required."""
from __future__ import annotations

import pytest

from experiment.embed_extract_alignment.models import (
    AlignedPair,
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
```

- [ ] **Step 3: 运行测试，确认 FAIL（Aligner 尚未实现）**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/experiment/embed_extract_alignment/test_aligner.py -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `aligner`

---

### Task 3: aligner.py — 实现 Aligner.align()

**Files:**
- Create: `experiment/embed_extract_alignment/aligner.py`

- [ ] **Step 1: 写 aligner.py**

```python
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
```

- [ ] **Step 2: 运行测试，确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/experiment/embed_extract_alignment/test_aligner.py -v
```

Expected: 全部 PASS

- [ ] **Step 3: Commit**

```bash
git add experiment/embed_extract_alignment/aligner.py \
        tests/experiment/__init__.py \
        tests/experiment/embed_extract_alignment/__init__.py \
        tests/experiment/embed_extract_alignment/test_aligner.py
git commit -m "feat: add aligner and unit tests for embed-extract alignment"
```

---

## Chunk 2: DiagnosticGenerator + report.py + run.py

### Task 4: diagnostic_generator.py — 子类化 WatermarkGenerator

**Files:**
- Create: `experiment/embed_extract_alignment/diagnostic_generator.py`

**注意**：这里复制了 `wfcllm/watermark/generator.py` 的 `generate()` 主循环及两个 helper 方法。如果 generator.py 有版本变动，需要同步更新。

- [ ] **Step 1: 写 diagnostic_generator.py**

```python
"""DiagnosticGenerator: WatermarkGenerator subclass that records EmbedEvents."""
from __future__ import annotations

import logging

import torch

from wfcllm.watermark.cascade import CascadeManager
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.generator import EmbedStats, GenerateResult, WatermarkGenerator
from wfcllm.watermark.retry_loop import RetryLoop

from experiment.embed_extract_alignment.models import EmbedEvent

logger = logging.getLogger(__name__)


class DiagnosticGenerator(WatermarkGenerator):
    """Subclass of WatermarkGenerator that records all EmbedEvents.

    Usage:
        gen = DiagnosticGenerator(model, tokenizer, encoder, enc_tokenizer, config)
        result = gen.generate(prompt)
        events = gen.embed_events   # list[EmbedEvent], populated after generate()
    """

    embed_events: list[EmbedEvent]

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerateResult:
        """Generate with watermark, recording all embed attempts into self.embed_events."""
        self.embed_events = []

        ctx = GenerationContext(
            model=self._model,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        ctx.prefill(prompt)

        stats = EmbedStats()
        cascade_mgr = CascadeManager(self._config)
        retry_loop = RetryLoop(
            ctx=ctx,
            config=self._config,
            verifier=self._verifier,
            keying=self._keying,
            entropy_est=self._entropy_est,
            structural_token_ids=self._structural_token_ids,
        )
        pending_fallbacks: list[str] = []

        while not ctx.is_finished():
            next_id = ctx.forward_and_sample()

            if next_id == ctx.eos_id:
                break

            event = ctx.last_event
            if event is None:
                continue

            if event.block_type == "compound":
                cascade_mgr.on_compound_block_start(ctx, event)
                self._diag_try_passive_fallback(ctx, event, stats, pending_fallbacks)
                continue

            # ── Simple block ──────────────────────────────────────────────
            stats.total_blocks += 1
            verify_result = self._verify_block(event)

            if verify_result.passed:
                # Recording point 1: simple passed (no retry needed)
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=True,
                ))
                continue

            block_cp = ctx.last_block_checkpoint
            if block_cp is None:
                # Recording point 3: failed, no checkpoint for retry
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=False,
                ))
                continue

            retry_result = retry_loop.run(block_cp, event)
            stats.retry_diagnostics.append(retry_result.diagnostics)

            if retry_result.success:
                # Recording point 2: simple passed after retry
                stats.embedded_blocks += 1
                pending_fallbacks.clear()
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=True,
                ))
                logger.debug("[RETRY OK] block #%d", stats.total_blocks)
            else:
                # Recording point 3: retry exhausted
                stats.failed_blocks += 1
                pending_fallbacks.append(event.block_text)
                cascade_mgr.on_simple_block_failed(event.block_text)
                self.embed_events.append(EmbedEvent(
                    path="simple",
                    block_text=event.block_text,
                    parent_node_type=event.parent_node_type or "module",
                    node_type=event.node_type,
                    passed=False,
                ))
                logger.debug("[RETRY FAILED] block #%d", stats.total_blocks)

                if cascade_mgr.should_cascade():
                    self._diag_try_cascade(ctx, cascade_mgr, retry_loop, stats, pending_fallbacks)

        return GenerateResult(code=ctx.generated_text, stats=stats)

    def _diag_try_passive_fallback(self, ctx, event, stats, pending_fallbacks) -> None:
        """Passive fallback with EmbedEvent recording (points 4 & 5)."""
        if not self._config.enable_fallback or not pending_fallbacks:
            return

        stats.total_blocks += 1
        block_entropy = self._entropy_est.estimate_block_entropy(event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(event.parent_node_type or "module")
        result = self._verifier.verify(event.block_text, valid_set, margin)

        # Recording point 4 (passed) or 5 (failed)
        self.embed_events.append(EmbedEvent(
            path="fallback",
            block_text=event.block_text,
            parent_node_type=event.parent_node_type or "module",
            node_type=event.node_type,
            passed=result.passed,
        ))

        if result.passed:
            stats.fallback_blocks += 1
            pending_fallbacks.clear()
            logger.debug("[FALLBACK OK] compound node=%s", event.node_type)
        else:
            logger.debug("[FALLBACK MISS] compound node=%s", event.node_type)

    def _diag_try_cascade(self, ctx, cascade_mgr, retry_loop, stats, pending_fallbacks) -> None:
        """Cascade regeneration with EmbedEvent recording (point 6).

        Records the NEWLY REGENERATED compound block's text/parent (not the triggering block).
        """
        cascade_cp = cascade_mgr.cascade(ctx)
        if cascade_cp is None:
            return

        compound_event = None
        for _ in range(self._config.max_new_tokens):
            next_id = ctx.forward_and_sample()
            if next_id == ctx.eos_id:
                break
            event = ctx.last_event
            if event is not None and event.block_type == "compound":
                compound_event = event
                break

        if compound_event is None:
            logger.debug("[CASCADE FAILED] could not regenerate compound block")
            return

        block_entropy = self._entropy_est.estimate_block_entropy(compound_event.block_text)
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        valid_set = self._keying.derive(compound_event.parent_node_type or "module")
        result = self._verifier.verify(compound_event.block_text, valid_set, margin)

        # Recording point 6: cascade passed or failed
        # block_text/parent_node_type = regenerated compound_event (per spec)
        self.embed_events.append(EmbedEvent(
            path="cascade",
            block_text=compound_event.block_text,
            parent_node_type=compound_event.parent_node_type or "module",
            node_type=compound_event.node_type,
            passed=result.passed,
        ))

        if result.passed:
            stats.cascade_blocks += 1
            pending_fallbacks.clear()
            logger.debug("[CASCADE OK] regenerated compound block passed")
        else:
            logger.debug("[CASCADE FAILED] regenerated compound block did not pass")
```

- [ ] **Step 2: 快速冒烟测试（不跑模型，仅测 embed_events 初始化）**

打开 Python REPL（不需要 pytest）：

```python
# 只验证 import 不报错，不跑实际推理
from experiment.embed_extract_alignment.diagnostic_generator import DiagnosticGenerator
print("import ok")
```

运行：
```bash
conda run -n WFCLLM python -c "from experiment.embed_extract_alignment.diagnostic_generator import DiagnosticGenerator; print('import ok')"
```

Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add experiment/embed_extract_alignment/diagnostic_generator.py
git commit -m "feat: add DiagnosticGenerator with 6 EmbedEvent recording points"
```

---

### Task 5: report.py — 控制台摘要 + JSON 序列化

**Files:**
- Create: `experiment/embed_extract_alignment/report.py`

- [ ] **Step 1: 写 report.py**

```python
"""Console summary printing and JSON serialization for alignment reports."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from experiment.embed_extract_alignment.models import PromptReport, SummaryReport


def print_prompt_summary(report: PromptReport) -> None:
    """Print one-line + detail summary for a single prompt."""
    print(
        f"[{report.prompt_id}] "
        f"embed={report.embed_total} "
        f"simple={report.embed_simple_passed}✓ "
        f"fallback={report.embed_fallback_passed}✓ "
        f"cascade={report.embed_cascade_passed} "
        f"failed={report.embed_simple_failed}"
    )
    print(
        f"  extract_simple={report.extract_simple_count}  "
        f"aligned={len(report.aligned_pairs)}  "
        f"unmatched_embed={report.embed_unmatched_count}  "
        f"unmatched_extract={report.extract_unmatched_count}"
    )
    print(
        f"  compound_aligned={report.compound_aligned_count}  "
        f"text_mismatch={report.text_mismatch_count}  "
        f"parent_mismatch={report.parent_mismatch_count}  "
        f"score_disagree={report.score_disagree_count}  "
        f"z={report.detect_z_score:.2f}"
    )


def print_summary(summary: SummaryReport) -> None:
    """Print aggregated summary across all prompts."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  prompts:           {summary.n_prompts}")
    print(f"  total_embed_events:{summary.total_embed_events}")
    print(f"  compound_events:   {summary.compound_only_events} ({summary.compound_ratio:.1%})")
    print(f"  text_mismatch:     {summary.text_mismatch_total}")
    print(f"  parent_mismatch:   {summary.parent_mismatch_total}")
    print(f"  score_disagree:    {summary.score_disagree_total}")
    print(f"  avg_embed_rate:    {summary.avg_embed_rate:.1%}")
    print(f"  avg_detect_z:      {summary.avg_detect_z:.2f}")


def build_summary(reports: list[PromptReport]) -> SummaryReport:
    """Aggregate PromptReports into a SummaryReport."""
    n = len(reports)
    if n == 0:
        return SummaryReport(
            n_prompts=0,
            total_embed_events=0, compound_only_events=0, compound_ratio=0.0,
            text_mismatch_total=0, parent_mismatch_total=0, score_disagree_total=0,
            avg_embed_rate=0.0, avg_detect_z=0.0,
        )

    total_embed = sum(r.embed_total for r in reports)
    compound_total = sum(r.embed_compound_total for r in reports)

    embed_rates = [
        r.embed_simple_passed / r.embed_total
        for r in reports if r.embed_total > 0
    ]
    avg_rate = sum(embed_rates) / len(embed_rates) if embed_rates else 0.0
    avg_z = sum(r.detect_z_score for r in reports) / n

    return SummaryReport(
        n_prompts=n,
        total_embed_events=total_embed,
        compound_only_events=compound_total,
        compound_ratio=compound_total / total_embed if total_embed > 0 else 0.0,
        text_mismatch_total=sum(r.text_mismatch_count for r in reports),
        parent_mismatch_total=sum(r.parent_mismatch_count for r in reports),
        score_disagree_total=sum(r.score_disagree_count for r in reports),
        avg_embed_rate=avg_rate,
        avg_detect_z=avg_z,
    )


def _to_dict(obj) -> object:
    """Recursively convert dataclasses and lists to JSON-serializable dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


def save_reports(
    reports: list[PromptReport],
    summary: SummaryReport,
    output_dir: str,
    timestamp: str,
) -> tuple[str, str]:
    """Save summary JSON and details JSONL. Returns (summary_path, details_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / f"summary_{timestamp}.json"
    details_path = out / f"details_{timestamp}.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_dict(summary), f, ensure_ascii=False, indent=2)

    with open(details_path, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(_to_dict(r), ensure_ascii=False) + "\n")

    return str(summary_path), str(details_path)
```

- [ ] **Step 2: Commit**

```bash
git add experiment/embed_extract_alignment/report.py
git commit -m "feat: add alignment report formatting and JSON serialization"
```

---

### Task 6: run.py — CLI 主入口

**Files:**
- Create: `experiment/embed_extract_alignment/run.py`

**前提**：需要一个已加载的 LLM。`run.py` 使用 `transformers.AutoModelForCausalLM` 加载，模型路径通过 `--model_path` 传入。encoder 固定使用 `data/models/codet5-base`。

- [ ] **Step 1: 写 run.py**

```python
"""CLI entry point for embed-extract alignment diagnostic experiment.

Usage:
    conda run -n WFCLLM python -m experiment.embed_extract_alignment.run \\
        --secret_key my-key \\
        --model_path /path/to/llm \\
        --n_samples 20 \\
        --output_dir data/diag_reports
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, T5EncoderModel

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.dataset_loader import load_prompts
from wfcllm.extract.config import ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.watermark.config import WatermarkConfig

from experiment.embed_extract_alignment.aligner import Aligner
from experiment.embed_extract_alignment.diagnostic_generator import DiagnosticGenerator
from experiment.embed_extract_alignment.models import PromptReport
from experiment.embed_extract_alignment.report import (
    build_summary,
    print_prompt_summary,
    print_summary,
    save_reports,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed-extract alignment diagnostic")
    p.add_argument("--secret_key", required=True, help="Watermark secret key")
    p.add_argument("--model_path", required=True, help="Path to LLM (HuggingFace format)")
    p.add_argument("--n_samples", type=int, default=20, help="Number of HumanEval prompts")
    p.add_argument("--output_dir", default="data/diag_reports", help="Output directory")
    p.add_argument("--dataset_path", default="data/datasets", help="Local datasets root")
    p.add_argument("--encoder_path", default="data/models/codet5-base", help="Encoder model path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_encoder(encoder_path: str, device: str):
    """Load T5EncoderModel in eval mode."""
    encoder = T5EncoderModel.from_pretrained(encoder_path).to(device).eval()
    from transformers import RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained(encoder_path)
    return encoder, tokenizer


def main() -> None:
    args = parse_args()

    print(f"Loading LLM from {args.model_path} ...", file=sys.stderr)
    lm = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device).eval()
    lm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading encoder from {args.encoder_path} ...", file=sys.stderr)
    encoder, enc_tokenizer = load_encoder(args.encoder_path, args.device)

    wm_config = WatermarkConfig(
        secret_key=args.secret_key,
        encoder_model_path=args.encoder_path,
        encoder_device=args.device,
    )
    extract_config = ExtractConfig(
        secret_key=args.secret_key,
        embed_dim=wm_config.encoder_embed_dim,
        lsh_d=wm_config.lsh_d,
        lsh_gamma=wm_config.lsh_gamma,
    )

    gen = DiagnosticGenerator(lm, lm_tokenizer, encoder, enc_tokenizer, wm_config)
    detector = WatermarkDetector(extract_config, encoder, enc_tokenizer, device=args.device)

    prompts = load_prompts("humaneval", args.dataset_path)[: args.n_samples]
    print(f"Loaded {len(prompts)} prompts. Starting diagnostic run...\n", file=sys.stderr)

    reports: list[PromptReport] = []
    for item in prompts:
        result = gen.generate(item["prompt"])
        embed_events = gen.embed_events

        detection_result = detector.detect(result.code)
        all_blocks = extract_statement_blocks(result.code)
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        report = Aligner.align(
            embed_events=embed_events,
            simple_blocks=simple_blocks,
            all_blocks=all_blocks,
            block_scores=detection_result.block_details,
            generated_code=result.code,
            prompt_id=item["id"],
            detect_z_score=detection_result.z_score,
            detect_is_watermarked=detection_result.is_watermarked,
        )
        reports.append(report)
        print_prompt_summary(report)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = build_summary(reports)
    print_summary(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path, details_path = save_reports(reports, summary, args.output_dir, timestamp)
    print(f"\nSaved: {summary_path}", file=sys.stderr)
    print(f"Saved: {details_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 验证 import 不报错**

```bash
conda run -n WFCLLM python -c "from experiment.embed_extract_alignment.run import main; print('import ok')"
```

Expected: `import ok`

- [ ] **Step 3: Commit**

```bash
git add experiment/embed_extract_alignment/run.py
git commit -m "feat: add run.py CLI entry for embed-extract alignment diagnostic"
```

---

### Task 7: 全量测试 + 最终 Commit

- [ ] **Step 1: 运行所有 aligner 测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/experiment/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: 运行现有 wfcllm 测试，确认没有破坏**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --ignore=tests/experiment
```

Expected: 全部 PASS（与实验前一致）

- [ ] **Step 3: Final commit**

```bash
git add experiment/embed_extract_alignment/ tests/experiment/
git commit -m "feat: complete embed-extract alignment diagnostic experiment"
```
