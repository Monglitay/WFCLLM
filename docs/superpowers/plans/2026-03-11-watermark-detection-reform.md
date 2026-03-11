# Watermark Detection Reform Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将检测层从 `{-1, 1}` 计分 + 硬编码 `z_threshold=3.0` 改造为 `{0, 1}` 计分 + 基于 FPR 的动态阈值 $M_r$（SEMSTAMP 风格）。

**Architecture:** 三步走：(1) 计分层改 `{0,1}`；(2) `z_threshold` 重命名为 `fpr_threshold` 并改判定条件；(3) 新增 `ThresholdCalibrator` + CLI 校准脚本。所有现有测试同步更新，不留兼容 shim。

**Tech Stack:** Python, pytest, scipy.stats.norm（已有依赖）

**Spec:** `docs/superpowers/specs/2026-03-11-watermark-detection-reform-design.md`

---

## Chunk 1: 计分层 {-1,1} → {0,1}

### Task 1: 更新 BlockScorer 计分规则

**Files:**
- Modify: `wfcllm/extract/scorer.py:25`
- Modify: `wfcllm/extract/config.py`（注释）
- Test: `tests/extract/test_scorer.py`

- [ ] **Step 1: 修改测试——命中用 `score == 0` 替换 miss 断言**

编辑 `tests/extract/test_scorer.py`，将以下内容：
```python
def test_score_single_block_miss(self, keying, mock_verifier):
    """passed=False from verifier -> score = -1."""
    mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.05)
    scorer = BlockScorer(keying, mock_verifier)
    block = _make_block("0")
    result = scorer.score_block(block, blocks=[block])

    assert result.score == -1
    assert result.min_margin == 0.05
```
改为：
```python
def test_score_single_block_miss(self, keying, mock_verifier):
    """passed=False from verifier -> score = 0."""
    mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.05)
    scorer = BlockScorer(keying, mock_verifier)
    block = _make_block("0")
    result = scorer.score_block(block, blocks=[block])

    assert result.score == 0
    assert result.min_margin == 0.05
```

- [ ] **Step 2: 运行测试，确认它现在失败（断言 0，但代码还返回 -1）**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py::TestBlockScorer::test_score_single_block_miss -v
```

预期：**FAIL**，错误为 `AssertionError: assert -1 == 0`。这证明测试有效，会捕获代码变更。

- [ ] **Step 3: 修改 scorer.py 计分规则**

编辑 `wfcllm/extract/scorer.py:25`，将：
```python
        score = 1 if result.passed else -1
```
改为：
```python
        score = 1 if result.passed else 0
```

- [ ] **Step 4: 验证测试现在失败（-1 → 0，断言 0 应通过）**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py::TestBlockScorer::test_score_single_block_miss -v
```

预期：PASS（score == 0）。

- [ ] **Step 5: 更新 config.py 注释**

编辑 `wfcllm/extract/config.py:24`，将：
```python
    score: int        # +1 (hit) or -1 (miss)
```
改为：
```python
    score: int        # 1 (hit) or 0 (miss)
```

- [ ] **Step 6: 运行 scorer 全部测试**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py -v
```

预期：全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add wfcllm/extract/scorer.py wfcllm/extract/config.py tests/extract/test_scorer.py
git commit -m "refactor: change block score from {-1,1} to {0,1}"
```

---

### Task 2: 同步更新 DPSelector 测试中的 -1 分值

**Files:**
- Test: `tests/extract/test_dp_selector.py`

**背景：** DP 逻辑本身不变，但测试中传入的 `-1` 分值需改为 `0`，同时更新注释使其与 `{0,1}` 计分语义一致。DP 选择结论不变（见设计文档中的验证）。

- [ ] **Step 1: 更新 `_make_score` helper 及所有使用 `-1` 的测试**

编辑 `tests/extract/test_dp_selector.py`：

1. `_make_score` 的 `min_margin` 条件（`score == 1 else 0.1` 已正确覆盖 0，无需改）

2. `test_two_independent_roots`：
```python
# 旧
scores = [_make_score("0", 1), _make_score("1", -1)]
# 新
scores = [_make_score("0", 1), _make_score("1", 0)]
```

3. `test_parent_beats_children`：注释和分值更新：
```python
# 旧注释："""Parent score (+1) > sum of children scores (-1 + -1 = -2) -> select parent."""
# 新注释："""Parent score (1) > sum of children scores (0 + 0 = 0) -> select parent."""
# 旧分值
scores = [
    _make_score("0", 1),   # parent: +1
    _make_score("1", -1),  # child1: -1
    _make_score("2", -1),  # child2: -1
]
# 新分值
scores = [
    _make_score("0", 1),  # parent: 1
    _make_score("1", 0),  # child1: 0
    _make_score("2", 0),  # child2: 0
]
```

4. `test_children_beat_parent`：注释和分值更新：
```python
# 旧注释："""Children sum (+1 + +1 = +2) > parent score (-1) -> select children."""
# 新注释："""Children sum (1 + 1 = 2) > parent score (0) -> select children."""
# 旧分值
scores = [
    _make_score("0", -1),  # parent: -1
    _make_score("1", 1),   # child1: +1
    _make_score("2", 1),   # child2: +1
]
# 新分值
scores = [
    _make_score("0", 0),  # parent: 0
    _make_score("1", 1),  # child1: 1
    _make_score("2", 1),  # child2: 1
]
```

5. `test_three_level_nesting`：注释和分值更新：
```python
# 旧注释（docstring 内）：
#   OPT(2) = +1
#   OPT(1) = max(-1, OPT(2)) = max(-1, +1) = +1 -> use children
#   OPT(0) = max(-1, OPT(1)) = max(-1, +1) = +1 -> use children
# 新注释：
#   OPT(2) = 1
#   OPT(1) = max(0, OPT(2)) = max(0, 1) = 1 -> use children
#   OPT(0) = max(0, OPT(1)) = max(0, 1) = 1 -> use children
# 旧分值
scores = [
    _make_score("0", -1),
    _make_score("1", -1),
    _make_score("2", 1),
]
# 新分值
scores = [
    _make_score("0", 0),
    _make_score("1", 0),
    _make_score("2", 1),
]
```

- [ ] **Step 2: 运行 dp_selector 测试**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_dp_selector.py -v
```

预期：全部 PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/extract/test_dp_selector.py
git commit -m "test: update dp_selector tests for {0,1} scoring"
```

---

### Task 3: 同步更新 HypothesisTester 测试中的 -1 分值

**Files:**
- Test: `tests/extract/test_hypothesis.py`

**背景：** `HypothesisTester.test()` 只统计 `score == 1` 的数量，传入 `score=-1` 或 `score=0` 在当前计数逻辑下效果相同。但为保持一致性，统一改为 `0`。

- [ ] **Step 1: 更新测试中所有 score=-1 的地方**

编辑 `tests/extract/test_hypothesis.py`：

1. `test_half_hits` 中：
```python
# 旧
scores += [_make_score(str(i + 10), -1) for i in range(10)]
# 新
scores += [_make_score(str(i + 10), 0) for i in range(10)]
```

2. `test_custom_threshold` 中：
```python
# 旧
scores += [_make_score(str(i + 15), -1) for i in range(5)]
# 新
scores += [_make_score(str(i + 15), 0) for i in range(5)]
```

3. `test_result_includes_block_details` 中：
```python
# 旧
scores = [_make_score("0", 1), _make_score("1", -1)]
# 新
scores = [_make_score("0", 1), _make_score("1", 0)]
```

4. `test_p_value_decreases_with_z` 中：
```python
# 旧
scores_low += [_make_score(str(i + 12), -1) for i in range(8)]
# 新
scores_low += [_make_score(str(i + 12), 0) for i in range(8)]
```

- [ ] **Step 2: 运行 hypothesis 测试**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_hypothesis.py -v
```

预期：全部 PASS。

- [ ] **Step 3: Commit**

```bash
git add tests/extract/test_hypothesis.py
git commit -m "test: update hypothesis tests for {0,1} scoring"
```

---

## Chunk 2: z_threshold → fpr_threshold 重命名 + 判定条件更新

### Task 4: 重命名配置字段及相关实现

**Files:**
- Modify: `wfcllm/extract/config.py`
- Modify: `wfcllm/extract/hypothesis.py`
- Modify: `wfcllm/extract/detector.py`
- Test: `tests/extract/test_config.py`
- Test: `tests/extract/test_hypothesis.py`
- Test: `tests/extract/test_detector.py`

- [ ] **Step 1: 先更新测试（TDD——让测试先失败）**

编辑 `tests/extract/test_config.py`：
```python
# 旧 test_defaults
assert cfg.z_threshold == 3.0
# 新
assert cfg.fpr_threshold == 3.0

# 旧 test_custom_threshold
cfg = ExtractConfig(secret_key="k", z_threshold=2.5)
assert cfg.z_threshold == 2.5
# 新
cfg = ExtractConfig(secret_key="k", fpr_threshold=2.5)
assert cfg.fpr_threshold == 2.5
```

编辑 `tests/extract/test_hypothesis.py`，将所有 `z_threshold=` kwarg 改为 `fpr_threshold=`：

```python
# fixture
return HypothesisTester(fpr_threshold=3.0)

# test_custom_threshold
tester = HypothesisTester(fpr_threshold=1.0)

# TestHypothesisTesterGamma.test_custom_gamma_z_score
tester = HypothesisTester(fpr_threshold=3.0, gamma=0.25)

# TestHypothesisTesterGamma.test_default_gamma_is_half
tester = HypothesisTester(fpr_threshold=3.0)
```

编辑 `tests/extract/test_detector.py`：
```python
# 旧
return ExtractConfig(secret_key="test-key", embed_dim=128, z_threshold=3.0)
# 新
return ExtractConfig(secret_key="test-key", embed_dim=128, fpr_threshold=3.0)
```

- [ ] **Step 2: 运行测试，确认失败**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_config.py tests/extract/test_hypothesis.py tests/extract/test_detector.py -v
```

预期：失败（AttributeError: 'ExtractConfig' object has no attribute 'fpr_threshold' 等）。

- [ ] **Step 3: 更新 config.py**

编辑 `wfcllm/extract/config.py`：
```python
# 旧
z_threshold: float = 3.0
# 新
fpr_threshold: float = 3.0  # M_r，由校准脚本生成；默认值 3.0 仅作占位
```

- [ ] **Step 4: 更新 hypothesis.py**

编辑 `wfcllm/extract/hypothesis.py`：

构造函数签名：
```python
# 旧
def __init__(self, z_threshold: float = 3.0, gamma: float = 0.5):
    self._z_threshold = z_threshold
# 新
def __init__(self, fpr_threshold: float = 3.0, gamma: float = 0.5):
    self._fpr_threshold = fpr_threshold
```

判定条件（`test` 方法中的 `DetectionResult` 构造，注意**操作符由 `>` 改为 `>=`**）：
```python
# 旧
is_watermarked=z_score > self._z_threshold,
# 新（operator 变更：> → >=，使 z_score 恰好等于阈值时也判定为有水印）
is_watermarked=z_score >= self._fpr_threshold,
```

- [ ] **Step 5: 更新 detector.py**

编辑 `wfcllm/extract/detector.py`：
```python
# 旧
self._tester = HypothesisTester(config.z_threshold, gamma=config.lsh_gamma)
# 新
self._tester = HypothesisTester(config.fpr_threshold, gamma=config.lsh_gamma)
```

- [ ] **Step 6: 运行所有 extract 测试**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ -v
```

预期：全部 PASS。

- [ ] **Step 7: Commit**

```bash
git add wfcllm/extract/config.py wfcllm/extract/hypothesis.py wfcllm/extract/detector.py \
        tests/extract/test_config.py tests/extract/test_hypothesis.py tests/extract/test_detector.py
git commit -m "refactor: rename z_threshold to fpr_threshold, change > to >= in hypothesis test"
```

---

## Chunk 3: ThresholdCalibrator + CLI 脚本

### Task 5: 实现 ThresholdCalibrator

**Files:**
- Create: `wfcllm/extract/calibrator.py`
- Test: `tests/extract/test_calibrator.py`

- [ ] **Step 1: 先写测试**

创建 `tests/extract/test_calibrator.py`：

```python
"""Tests for ThresholdCalibrator."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest

from wfcllm.common.ast_parser import StatementBlock
from wfcllm.extract.calibrator import ThresholdCalibrator
from wfcllm.extract.config import BlockScore


def _make_block(block_id: str, depth: int = 0, parent_id: str | None = None) -> StatementBlock:
    return StatementBlock(
        block_id=block_id,
        block_type="simple",
        node_type="expression_statement",
        source="x = 1",
        start_line=1,
        end_line=1,
        depth=depth,
        parent_id=parent_id,
        children_ids=[],
    )


class TestThresholdCalibrator:
    @pytest.fixture
    def mock_scorer(self):
        scorer = MagicMock()
        return scorer

    def test_calibrate_returns_dict_with_required_keys(self, mock_scorer):
        """calibrate() returns dict with fpr, fpr_threshold, n_samples."""
        mock_scorer.score_all.return_value = [
            BlockScore(block_id="0", score=0, min_margin=0.1)
        ]

        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        corpus = [{"generated_code": "x = 1\n"}]
        result = calibrator.calibrate(corpus, fpr=0.05)

        assert "fpr" in result
        assert "fpr_threshold" in result
        assert "n_samples" in result
        assert result["fpr"] == 0.05
        assert result["n_samples"] == 1

    def test_calibrate_fpr_01_returns_99th_percentile(self, mock_scorer):
        """FPR=0.01 -> threshold is 99th percentile of Z scores."""
        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        z_scores = list(range(100))  # known distribution: 0,1,...,99
        threshold = calibrator._percentile_threshold(z_scores, fpr=0.01)

        # 99th percentile of 0..99: idx = 0.99 * 99 = 98.01, interpolated
        assert threshold == pytest.approx(99 * 0.99, rel=0.01)

    def test_calibrate_fpr_05_returns_95th_percentile(self, mock_scorer):
        """FPR=0.05 -> threshold is 95th percentile of Z scores."""
        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        z_scores = list(range(100))
        threshold = calibrator._percentile_threshold(z_scores, fpr=0.05)

        # 95th percentile of 0..99
        assert threshold == pytest.approx(99 * 0.95, rel=0.01)

    def test_calibrate_empty_corpus_raises(self, mock_scorer):
        """Empty corpus raises ValueError."""
        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        with pytest.raises(ValueError, match="empty"):
            calibrator.calibrate([], fpr=0.01)

    def test_calibrate_skips_no_block_samples(self, mock_scorer):
        """All corpus entries with no parseable blocks -> fpr_threshold=0.0 (no signal)."""
        mock_scorer.score_all.return_value = []

        calibrator = ThresholdCalibrator(mock_scorer, gamma=0.5)
        corpus = [{"generated_code": ""}]  # empty code -> no blocks -> skip
        result = calibrator.calibrate(corpus, fpr=0.05)

        assert result["n_samples"] == 1
        # No Z scores collected -> fallback to 0.0 (accept all, user should know corpus is bad)
        assert result["fpr_threshold"] == 0.0
```

- [ ] **Step 2: 运行测试，确认失败**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_calibrator.py -v
```

预期：`ImportError: cannot import name 'ThresholdCalibrator'`。

- [ ] **Step 3: 实现 ThresholdCalibrator**

创建 `wfcllm/extract/calibrator.py`：

```python
"""Offline FPR threshold calibration for watermark detection."""

from __future__ import annotations

import math

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.dp_selector import DPSelector
from wfcllm.extract.scorer import BlockScorer


class ThresholdCalibrator:
    """Compute FPR-based detection threshold M_r from a negative corpus.

    Uses BlockScorer + DPSelector to compute SEMSTAMP Z scores for each
    sample, then returns the (1-fpr) percentile as the threshold M_r.

    Does NOT depend on HypothesisTester to avoid circular logic.
    """

    def __init__(self, scorer: BlockScorer, gamma: float = 0.5):
        self._scorer = scorer
        self._gamma = gamma
        self._dp = DPSelector()

    def calibrate(self, corpus: list[dict], fpr: float) -> dict:
        """Compute M_r from a list of negative-sample records.

        Args:
            corpus: List of dicts with key "generated_code" (str).
            fpr: Target false positive rate, e.g. 0.01 for 1%.

        Returns:
            Dict with keys: fpr, fpr_threshold (M_r), n_samples.

        Raises:
            ValueError: If corpus is empty or no valid Z scores collected.
        """
        if not corpus:
            raise ValueError("corpus is empty")

        z_scores: list[float] = []
        for record in corpus:
            code = record.get("generated_code", "")
            blocks = extract_statement_blocks(code)
            if not blocks:
                continue

            scores = self._scorer.score_all(blocks)
            selected_ids = set(self._dp.select(blocks, scores))
            selected = [s for s in scores if s.block_id in selected_ids]

            m = len(selected)
            if m == 0:
                continue

            x = sum(1 for s in selected if s.score == 1)
            gamma = self._gamma
            z = (x - m * gamma) / math.sqrt(m * gamma * (1 - gamma))
            z_scores.append(z)

        fpr_threshold: float
        if not z_scores:
            fpr_threshold = 0.0
        else:
            fpr_threshold = self._percentile_threshold(z_scores, fpr)

        return {
            "fpr": fpr,
            "fpr_threshold": fpr_threshold,
            "n_samples": len(corpus),
        }

    @staticmethod
    def _percentile_threshold(z_scores: list[float], fpr: float) -> float:
        """Return (1-fpr) percentile of z_scores via linear interpolation."""
        sorted_z = sorted(z_scores)
        n = len(sorted_z)
        p = 1.0 - fpr
        idx = p * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return sorted_z[lo] + (sorted_z[hi] - sorted_z[lo]) * (idx - lo)
```

- [ ] **Step 4: 运行 calibrator 测试**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_calibrator.py -v
```

预期：全部 PASS（注意 `test_calibrate_skips_no_block_samples` 对 `fpr_threshold=0.0` 的断言）。

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/calibrator.py tests/extract/test_calibrator.py
git commit -m "feat: add ThresholdCalibrator for FPR-based threshold calibration"
```

---

### Task 6: 实现 calibrate.py CLI 脚本

**Files:**
- Create: `scripts/calibrate.py`

**背景：** CLI 脚本是用户入口，不需要单元测试（测试覆盖在 `ThresholdCalibrator`），但需要确保能正常运行。

- [ ] **Step 1: 创建 scripts/ 目录（如不存在）**

```bash
mkdir -p scripts
```

- [ ] **Step 2: 实现 scripts/calibrate.py**

```python
#!/usr/bin/env python
"""CLI tool to calibrate FPR-based watermark detection threshold.

Usage:
    python scripts/calibrate.py \\
        --input data/negative_corpus.jsonl \\
        --output threshold.json \\
        --fpr 0.01 \\
        --secret-key <key> \\
        --model data/models/codet5-base \\
        [--device cpu]

Output threshold.json example:
    {"fpr": 0.01, "fpr_threshold": 2.87, "n_samples": 500}

Set fpr_threshold as ExtractConfig.fpr_threshold before deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate FPR-based watermark detection threshold."
    )
    parser.add_argument("--input", required=True, help="Path to negative corpus JSONL")
    parser.add_argument("--output", required=True, help="Path to write threshold JSON")
    parser.add_argument(
        "--fpr", type=float, default=0.01,
        help="Target false positive rate (default: 0.01)",
    )
    parser.add_argument("--secret-key", required=True, help="Watermark secret key")
    parser.add_argument(
        "--model", required=True,
        help="Path to encoder model (e.g. data/models/codet5-base)",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--lsh-d", type=int, default=3, help="LSH projection count (default: 3)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5,
        help="LSH valid-region fraction gamma (default: 0.5)",
    )
    return parser.parse_args()


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = _parse_args()

    # Lazy imports to keep startup fast when --help is used
    import torch
    from transformers import AutoModel, AutoTokenizer

    from wfcllm.extract.calibrator import ThresholdCalibrator
    from wfcllm.extract.scorer import BlockScorer
    from wfcllm.watermark.keying import WatermarkKeying
    from wfcllm.watermark.lsh_space import LSHSpace
    from wfcllm.watermark.verifier import ProjectionVerifier

    print(f"Loading model from {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(args.device)
    encoder.eval()

    lsh_space = LSHSpace(args.secret_key, args.embed_dim, args.lsh_d)
    keying = WatermarkKeying(args.secret_key, args.lsh_d, args.gamma)
    verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=args.device)
    scorer = BlockScorer(keying, verifier)

    print(f"Loading corpus from {args.input} ...", file=sys.stderr)
    corpus = _load_jsonl(args.input)
    print(f"  {len(corpus)} samples loaded.", file=sys.stderr)

    calibrator = ThresholdCalibrator(scorer, gamma=args.gamma)
    result = calibrator.calibrate(corpus, fpr=args.fpr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Calibration complete:\n"
        f"  FPR target    : {result['fpr']}\n"
        f"  M_r threshold : {result['fpr_threshold']:.4f}\n"
        f"  Samples used  : {result['n_samples']}\n"
        f"  Output        : {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: 验证脚本语法正确**

```
conda run -n WFCLLM python -c "import ast; ast.parse(open('scripts/calibrate.py').read()); print('OK')"
```

预期：`OK`。

- [ ] **Step 4: 运行完整测试套件确认无回归**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ -v
```

预期：全部 PASS。

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate.py
git commit -m "feat: add calibrate.py CLI for FPR threshold calibration"
```

---

## Chunk 4: 最终验证

### Task 7: 全量测试 + 更新 __init__.py 导出

**Files:**
- Modify: `wfcllm/extract/__init__.py`（如需暴露 ThresholdCalibrator）

- [ ] **Step 1: 检查 __init__.py 当前导出**

```
cat wfcllm/extract/__init__.py
```

- [ ] **Step 2: 按需将 ThresholdCalibrator 加入公共导出**

若 `__init__.py` 中有其他类的导出，添加：
```python
from wfcllm.extract.calibrator import ThresholdCalibrator
```

- [ ] **Step 3: 运行全量测试**

```
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

预期：全部 PASS，无 `-1` 相关断言，无 `z_threshold` 相关引用。

- [ ] **Step 4: 确认代码中无 z_threshold 残留**

```bash
grep -r "z_threshold" wfcllm/ tests/ scripts/
```

预期：无输出。

- [ ] **Step 5: 确认代码中无 score == -1 残留**

```bash
grep -r "score == -1\|score=-1\|score: -1" wfcllm/ tests/
```

预期：无输出。

- [ ] **Step 6: 最终 Commit**

```bash
git add wfcllm/extract/__init__.py
git commit -m "chore: export ThresholdCalibrator from wfcllm.extract"
```
