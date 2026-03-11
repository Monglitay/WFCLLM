# Extraction Simple-Block-Only Refactor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the low watermark detection rate (8.5%) by removing compound block noise from the extraction pipeline — only score simple blocks.

**Architecture:** The extraction pipeline currently scores all AST blocks (simple + compound), but only simple blocks carry watermark signal. We filter to simple blocks only in detector, scorer, and calibrator. Since simple blocks are AST leaves and never nest, DP deduplication becomes unnecessary.

**Tech Stack:** Python, pytest, tree-sitter, PyTorch (encoder inference)

**Spec:** `docs/superpowers/specs/2026-03-11-extraction-simple-only-design.md`

**Test command:** `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ -v`

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `wfcllm/extract/scorer.py` | Modify | Change `score_all` signature to accept `(target_blocks, all_blocks)` |
| `wfcllm/extract/detector.py` | Modify | Filter simple blocks, skip DP, pass dual args to scorer |
| `wfcllm/extract/calibrator.py` | Modify | Filter simple blocks, skip DP, pass dual args to scorer |
| `tests/extract/test_scorer.py` | Modify | Update `score_all` calls to dual-parameter form |
| `tests/extract/test_detector.py` | Modify | Update assertions for simple-only semantics |
| `tests/extract/test_calibrator.py` | Modify | Remove DP mock, update scorer mock pattern |

---

## Chunk 1: Scorer Refactor

### Task 1: Update `scorer.score_all` signature

**Files:**
- Modify: `wfcllm/extract/scorer.py:32-33`
- Test: `tests/extract/test_scorer.py`

- [ ] **Step 1: Update test for new `score_all` signature**

In `tests/extract/test_scorer.py`, the test `test_score_all_returns_all_blocks` (line 106) calls `scorer.score_all(blocks)` with one arg. Update it to pass `(target_blocks, all_blocks)`.

```python
# tests/extract/test_scorer.py – inside class TestBlockScorer
def test_score_all_returns_all_blocks(self, keying, mock_verifier):
    mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.3)
    scorer = BlockScorer(keying, mock_verifier)
    blocks = [_make_block("0"), _make_block("1")]
    # New: pass (target_blocks, all_blocks)
    results = scorer.score_all(blocks, blocks)

    assert len(results) == 2
    assert results[0].block_id == "0"
    assert results[1].block_id == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py::TestBlockScorer::test_score_all_returns_all_blocks -v`
Expected: FAIL — `score_all()` takes 2 positional arguments but 3 were given

- [ ] **Step 3: Update `scorer.py` implementation**

Change `score_all` in `wfcllm/extract/scorer.py`:

```python
def score_all(
    self,
    target_blocks: list[StatementBlock],
    all_blocks: list[StatementBlock],
) -> list[BlockScore]:
    """Score target blocks. all_blocks needed for parent_id → node_type lookup."""
    return [self.score_block(b, all_blocks) for b in target_blocks]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_scorer.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/scorer.py tests/extract/test_scorer.py
git commit -m "refactor: scorer.score_all accepts (target_blocks, all_blocks)"
```

---

## Chunk 2: Detector Refactor

### Task 2: Update `detector.detect` to simple-only

**Files:**
- Modify: `wfcllm/extract/detector.py`
- Test: `tests/extract/test_detector.py`

- [ ] **Step 1: Update detector tests for simple-only semantics**

In `tests/extract/test_detector.py`, update the following tests (all are methods in `class TestWatermarkDetector`, so keep `self` as first param):

**`test_detect_returns_detection_result` (line 37):** Current input `"x = 1\ny = 2\n"` has no compound blocks. Replace with compound+simple input to verify compound blocks are excluded. Add assertion `independent_blocks == total_blocks`:

```python
def test_detect_returns_detection_result(
    self, config, mock_encoder, mock_tokenizer
):
    detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
    # Use compound+simple input to verify compound block is excluded
    code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
    result = detector.detect(code)
    assert isinstance(result, DetectionResult)
    # Only simple blocks counted (x = i + 1, y = i * 2), not the for compound block
    assert result.total_blocks == 2
    assert result.independent_blocks == result.total_blocks  # all simple blocks selected
    assert isinstance(result.z_score, float)
    assert isinstance(result.p_value, float)
```

**`test_block_details_include_all_blocks` (line 58):** Rename and update — block_details now only includes simple blocks:

```python
def test_block_details_include_simple_blocks(
    self, config, mock_encoder, mock_tokenizer
):
    """block_details should contain only simple blocks, not compound blocks."""
    detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
    code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
    result = detector.detect(code)
    assert len(result.block_details) == result.total_blocks
    # total_blocks should be 2 (only the simple blocks), not 3
    assert result.total_blocks == 2
```

**`test_selected_flag_set` (line 70):** All simple blocks are now always selected:

```python
def test_selected_flag_set(self, config, mock_encoder, mock_tokenizer):
    """All simple blocks should have selected=True."""
    detector = WatermarkDetector(config, mock_encoder, mock_tokenizer, device="cpu")
    code = "for i in range(10):\n    x = i + 1\n    y = i * 2\n"
    result = detector.detect(code)
    # All simple blocks are selected (no DP filtering)
    assert all(s.selected for s in result.block_details)
    assert result.independent_blocks == result.total_blocks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py -v`
Expected: At least 2-3 tests FAIL (old detector still scores compound blocks)

- [ ] **Step 3: Update `detector.py` implementation**

Replace the content of `wfcllm/extract/detector.py`:

```python
"""High-level watermark detection entry point."""

from __future__ import annotations

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.config import DetectionResult, ExtractConfig
from wfcllm.extract.hypothesis import HypothesisTester
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier


class WatermarkDetector:
    """One-call watermark detection pipeline."""

    def __init__(
        self,
        config: ExtractConfig,
        encoder,
        tokenizer,
        device: str = "cuda",
    ):
        lsh_space = LSHSpace(config.secret_key, config.embed_dim, config.lsh_d)
        keying = WatermarkKeying(config.secret_key, config.lsh_d, config.lsh_gamma)
        verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=device)
        self._scorer = BlockScorer(keying, verifier)
        self._tester = HypothesisTester(config.fpr_threshold, gamma=config.lsh_gamma)

    def detect(self, code: str) -> DetectionResult:
        blocks = extract_statement_blocks(code)
        if not blocks:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_blocks=0, independent_blocks=0, hit_blocks=0,
                block_details=[],
            )

        # Only simple blocks carry watermark signal
        simple_blocks = [b for b in blocks if b.block_type == "simple"]
        if not simple_blocks:
            return DetectionResult(
                is_watermarked=False, z_score=0.0, p_value=1.0,
                total_blocks=0, independent_blocks=0, hit_blocks=0,
                block_details=[],
            )

        # all_blocks passed for parent_id → node_type lookup
        scores = self._scorer.score_all(simple_blocks, blocks)

        # Simple blocks are AST leaves — inherently non-overlapping, skip DP
        result = self._tester.test(scores, total_blocks=len(simple_blocks))
        for s in scores:
            s.selected = True
        result.block_details = scores
        return result
```

- [ ] **Step 4: Run tests to verify pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/detector.py tests/extract/test_detector.py
git commit -m "refactor: detector filters simple blocks only, removes DP dependency"
```

---

## Chunk 3: Calibrator Refactor

### Task 3: Update calibrator to simple-only

**Files:**
- Modify: `wfcllm/extract/calibrator.py`
- Test: `tests/extract/test_calibrator.py`

- [ ] **Step 1: Update calibrator tests**

In `tests/extract/test_calibrator.py`:

**`test_calibrate_returns_dict_with_required_keys` (line 35):** The mock scorer's `score_all` is called internally. The existing mock `mock_scorer.score_all.return_value = [...]` should still work since MagicMock accepts any arguments. However, we should verify the calibrator no longer uses DP. No test code changes needed for this specific test since it mocks the scorer.

**All tests:** Since the calibrator no longer uses `DPSelector`, remove any dependency on it from the test fixtures. The current fixture `mock_scorer` at line 30 is a plain MagicMock — it already doesn't mock DP. The real `ThresholdCalibrator.__init__` currently creates `self._dp = DPSelector()`, so after we remove that, the test's `ThresholdCalibrator(mock_scorer)` call still works.

No test modifications needed for existing tests — the existing tests mock `score_all` at the scorer level, and the calibrator's internal change (removing DP) is transparent to mocks. But let's add one test to verify simple-block filtering behavior (inside `class TestThresholdCalibrator`):

```python
def test_calibrate_filters_simple_blocks_only(self, mock_scorer):
    """Calibrator should only score simple blocks, not compound."""
    mock_scorer.score_all.return_value = [
        BlockScore(block_id="1", score=1, min_margin=0.5),
    ]
    calibrator = ThresholdCalibrator(mock_scorer)
    # Code with compound (for) + simple (x = 1) blocks
    corpus = [{"generated_code": "for i in range(10):\n    x = 1\n"}]
    result = calibrator.calibrate(corpus, fpr=0.01)
    # score_all should be called with simple blocks only as first arg
    call_args = mock_scorer.score_all.call_args
    target_blocks = call_args[0][0]
    assert all(b.block_type == "simple" for b in target_blocks)
```

- [ ] **Step 2: Run tests to verify current state**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_calibrator.py -v`
Expected: Existing tests PASS, new test FAILS (calibrator still passes all blocks)

- [ ] **Step 3: Update `calibrator.py` implementation**

Replace content of `wfcllm/extract/calibrator.py`:

```python
"""Offline FPR threshold calibration for watermark detection."""

from __future__ import annotations

import math

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.extract.scorer import BlockScorer


class ThresholdCalibrator:
    """Compute FPR-based detection threshold M_r from a negative corpus.

    Uses BlockScorer to compute Z scores for each sample's simple blocks,
    then returns the (1-fpr) percentile as the threshold M_r.
    """

    def __init__(self, scorer: BlockScorer, gamma: float = 0.5):
        self._scorer = scorer
        self._gamma = gamma

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

            # Only simple blocks carry watermark signal
            simple_blocks = [b for b in blocks if b.block_type == "simple"]
            if not simple_blocks:
                continue

            scores = self._scorer.score_all(simple_blocks, blocks)

            m = len(scores)
            if m == 0:
                continue

            x = sum(1 for s in scores if s.score == 1)
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

- [ ] **Step 4: Run all tests to verify pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_calibrator.py -v`
Expected: All tests PASS (including new test)

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/calibrator.py tests/extract/test_calibrator.py
git commit -m "refactor: calibrator filters simple blocks only, removes DP dependency"
```

---

## Chunk 4: Full Suite Validation

### Task 4: Run full test suite and fix regressions

**Files:** All modified files above

- [ ] **Step 1: Run full extract test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ -v`
Expected: All tests PASS

- [ ] **Step 2: Run full project test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v`
Expected: All tests PASS. If any fail, investigate — they may be importing from extract modules with the old signature.

- [ ] **Step 3: Fix any regressions found**

If tests fail, check for:
- Other callers of `scorer.score_all()` using old single-arg signature
- Tests importing `DPSelector` from `detector` or `calibrator` modules
- Pipeline tests that assert on `total_blocks` or `independent_blocks` values

- [ ] **Step 4: Final commit if fixes needed**

```bash
git add -u
git commit -m "fix: resolve test regressions from extraction simple-only refactor"
```

---

## Chunk 5: Config Mismatch Warning (Informational)

### Task 5: Document config mismatch between watermark and extract

**Note:** During analysis, a config mismatch was found in `configs/base_config.json`:

| Parameter | `watermark` section | `extract` section |
|-----------|-------------------|-----------------|
| `lsh_d` | 4 | 3 |
| `lsh_gamma` | 0.75 | 0.5 |

If these were used for the actual run, the embedding and extraction would use **different LSH parameters**, making extraction impossible. The code defaults (`WatermarkConfig.lsh_d=3`, `ExtractConfig.lsh_d=3`) match, so the actual run likely used defaults or CLI overrides rather than this config file.

- [ ] **Step 1: Verify which config was used for the run**

Check the watermark log header or `run.py` to confirm whether `base_config.json` was used or if CLI defaults were applied.

- [ ] **Step 2: Fix config if needed**

If `base_config.json` is intended to be the canonical config, align the `lsh_d` and `lsh_gamma` values between watermark and extract sections. Both should use the same values.
