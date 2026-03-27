# Humaneval Watermark Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 先用已保存的 Humaneval artifacts 完成离线回归诊断与统计口径审计，再在不破坏 `FPR=0.05` 的前提下完成最小必要修复与参数回退，把后续验证推进到 `watermark_rate >= 0.45` 的可验证路径上。

**Architecture:** 先补一个纯离线的 artifact-analysis 模块，对 `details.jsonl` 和 `watermarked.jsonl` 做兼容性检查、逐样本 delta、参数重建和异常筛查；再把提取端 `z` 统计与校准端阈值计算共用同一套公式，并把 calibration regime 明确写入 summary；最后只在证据明确时走“参数回退”或“代码修复”中的一条分支，并优先用保存下来的 `watermarked` 文件做 extract-only 验证。

**Tech Stack:** Python 3.13, pytest, dataclasses, JSON/JSONL, existing `run.py` CLI, HF offline local datasets/models

**Spec source of truth:** `/home/monglitay/PycharmProjects/WFCLLM/docs/superpowers/specs/2026-03-25-humaneval-watermark-optimization-design.md`

**Working directory assumption:** Unless a command already uses absolute paths, run it from `/home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt`. Prefer absolute paths for artifact inputs/outputs when wiring CLI validation.

---

## File Structure

### New files

- `wfcllm/extract/offline_analysis.py`
  - 纯函数和 dataclass：加载 summary/details/watermarked artifacts、做 schema compatibility 检查、逐样本 delta 计算、真实参数重建、异常样本筛查、生成离线分析报告。
- `tests/extract/test_offline_analysis.py`
  - 覆盖 compatibility precheck、参数重建、flipped sample 统计、异常筛查、报告序列化。

### Existing files to modify

- `wfcllm/extract/hypothesis.py:13-104`
  - 统一 `expected_hits / variance / z_score` 计算入口，避免 detector 与 calibrator 公式漂移。
- `wfcllm/extract/calibrator.py:13-132`
  - 改为复用 `hypothesis.py` 的统计函数，并返回更明确的 calibration metadata。
- `wfcllm/extract/pipeline.py:25-242`
  - 在 summary `meta` 中写入 calibration regime / decision rule，让 `FPR=0.05` 的来源可追踪。
  - 仅在回归测试暴露 accounting 问题时做最小修复，优先限制在 metadata 应用、alignment 结果解释、hypothesis mode 决策。
- `wfcllm/extract/alignment.py:75-260`
  - 仅在回归测试暴露重建/对账口径问题时做最小修复。
- `wfcllm/watermark/gamma_schedule.py:19-75`
  - 仅在回归测试暴露 quantization / anchor 映射问题时做最小修复。
- `run.py:74-203`
  - 给 extract summary 透传 calibration metadata，并提供离线分析命令入口（只绑定已有 artifact，不触发生成）。
- `configs/base_config_B.json`
  - 只有在离线报告确认“主要是参数漂移”时，才回退到历史最佳区域（`lsh_d=4`, anchors `0.75/0.75/0.50/0.50/0.25`）。

### Existing tests to modify

- `tests/extract/test_hypothesis.py`
- `tests/extract/test_calibrator.py`
- `tests/extract/test_pipeline.py`
- `tests/extract/test_adaptive_roundtrip.py`
- `tests/extract/test_extract_param_resolution.py`
- `tests/test_run.py`
- `tests/test_run_config.py`
- `tests/watermark/test_gamma_schedule.py`

---

### Task 1: Add offline artifact compatibility and delta-analysis core

**Files:**
- Create: `wfcllm/extract/offline_analysis.py`
- Create: `tests/extract/test_offline_analysis.py`

- [ ] **Step 1: Write the failing compatibility and parameter-reconstruction tests**

```python
from wfcllm.extract.offline_analysis import (
    ArtifactCompatibility,
    compare_run_parameters,
    load_run_artifacts,
)


def test_compare_run_parameters_prefers_saved_artifact_metadata(tmp_path):
    left = tmp_path / "left.jsonl"
    left.write_text(
        '{"id":"HumanEval/0","watermark_params":{"lsh_d":4,"adaptive_gamma":{"anchors":{"p10":0.75,"p50":0.75,"p75":0.5,"p90":0.5,"p95":0.25}}}}\n',
        encoding="utf-8",
    )
    right = tmp_path / "right.jsonl"
    right.write_text(
        '{"id":"HumanEval/0","watermark_params":{"lsh_d":5,"adaptive_gamma":{"anchors":{"p10":0.9,"p50":0.75,"p75":0.6,"p90":0.5,"p95":0.25}}}}\n',
        encoding="utf-8",
    )

    diff = compare_run_parameters(load_run_artifacts(watermarked_path=left), load_run_artifacts(watermarked_path=right))

    assert diff["lsh_d"] == {"left": 4, "right": 5}
    assert diff["anchors"]["p10"] == {"left": 0.75, "right": 0.9}


def test_artifact_compatibility_requires_same_id_set(tmp_path):
    left = tmp_path / "left_details.jsonl"
    left.write_text('{"id":"HumanEval/0","z_score":1.0,"hits":1,"independent_blocks":1,"is_watermarked":false,"p_value":0.5}\n', encoding="utf-8")
    right = tmp_path / "right_details.jsonl"
    right.write_text('{"id":"HumanEval/1","z_score":1.0,"hits":1,"independent_blocks":1,"is_watermarked":false,"p_value":0.5}\n', encoding="utf-8")

    compatibility = ArtifactCompatibility.from_details(left, right)

    assert compatibility.is_comparable is False
    assert "id coverage" in compatibility.reasons[0]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_offline_analysis.py -v
```

Expected: `ModuleNotFoundError` or import failure for `wfcllm.extract.offline_analysis`.

- [ ] **Step 3: Implement the minimal offline-analysis module**

```python
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class RunArtifacts:
    summary: dict | None
    details: list[dict]
    watermarked: list[dict]


def load_run_artifacts(*, summary_path: Path | None = None, details_path: Path | None = None, watermarked_path: Path | None = None) -> RunArtifacts:
    def _load_json(path: Path | None):
        if path is None:
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _load_jsonl(path: Path | None):
        if path is None:
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    return RunArtifacts(
        summary=_load_json(summary_path),
        details=_load_jsonl(details_path),
        watermarked=_load_jsonl(watermarked_path),
    )
```

- [ ] **Step 4: Extend the tests to cover per-sample deltas and anomaly flags**

```python
def test_compare_details_flags_flip_from_true_to_false():
    left_rows = [{"id": "HumanEval/1", "is_watermarked": True, "z_score": 2.1, "hits": 8, "independent_blocks": 10, "p_value": 0.02}]
    right_rows = [{"id": "HumanEval/1", "is_watermarked": False, "z_score": 1.2, "hits": 8, "independent_blocks": 10, "p_value": 0.11}]

    report = build_detail_delta_report(left_rows, right_rows)

    assert report["flipped_to_false"] == ["HumanEval/1"]
    assert report["z_score_deltas"]["HumanEval/1"] == pytest.approx(-0.9)
```

- [ ] **Step 5: Run the focused test file and make it pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_offline_analysis.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the offline-analysis core**

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/offline_analysis.py \
  tests/extract/test_offline_analysis.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "feat: add offline watermark artifact analysis"
```

### Task 2: Add report generation for the saved Humaneval runs

**Files:**
- Modify: `wfcllm/extract/offline_analysis.py`
- Modify: `tests/extract/test_offline_analysis.py`
- Modify: `run.py:74-203`
- Modify: `tests/test_run.py`

**CLI contract for this task:** keep using `run.py --phase extract`, but add a compare-only mode that exits before normal extraction when all six compare inputs are present:
- `--compare-summary-left`
- `--compare-details-left`
- `--compare-watermarked-left`
- `--compare-summary-right`
- `--compare-details-right`
- `--compare-watermarked-right`
- `--compare-output`

This mode must write one JSON report and return `0` without starting generation or normal extraction.

- [ ] **Step 1: Write a failing CLI/report test**

```python
def test_run_offline_analysis_writes_json_report(tmp_path, monkeypatch):
    import run as run_module

    output_path = tmp_path / "report.json"
    rc = run_module.run_offline_analysis(
        argparse.Namespace(
            compare_summary_left="/home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260323_150658_summary.json",
            compare_details_left="/home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260323_150658_details.jsonl",
            compare_watermarked_left="/home/monglitay/PycharmProjects/WFCLLM/data/watermarked/humaneval_20260323_150658.jsonl",
            compare_summary_right="/home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260324_184914_summary.json",
            compare_details_right="/home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260324_184914_details.jsonl",
            compare_watermarked_right="/home/monglitay/PycharmProjects/WFCLLM/data/watermarked/humaneval_20260324_184914.jsonl",
            compare_output=str(output_path),
        )
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert report["compatibility"]["is_comparable"] is True
    assert "regression_classification" in report
```

- [ ] **Step 2: Run the focused tests to verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_offline_analysis.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run.py -k offline_analysis -v
```

Expected: missing `run_offline_analysis` / parser flags / report builder.

- [ ] **Step 3: Extend the offline-analysis tests to require an explicit branch-decision payload**

```python
def test_build_regression_report_classifies_parameter_drift_without_code_bug():
    report = build_regression_report(left_artifacts, right_artifacts)

    assert set(report["regression_classification"]) == {
        "parameter_drift",
        "adaptive_gamma_shift",
        "extraction_conservatism",
        "calibration_drift",
        "implementation_bug",
        "recommended_branch",
    }
```

- [ ] **Step 4: Implement the smallest usable report builder and `run.py` compare-only entry point**

The report must include all of the following top-level keys so Task 6 can make a branch decision without rereading artifacts manually:

```python
{
    "compatibility": {...},
    "parameter_diff": {...},
    "detail_delta": {...},
    "embedding_delta": {...},
    "anomalies": {...},
    "regression_classification": {
        "parameter_drift": bool,
        "adaptive_gamma_shift": bool,
        "extraction_conservatism": bool,
        "calibration_drift": bool,
        "implementation_bug": bool,
        "recommended_branch": "A" | "B" | "C" | "stop",
    },
}
```

- [ ] **Step 5: Run the report against the saved Humaneval artifacts**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/run.py \
  --phase extract \
  --compare-summary-left /home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260323_150658_summary.json \
  --compare-details-left /home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260323_150658_details.jsonl \
  --compare-watermarked-left /home/monglitay/PycharmProjects/WFCLLM/data/watermarked/humaneval_20260323_150658.jsonl \
  --compare-summary-right /home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260324_184914_summary.json \
  --compare-details-right /home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260324_184914_details.jsonl \
  --compare-watermarked-right /home/monglitay/PycharmProjects/WFCLLM/data/watermarked/humaneval_20260324_184914.jsonl \
  --compare-output /home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260325_offline_analysis.json
```

Expected: `/home/monglitay/PycharmProjects/WFCLLM/data/results/humaneval_20260325_offline_analysis.json` exists and includes explicit evidence for parameter drift vs adaptive-gamma shift vs extraction conservatism vs calibration drift vs implementation bug.

- [ ] **Step 6: Re-run the focused tests and make them pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_offline_analysis.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run.py -k offline_analysis -v
```

Expected: PASS.

- [ ] **Step 7: Commit the offline report entry point**

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/offline_analysis.py \
  run.py \
  tests/extract/test_offline_analysis.py \
  tests/test_run.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "feat: add humaneval offline regression report"
```

### Task 3: Make detector and calibrator share the same z-score math

**Files:**
- Modify: `wfcllm/extract/hypothesis.py:13-104`
- Modify: `wfcllm/extract/calibrator.py:13-132`
- Modify: `tests/extract/test_hypothesis.py`
- Modify: `tests/extract/test_calibrator.py`

- [ ] **Step 1: Add failing cross-module invariants**

```python
def test_calibrator_and_hypothesis_share_adaptive_distribution_parameters():
    scores = [
        BlockScore(block_id="0", score=1, min_margin=0.1, gamma_effective=0.2),
        BlockScore(block_id="1", score=0, min_margin=0.1, gamma_effective=0.8),
    ]

    expected_hits, variance = adaptive_distribution_parameters(scores, gamma=0.5, mode="adaptive")

    assert expected_hits == pytest.approx(1.0)
    assert variance == pytest.approx(0.32)
```

- [ ] **Step 2: Run the focused stats tests to verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_hypothesis.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_calibrator.py -v
```

Expected: missing shared helper / duplicated logic still present.

- [ ] **Step 3: Extract the shared math and reuse it from both call sites**

```python
def distribution_parameters(scores: list[BlockScore], gamma: float, mode: Literal["fixed", "adaptive"]) -> tuple[float, float]:
    if mode == "adaptive":
        expected_hits = sum(score.gamma_effective for score in scores)
        variance = sum(score.gamma_effective * (1 - score.gamma_effective) for score in scores)
        return expected_hits, variance
    m = len(scores)
    return m * gamma, m * gamma * (1 - gamma)


def compute_z_score(observed_hits: int, expected_hits: float, variance: float) -> float:
    if variance <= 0.0:
        ...
    return (observed_hits - expected_hits) / math.sqrt(variance)
```

- [ ] **Step 4: Re-run the stats tests and make them pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_hypothesis.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_calibrator.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit the shared statistics change**

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/hypothesis.py \
  wfcllm/extract/calibrator.py \
  tests/extract/test_hypothesis.py \
  tests/extract/test_calibrator.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "fix: unify extract and calibration z-score math"
```

### Task 4: Persist calibration regime in extract summaries

**Files:**
- Modify: `wfcllm/extract/pipeline.py:25-242`
- Modify: `run.py:126-203`
- Modify: `tests/extract/test_pipeline.py`
- Modify: `tests/test_run_config.py`

Persist the calibration regime as structured metadata, not an opaque blob. The expected shape in `summary["meta"]["calibration"]` is:

```python
{
    "source": str,
    "fpr": float,
    "threshold": float,
    "hypothesis_mode": "fixed" | "adaptive",
    "statistic_definition": str,
    "decision_rule": str,
}
```

- [ ] **Step 1: Write failing tests for calibration metadata in summary output**

```python
def test_summary_includes_declared_calibration_regime(tmp_path):
    detector = MagicMock()
    detector.detect.return_value = _make_detection_result(True, 4.5)
    pipeline = ExtractPipeline(
        detector=detector,
        config=ExtractPipelineConfig(
            input_file=str(tmp_path / "input.jsonl"),
            output_dir=str(tmp_path),
            summary_metadata={
                "calibration": {
                    "source": "data/negative_corpus.jsonl",
                    "fpr": 0.05,
                    "threshold": 1.2,
                    "hypothesis_mode": "adaptive",
                    "statistic_definition": "sum(gamma_i), sum(gamma_i*(1-gamma_i))",
                    "decision_rule": "z_score >= threshold",
                }
            },
        ),
    )
    ...
    calibration = summary_doc["meta"]["calibration"]
    assert calibration == {
        "source": "data/negative_corpus.jsonl",
        "fpr": 0.05,
        "threshold": 1.2,
        "hypothesis_mode": "adaptive",
        "statistic_definition": "sum(gamma_i), sum(gamma_i*(1-gamma_i))",
        "decision_rule": "z_score >= threshold",
    }
```

- [ ] **Step 2: Run the pipeline/config tests to verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_pipeline.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run_config.py -k calibration -v
```

Expected: `ExtractPipelineConfig` does not accept `summary_metadata`, or summary lacks one or more required calibration keys.

- [ ] **Step 3: Implement explicit summary metadata plumbing**

```python
@dataclass
class ExtractPipelineConfig:
    input_file: str
    output_dir: str
    resume: str | None = None
    summary_metadata: dict | None = None

...
summary = {
    "meta": {
        "input_file": self._config.input_file,
        "total_samples": total,
        "scored_samples": len(scored_rows),
        "invalid_samples": len(invalid_rows),
        **(self._config.summary_metadata or {}),
    },
    ...
}
```

- [ ] **Step 4: Re-run the focused tests and make them pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_pipeline.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run_config.py -k calibration -v
```

Expected: PASS.

- [ ] **Step 5: Commit the calibration-regime metadata change**

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/pipeline.py \
  run.py \
  tests/extract/test_pipeline.py \
  tests/test_run_config.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "feat: record extract calibration regime"
```

### Task 5: Add regression tests for the historical Humaneval region and fix only the failing surface

**Files:**
- Modify: `tests/watermark/test_gamma_schedule.py`
- Modify: `tests/extract/test_adaptive_roundtrip.py`
- Modify: `tests/extract/test_extract_param_resolution.py`
- Modify only if tests fail: `wfcllm/watermark/gamma_schedule.py:19-75`
- Modify only if tests fail: `wfcllm/extract/alignment.py:75-260`
- Modify only if tests fail: `wfcllm/extract/detector.py:33-184`

- [ ] **Step 1: Write failing regression tests around the saved historical best region**

```python
def test_saved_historical_best_anchor_region_roundtrips_without_numeric_mismatch(tmp_path):
    anchors = {"p10": 0.75, "p50": 0.75, "p75": 0.50, "p90": 0.50, "p95": 0.25}
    metadata, _ = _adaptive_metadata("x = 1\n", tmp_path)
    metadata["watermark_params"]["lsh_d"] = 4
    metadata["watermark_params"]["adaptive_gamma"]["anchors"] = anchors

    detector = WatermarkDetector(ExtractConfig(secret_key="k", lsh_d=4), MagicMock(), MagicMock(), device="cpu")
    ...
    assert result.contract_valid is True
    assert result.alignment_ok is True
```

- [ ] **Step 2: Run only the regression-focused tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/watermark/test_gamma_schedule.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_adaptive_roundtrip.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_extract_param_resolution.py -v
```

Expected: either PASS immediately, or a single narrow failure showing whether the problem sits in gamma quantization, alignment rebuild, or metadata application.

- [ ] **Step 3: If the failure is in gamma quantization, add a dedicated failing test in `tests/watermark/test_gamma_schedule.py`**

```python
def test_quantize_gamma_preserves_historical_anchor_resolution_for_lsh_d_4():
    resolution = quantize_gamma(0.75, lsh_d=4)
    assert resolution.k == 12
    assert resolution.gamma_effective == pytest.approx(0.75)
```

- [ ] **Step 4: If the Step 3 test fails, make the minimal `wfcllm/watermark/gamma_schedule.py` fix and rerun only that file**

```python
k_unclipped = round(clipped_gamma * universe_size)
```

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/watermark/test_gamma_schedule.py -v
```

Expected: PASS.

- [ ] **Step 5: If the failure is in alignment rebuild, add a dedicated failing test in `tests/extract/test_adaptive_roundtrip.py`**

```python
def test_alignment_prefers_embedded_adaptive_schedule_over_config_fallback(tmp_path):
    metadata, _ = _adaptive_metadata("x = 1\n", tmp_path)
    ...
    assert result.alignment_ok is True
    assert result.contract_valid is True
```

- [ ] **Step 6: If the Step 5 test fails, make the minimal `wfcllm/extract/alignment.py` fix and rerun only that file**

```python
schedule = _build_embedded_schedule(watermark_metadata) or _build_config_schedule(adaptive_gamma_config)
```

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_adaptive_roundtrip.py -v
```

Expected: PASS.

- [ ] **Step 7: If the failure is in detector metadata application, add a dedicated failing test in `tests/extract/test_extract_param_resolution.py`**

```python
def test_detector_applies_per_block_gamma_effective_from_saved_metadata(tmp_path):
    metadata, _ = _adaptive_metadata("x = 1\n", tmp_path)
    ...
    assert block_scores[0].gamma_effective == pytest.approx(0.75)
```

- [ ] **Step 8: If the Step 7 test fails, make the minimal `wfcllm/extract/detector.py` fix and rerun only that file**

```python
if score.block_id in gamma_by_block_id:
    score.gamma_effective = gamma_by_block_id[score.block_id]
```

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_extract_param_resolution.py -v
```

Expected: PASS.

- [ ] **Step 9: Re-run only the regression-focused files touched by the confirmed failure path**

If the confirmed path was gamma quantization, rerun:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/watermark/test_gamma_schedule.py -v
```

If the confirmed path was alignment rebuild, rerun:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_adaptive_roundtrip.py -v
```

If the confirmed path was detector metadata application, rerun:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_extract_param_resolution.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit only the regression guardrail files that actually changed**

If only `tests/watermark/test_gamma_schedule.py` and `wfcllm/watermark/gamma_schedule.py` changed:

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  tests/watermark/test_gamma_schedule.py \
  wfcllm/watermark/gamma_schedule.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "test: lock humaneval gamma quantization regression"
```

If only `tests/extract/test_adaptive_roundtrip.py` and `wfcllm/extract/alignment.py` changed:

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  tests/extract/test_adaptive_roundtrip.py \
  wfcllm/extract/alignment.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "test: lock humaneval alignment regression"
```

If only `tests/extract/test_extract_param_resolution.py` and `wfcllm/extract/detector.py` changed:

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  tests/extract/test_extract_param_resolution.py \
  wfcllm/extract/detector.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "test: lock humaneval detector metadata regression"
```

### Task 6: Validate with saved artifacts first, then choose exactly one optimization branch

**Files:**
- Verify only first: saved artifact outputs under `/home/monglitay/PycharmProjects/WFCLLM/data/results/` and `/home/monglitay/PycharmProjects/WFCLLM/data/watermarked/`
- Branch A modify: `configs/base_config_B.json`, `tests/test_run_config.py`
- Branch B modify: whichever of `wfcllm/extract/detector.py`, `wfcllm/extract/alignment.py`, `wfcllm/watermark/gamma_schedule.py` was proven faulty in Task 5
- Branch C modify: `wfcllm/extract/calibrator.py`, `wfcllm/extract/pipeline.py`, `tests/extract/test_calibrator.py`, `tests/extract/test_pipeline.py`

- [ ] **Step 1: Run extract-only validation on the saved watermarked artifacts**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/run.py \
  --config /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/configs/base_config_B.json \
  --phase extract \
  --input-file /home/monglitay/PycharmProjects/WFCLLM/data/watermarked/humaneval_20260323_150658.jsonl

HF_HUB_OFFLINE=1 conda run -n WFCLLM python /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/run.py \
  --config /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/configs/base_config_B.json \
  --phase extract \
  --input-file /home/monglitay/PycharmProjects/WFCLLM/data/watermarked/humaneval_20260324_184914.jsonl
```

Expected: two new extract outputs appear under `/home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/data/results/` using the project’s existing extract naming scheme, and each new summary contains `summary["meta"]["calibration"]` plus the standard `summary.watermark_rate`, `summary.mean_z_score`, and `summary.mean_blocks` keys.

- [ ] **Step 2: Use the offline report and extract-only reruns to choose exactly one branch**

Decision rule:

- Choose **Branch A** if `regression_classification.recommended_branch == "A"`, the report says `parameter_drift=True`, and `implementation_bug=False`.
- Choose **Branch B** if `regression_classification.recommended_branch == "B"`, the report says `implementation_bug=True`, and Task 5 isolated one specific faulty file.
- Choose **Branch C** if `regression_classification.recommended_branch == "C"` or `calibration_drift=True`, meaning threshold/regime derivation changed and must be re-declared before claiming `FPR=0.05` preservation.
- Choose **stop** if the report says artifacts are not comparable or if Branch A/B/C evidence is still contradictory.

- [ ] **Step 3A: If Branch A, write the failing config regression test**

```python
def test_base_config_b_restores_humaneval_best_known_region():
    cfg = json.loads(Path("configs/base_config_B.json").read_text(encoding="utf-8"))
    anchors = cfg["watermark"]["adaptive_gamma"]["anchors"]
    assert cfg["watermark"]["lsh_d"] == 4
    assert cfg["extract"]["lsh_d"] == 4
    assert anchors == {"p10": 0.75, "p50": 0.75, "p75": 0.50, "p90": 0.50, "p95": 0.25}
```

Expected before edit: FAIL against the current `configs/base_config_B.json`.

- [ ] **Step 4A: If Branch A, update `configs/base_config_B.json` to the historical-best region and rerun the config test**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run_config.py -k base_config_b -v
```

Expected: PASS.

- [ ] **Step 5A: If Branch A, commit only the config branch files**

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  configs/base_config_B.json \
  tests/test_run_config.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "fix: restore humaneval best-known config region"
```

- [ ] **Step 3B: If Branch B, write one exact failing regression test for the confirmed bug path**

If Task 5 isolated `wfcllm/extract/detector.py`, add to `tests/extract/test_extract_param_resolution.py`:

```python
def test_detector_applies_saved_gamma_effective_before_hypothesis_test(tmp_path):
    metadata, _ = _adaptive_metadata("x = 1\n", tmp_path)
    ...
    assert result.block_details[0].gamma_effective == pytest.approx(0.75)
```

If Task 5 isolated `wfcllm/extract/alignment.py`, add to `tests/extract/test_adaptive_roundtrip.py`:

```python
def test_alignment_uses_embedded_schedule_when_config_schedule_disagrees(tmp_path):
    metadata, _ = _adaptive_metadata("x = 1\n", tmp_path)
    ...
    assert result.alignment_ok is True
```

If Task 5 isolated `wfcllm/watermark/gamma_schedule.py`, add to `tests/watermark/test_gamma_schedule.py`:

```python
def test_quantize_gamma_keeps_historical_humaneval_anchor_mapping():
    resolution = quantize_gamma(0.50, lsh_d=4)
    assert resolution.k == 8
    assert resolution.gamma_effective == pytest.approx(0.5)
```

Expected before edit: FAIL in the single file corresponding to the confirmed bug path.

- [ ] **Step 4B: If Branch B, apply the minimal code fix and rerun only that focused test file**

Run the smallest relevant `pytest` target from Task 5.

Expected: PASS.

- [ ] **Step 5B: If Branch B, commit only the bug-path files that actually changed**

If the bug path was detector:

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/detector.py \
  tests/extract/test_extract_param_resolution.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "fix: align detector gamma metadata application"
```

If the bug path was alignment:

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/alignment.py \
  tests/extract/test_adaptive_roundtrip.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "fix: align adaptive schedule reconstruction"
```

If the bug path was gamma schedule:

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/watermark/gamma_schedule.py \
  tests/watermark/test_gamma_schedule.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "fix: align humaneval gamma quantization"
```

- [ ] **Step 3C: If Branch C, write failing tests for calibration-drift handling**

```python
def test_calibrator_reports_declared_regime_for_adaptive_threshold(tmp_path):
    result = calibrate_negative_corpus(...)
    assert result["mode"] == "adaptive"
    assert result["fpr"] == 0.05
    assert result["statistic_definition"]
    assert result["decision_rule"] == "z_score >= threshold"
```

- [ ] **Step 4C: If Branch C, update calibrator/pipeline metadata so the threshold regime is re-declared explicitly, then rerun only calibrator/pipeline tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_calibrator.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_pipeline.py -k calibration -v
```

Expected: PASS.

- [ ] **Step 5C: If Branch C, commit only the calibration branch files**

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt add \
  wfcllm/extract/calibrator.py \
  wfcllm/extract/pipeline.py \
  tests/extract/test_calibrator.py \
  tests/extract/test_pipeline.py
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt commit -m "fix: declare calibration regime explicitly"
```

- [ ] **Step 6: Run the narrowest end-to-end validation only if the chosen branch still requires a rerun**

Only run this step if either:
- Branch B changed code used during watermark generation or extraction and the saved-artifact evidence is not sufficient on its own, or
- Branch C changed threshold/calibration semantics and you need one fresh sample to verify the new declared regime wiring.

Skip this step entirely if Branch A fully explains the regression through saved-artifact parameter drift and the extract-only validation already matches that conclusion.

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/run.py \
  --config /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/configs/base_config_B.json \
  --phase watermark \
  --dataset humaneval \
  --sample-limit 32
```

Then run extract on the produced file:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/run.py \
  --config /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/configs/base_config_B.json \
  --phase extract \
  --input-file <watermark_output_jsonl>
```

Expected: a small-sample sanity check before any full Humaneval rerun.

- [ ] **Step 7: Do not escalate to a full Humaneval rerun unless the small-sample validation supports it**

If the small-sample sanity check regresses, stop and hand the evidence back for review instead of escalating.

### Task 7: Final verification before completion

**Files:**
- Verify only

- [ ] **Step 1: Run the full focused verification suite**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_offline_analysis.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_hypothesis.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_calibrator.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_pipeline.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_adaptive_roundtrip.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/extract/test_extract_param_resolution.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/watermark/test_gamma_schedule.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run.py \
  /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests/test_run_config.py -v
```

Expected: PASS.

- [ ] **Step 2: Run the whole project test suite only if the focused suite passes and Task 6 changed shared extract/runtime surfaces**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest /home/monglitay/PycharmProjects/WFCLLM/.worktrees/humaneval-watermark-opt/tests -v
```

Expected: PASS.

Skip this step if the final change was strictly config-only and the focused suite already covers the touched files.

- [ ] **Step 3: Record the verification evidence for handoff**

Capture:

- path to `data/results/humaneval_20260325_offline_analysis.json`
- extract-only rerun summaries for the saved 20260323 and 20260324 watermark files
- small-sample end-to-end validation summary
- exact branch chosen in Task 6 (`parameter drift` or `code bug`)

- [ ] **Step 4: Do not run a full Humaneval rerun unless the small-sample validation supports it**

If the small-sample sanity check regresses, stop here and hand the evidence back for review instead of escalating to a full rerun.
