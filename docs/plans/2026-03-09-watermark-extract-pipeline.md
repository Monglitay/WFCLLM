# Watermark & Extract Pipeline 实现计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 watermark 和 extract 阶段从单条处理改为面向数据集的批量 pipeline，支持 HumanEval/MBPP，生成水印 JSONL 数据集，并在 extract 阶段输出研究级统计指标。

**Architecture:** 在现有 `wfcllm/watermark/` 和 `wfcllm/extract/` 各自新增 `pipeline.py`，封装数据集加载、批量处理、JSONL I/O 逻辑；`run.py` 调用 pipeline，不再接受单条 prompt/code-file。核心模块（`generator.py`、`detector.py`）不改动。

**Tech Stack:** Python 3.10+, datasets (HuggingFace), tqdm, scipy.stats, dataclasses, pytest + unittest.mock

---

### Task 1: WatermarkPipeline — 配置与测试骨架

**Files:**
- Create: `wfcllm/watermark/pipeline.py`
- Create: `tests/watermark/test_pipeline.py`

**Step 1: 编写失败的测试（配置 dataclass）**

```python
# tests/watermark/test_pipeline.py
"""Tests for wfcllm.watermark.pipeline."""
from __future__ import annotations

import pytest
from wfcllm.watermark.pipeline import WatermarkPipelineConfig


class TestWatermarkPipelineConfig:
    def test_default_fields(self):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        assert cfg.dataset == "humaneval"
        assert cfg.output_dir == "data/watermarked"
        assert cfg.dataset_path == "data/datasets"

    def test_invalid_dataset_raises(self):
        with pytest.raises(ValueError, match="dataset must be"):
            WatermarkPipelineConfig(
                dataset="unknown",
                output_dir="data/watermarked",
                dataset_path="data/datasets",
            )
```

**Step 2: 验证测试失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py -v
```
Expected: `ImportError: cannot import name 'WatermarkPipelineConfig'`

**Step 3: 实现 `WatermarkPipelineConfig`**

```python
# wfcllm/watermark/pipeline.py
"""Batch watermarking pipeline over HumanEval/MBPP datasets."""
from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_DATASETS = ("humaneval", "mbpp")


@dataclass
class WatermarkPipelineConfig:
    """Configuration for batch watermark embedding pipeline."""

    dataset: str            # "humaneval" or "mbpp"
    output_dir: str         # e.g. "data/watermarked"
    dataset_path: str       # local datasets root, e.g. "data/datasets"

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )
```

**Step 4: 验证测试通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py -v
```
Expected: PASS (2 tests)

**Step 5: 提交**

```bash
git add wfcllm/watermark/pipeline.py tests/watermark/test_pipeline.py
git commit -m "feat: add WatermarkPipelineConfig with validation"
```

---

### Task 2: WatermarkPipeline — 数据集加载

**Files:**
- Modify: `wfcllm/watermark/pipeline.py`
- Modify: `tests/watermark/test_pipeline.py`

**Step 1: 编写失败的测试（数据集加载）**

在 `tests/watermark/test_pipeline.py` 末尾追加：

```python
from unittest.mock import patch, MagicMock
from wfcllm.watermark.pipeline import WatermarkPipeline


class TestWatermarkPipelineLoadPrompts:
    """Tests for _load_prompts() — uses mocked datasets library."""

    @pytest.fixture
    def pipeline(self):
        cfg = WatermarkPipelineConfig(
            dataset="humaneval",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        generator = MagicMock()
        return WatermarkPipeline(generator=generator, config=cfg)

    def test_load_humaneval_returns_list_of_dicts(self, pipeline):
        mock_ds = {
            "test": [
                {"task_id": "HumanEval/0", "prompt": "def foo():\n    pass\n"},
            ]
        }
        with patch("wfcllm.watermark.pipeline.load_dataset", return_value=mock_ds):
            prompts = pipeline._load_prompts()
        assert len(prompts) == 1
        assert prompts[0]["id"] == "HumanEval/0"
        assert "def foo():" in prompts[0]["prompt"]

    def test_load_mbpp_returns_list_of_dicts(self):
        cfg = WatermarkPipelineConfig(
            dataset="mbpp",
            output_dir="data/watermarked",
            dataset_path="data/datasets",
        )
        pipeline = WatermarkPipeline(generator=MagicMock(), config=cfg)
        mock_ds = {
            "train": [
                {"task_id": 1, "text": "Write a function", "code": "def f(): pass"},
            ]
        }
        with patch("wfcllm.watermark.pipeline.load_dataset", return_value=mock_ds):
            prompts = pipeline._load_prompts()
        assert len(prompts) == 1
        assert prompts[0]["id"] == "mbpp/1"
        assert "Write a function" in prompts[0]["prompt"]
```

**Step 2: 验证测试失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py::TestWatermarkPipelineLoadPrompts -v
```
Expected: `AttributeError: WatermarkPipeline` 或 `ImportError`

**Step 3: 实现 `WatermarkPipeline._load_prompts()`**

在 `wfcllm/watermark/pipeline.py` 追加（`WatermarkPipelineConfig` 之后）：

```python
from pathlib import Path
from datasets import load_dataset

from wfcllm.watermark.generator import WatermarkGenerator


class WatermarkPipeline:
    """Batch watermark embedding over a HumanEval or MBPP dataset."""

    def __init__(self, generator: WatermarkGenerator, config: WatermarkPipelineConfig):
        self._generator = generator
        self._config = config

    def _load_prompts(self) -> list[dict]:
        """Load prompts from local dataset. Returns list of {"id", "prompt"}."""
        dataset_path = str(Path(self._config.dataset_path) / self._config.dataset)

        if self._config.dataset == "humaneval":
            ds = load_dataset(
                "openai/openai_humaneval",
                cache_dir=dataset_path,
                download_mode="reuse_cache_if_exists",
            )
            prompts = []
            for split in ds:
                for item in ds[split]:
                    prompts.append({
                        "id": item["task_id"],
                        "prompt": item["prompt"],
                    })
            return prompts

        # mbpp
        ds = load_dataset(
            "google-research-datasets/mbpp",
            "full",
            cache_dir=dataset_path,
            download_mode="reuse_cache_if_exists",
        )
        prompts = []
        for split in ds:
            for item in ds[split]:
                # Use problem description as prompt; mbpp task_id is int
                prompts.append({
                    "id": f"mbpp/{item['task_id']}",
                    "prompt": item["text"],
                })
        return prompts
```

**Step 4: 验证测试通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py -v
```
Expected: PASS (all tests so far)

**Step 5: 提交**

```bash
git add wfcllm/watermark/pipeline.py tests/watermark/test_pipeline.py
git commit -m "feat: add WatermarkPipeline._load_prompts() for humaneval/mbpp"
```

---

### Task 3: WatermarkPipeline — run() 方法与 JSONL 输出

**Files:**
- Modify: `wfcllm/watermark/pipeline.py`
- Modify: `tests/watermark/test_pipeline.py`

**Step 1: 编写失败的测试（run 方法）**

```python
# 追加到 tests/watermark/test_pipeline.py
import json
import tempfile
from pathlib import Path
from wfcllm.watermark.generator import GenerateResult


class TestWatermarkPipelineRun:
    """Tests for WatermarkPipeline.run() — mocks generator and dataset."""

    @pytest.fixture
    def mock_result(self):
        return GenerateResult(
            code="def foo():\n    return 1\n",
            total_blocks=3,
            embedded_blocks=2,
            failed_blocks=1,
            fallback_blocks=0,
        )

    def test_run_creates_jsonl(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = MagicMock()
            generator.generate.return_value = mock_result

            pipeline = WatermarkPipeline(generator=generator, config=cfg)

            mock_prompts = [
                {"id": "HumanEval/0", "prompt": "def foo():\n"},
                {"id": "HumanEval/1", "prompt": "def bar():\n"},
            ]
            with patch.object(pipeline, "_load_prompts", return_value=mock_prompts):
                output_path = pipeline.run()

            # File exists
            assert Path(output_path).exists()
            assert output_path.endswith(".jsonl")

            # Parse JSONL
            lines = Path(output_path).read_text().strip().splitlines()
            assert len(lines) == 2

            record = json.loads(lines[0])
            assert record["id"] == "HumanEval/0"
            assert record["dataset"] == "humaneval"
            assert record["prompt"] == "def foo():\n"
            assert record["generated_code"] == mock_result.code
            assert record["total_blocks"] == 3
            assert record["embedded_blocks"] == 2
            assert record["failed_blocks"] == 1
            assert record["fallback_blocks"] == 0
            assert abs(record["embed_rate"] - 2/3) < 1e-6

    def test_run_returns_output_path(self, mock_result):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="mbpp",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = MagicMock()
            generator.generate.return_value = mock_result
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "mbpp/1", "prompt": "Write a function"}
            ]):
                output_path = pipeline.run()
            assert "mbpp" in output_path
            assert output_path.endswith(".jsonl")

    def test_embed_rate_zero_blocks(self):
        """embed_rate is 0.0 when total_blocks is 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = WatermarkPipelineConfig(
                dataset="humaneval",
                output_dir=tmpdir,
                dataset_path="data/datasets",
            )
            generator = MagicMock()
            generator.generate.return_value = GenerateResult(
                code="", total_blocks=0, embedded_blocks=0,
                failed_blocks=0, fallback_blocks=0,
            )
            pipeline = WatermarkPipeline(generator=generator, config=cfg)
            with patch.object(pipeline, "_load_prompts", return_value=[
                {"id": "HumanEval/0", "prompt": "def foo():"}
            ]):
                output_path = pipeline.run()
            record = json.loads(Path(output_path).read_text().strip())
            assert record["embed_rate"] == 0.0
```

**Step 2: 验证测试失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py::TestWatermarkPipelineRun -v
```
Expected: FAIL (method not implemented)

**Step 3: 实现 `WatermarkPipeline.run()`**

追加到 `wfcllm/watermark/pipeline.py` 的 `WatermarkPipeline` 类：

```python
import json
import sys
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]


def run(self) -> str:
    """Run batch watermarking. Returns path to output JSONL file."""
    prompts = self._load_prompts()
    total = len(prompts)

    out_dir = Path(self._config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{self._config.dataset}_{timestamp}.jsonl"

    iterator = (
        tqdm(prompts, desc=f"Watermarking {self._config.dataset}", unit="prompt")
        if tqdm is not None
        else prompts
    )

    with open(out_path, "w", encoding="utf-8") as f:
        for i, item in enumerate(iterator):
            result = self._generator.generate(item["prompt"])
            embed_rate = (
                result.embedded_blocks / result.total_blocks
                if result.total_blocks > 0
                else 0.0
            )
            record = {
                "id": item["id"],
                "dataset": self._config.dataset,
                "prompt": item["prompt"],
                "generated_code": result.code,
                "total_blocks": result.total_blocks,
                "embedded_blocks": result.embedded_blocks,
                "failed_blocks": result.failed_blocks,
                "fallback_blocks": result.fallback_blocks,
                "embed_rate": embed_rate,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Per-sample summary line
            summary = (
                f"  ✓ {item['id']} | "
                f"blocks: {result.embedded_blocks}/{result.total_blocks} | "
                f"failed: {result.failed_blocks} | "
                f"fallback: {result.fallback_blocks} | "
                f"embed_rate: {embed_rate:.1%}"
            )
            print(summary, file=sys.stderr)

    return str(out_path)
```

注意：将上面 `run` 方法移入 `WatermarkPipeline` 类中，并删除 `def` 前的顶层缩进。同时将 `import json`, `import sys`, `from datetime import datetime` 移到文件顶部。

**Step 4: 验证所有 watermark pipeline 测试通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py -v
```
Expected: PASS (all tests)

**Step 5: 更新 `wfcllm/watermark/__init__.py` 导出**

在文件末尾追加：
```python
from wfcllm.watermark.pipeline import WatermarkPipeline, WatermarkPipelineConfig
```

**Step 6: 提交**

```bash
git add wfcllm/watermark/pipeline.py wfcllm/watermark/__init__.py tests/watermark/test_pipeline.py
git commit -m "feat: implement WatermarkPipeline.run() with JSONL output and progress display"
```

---

### Task 4: ExtractPipeline — 配置与统计计算

**Files:**
- Create: `wfcllm/extract/pipeline.py`
- Create: `tests/extract/test_pipeline.py`

**Step 1: 编写失败的测试**

```python
# tests/extract/test_pipeline.py
"""Tests for wfcllm.extract.pipeline."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig
from wfcllm.extract.config import DetectionResult


class TestExtractPipelineConfig:
    def test_default_fields(self):
        cfg = ExtractPipelineConfig(
            input_file="data/watermarked/humaneval_20260309.jsonl",
            output_dir="data/results",
        )
        assert cfg.input_file == "data/watermarked/humaneval_20260309.jsonl"
        assert cfg.output_dir == "data/results"


def _make_detection_result(is_watermarked: bool, z_score: float) -> DetectionResult:
    return DetectionResult(
        is_watermarked=is_watermarked,
        z_score=z_score,
        p_value=0.001 if is_watermarked else 0.5,
        total_blocks=10,
        independent_blocks=8,
        hit_blocks=7 if is_watermarked else 4,
        block_details=[],
    )


class TestExtractPipelineStatistics:
    """Tests for statistical computation in run()."""

    def _make_jsonl(self, tmpdir: str, n: int = 4) -> str:
        path = Path(tmpdir) / "test.jsonl"
        records = [
            {
                "id": f"HumanEval/{i}",
                "dataset": "humaneval",
                "prompt": f"def f{i}():\n",
                "generated_code": f"def f{i}():\n    return {i}\n",
                "total_blocks": 5,
                "embedded_blocks": 3,
                "failed_blocks": 1,
                "fallback_blocks": 0,
                "embed_rate": 0.6,
            }
            for i in range(n)
        ]
        path.write_text("\n".join(json.dumps(r) for r in records))
        return str(path)

    def test_run_creates_report_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=4)
            cfg = ExtractPipelineConfig(
                input_file=jsonl_path,
                output_dir=tmpdir,
            )
            detector = MagicMock()
            # 3 watermarked, 1 not
            detector.detect.side_effect = [
                _make_detection_result(True, 4.5),
                _make_detection_result(True, 3.8),
                _make_detection_result(True, 5.1),
                _make_detection_result(False, 1.2),
            ]

            pipeline = ExtractPipeline(detector=detector, config=cfg)
            report_path = pipeline.run()

            assert Path(report_path).exists()
            assert report_path.endswith("_report.json")

            report = json.loads(Path(report_path).read_text())

            # meta
            assert report["meta"]["total_samples"] == 4
            assert report["meta"]["input_file"] == jsonl_path

            # summary
            assert abs(report["summary"]["watermark_rate"] - 0.75) < 1e-6
            assert len(report["summary"]["watermark_rate_ci_95"]) == 2
            assert report["summary"]["mean_z_score"] == pytest.approx(
                (4.5 + 3.8 + 5.1 + 1.2) / 4, abs=1e-4
            )
            assert "std_z_score" in report["summary"]
            assert "mean_p_value" in report["summary"]
            assert "mean_blocks" in report["summary"]
            assert "embed_rate_distribution" in report["summary"]

            dist = report["summary"]["embed_rate_distribution"]
            assert "mean" in dist
            assert "std" in dist
            assert "p25" in dist
            assert "p50" in dist
            assert "p75" in dist

            # per_sample
            assert len(report["per_sample"]) == 4
            first = report["per_sample"][0]
            assert first["id"] == "HumanEval/0"
            assert first["is_watermarked"] is True
            assert "z_score" in first
            assert "p_value" in first
            assert "independent_blocks" in first
            assert "hits" in first

    def test_watermark_rate_ci_lower_le_upper(self):
        """CI lower bound should be <= upper bound."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jsonl_path = self._make_jsonl(tmpdir, n=10)
            cfg = ExtractPipelineConfig(input_file=jsonl_path, output_dir=tmpdir)
            detector = MagicMock()
            detector.detect.side_effect = [
                _make_detection_result(i % 2 == 0, float(i)) for i in range(10)
            ]
            pipeline = ExtractPipeline(detector=detector, config=cfg)
            report_path = pipeline.run()
            report = json.loads(Path(report_path).read_text())
            lo, hi = report["summary"]["watermark_rate_ci_95"]
            assert lo <= hi
```

**Step 2: 验证测试失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_pipeline.py -v
```
Expected: `ImportError: cannot import name 'ExtractPipeline'`

**Step 3: 实现 `ExtractPipeline`**

```python
# wfcllm/extract/pipeline.py
"""Batch watermark extraction pipeline over a JSONL watermark dataset."""
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

try:
    import scipy.stats as _stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from wfcllm.extract.detector import WatermarkDetector


@dataclass
class ExtractPipelineConfig:
    """Configuration for batch watermark extraction pipeline."""

    input_file: str     # Path to input JSONL produced by WatermarkPipeline
    output_dir: str     # Directory for report JSON output


class ExtractPipeline:
    """Batch watermark detection and statistical reporting."""

    def __init__(self, detector: WatermarkDetector, config: ExtractPipelineConfig):
        self._detector = detector
        self._config = config

    def run(self) -> str:
        """Run batch extraction. Returns path to output report JSON."""
        records = self._load_jsonl()
        total = len(records)

        iterator = (
            tqdm(records, desc="Extracting", unit="sample")
            if tqdm is not None
            else records
        )

        per_sample = []
        z_scores = []
        p_values = []
        block_counts = []
        embed_rates = []

        for item in iterator:
            result = self._detector.detect(item["generated_code"])
            per_sample.append({
                "id": item["id"],
                "is_watermarked": result.is_watermarked,
                "z_score": result.z_score,
                "p_value": result.p_value,
                "independent_blocks": result.independent_blocks,
                "hits": result.hit_blocks,
            })
            z_scores.append(result.z_score)
            p_values.append(result.p_value)
            block_counts.append(result.independent_blocks)
            embed_rates.append(item.get("embed_rate", 0.0))

            summary_line = (
                f"  ✓ {item['id']} | "
                f"z={result.z_score:.2f} | "
                f"blocks={result.independent_blocks} | "
                f"watermarked={result.is_watermarked}"
            )
            print(summary_line, file=sys.stderr)

        watermarked = sum(1 for s in per_sample if s["is_watermarked"])
        watermark_rate = watermarked / total if total > 0 else 0.0

        report = {
            "meta": {
                "input_file": self._config.input_file,
                "total_samples": total,
            },
            "summary": {
                "watermark_rate": watermark_rate,
                "watermark_rate_ci_95": self._proportion_ci(watermarked, total),
                "mean_z_score": _mean(z_scores),
                "std_z_score": _std(z_scores),
                "mean_p_value": _mean(p_values),
                "mean_blocks": _mean(block_counts),
                "embed_rate_distribution": _distribution_stats(embed_rates),
            },
            "per_sample": per_sample,
        }

        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self._config.input_file).stem
        out_path = out_dir / f"{stem}_report.json"
        out_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return str(out_path)

    def _load_jsonl(self) -> list[dict]:
        path = Path(self._config.input_file)
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _proportion_ci(k: int, n: int, confidence: float = 0.95) -> list[float]:
        """Wilson score confidence interval for a proportion."""
        if n == 0:
            return [0.0, 0.0]
        if _HAS_SCIPY:
            lo, hi = _stats.proportion_confint(k, n, alpha=1 - confidence, method="wilson")
            return [round(lo, 6), round(hi, 6)]
        # Fallback: normal approximation
        p = k / n
        z = 1.96  # 95% CI
        margin = z * math.sqrt(p * (1 - p) / n)
        return [round(max(0.0, p - margin), 6), round(min(1.0, p + margin), 6)]


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)


def _distribution_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0}
    sorted_v = sorted(values)
    n = len(sorted_v)

    def percentile(p: float) -> float:
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * (idx - lo)

    return {
        "mean": round(_mean(values), 6),
        "std": round(_std(values), 6),
        "p25": round(percentile(0.25), 6),
        "p50": round(percentile(0.50), 6),
        "p75": round(percentile(0.75), 6),
    }
```

**Step 4: 验证所有 extract pipeline 测试通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_pipeline.py -v
```
Expected: PASS (all tests)

**Step 5: 更新 `wfcllm/extract/__init__.py` 导出**

追加：
```python
from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig
```

**Step 6: 提交**

```bash
git add wfcllm/extract/pipeline.py wfcllm/extract/__init__.py tests/extract/test_pipeline.py
git commit -m "feat: implement ExtractPipeline with research-grade statistical report"
```

---

### Task 5: 更新 `configs/base_config.json`

**Files:**
- Modify: `configs/base_config.json`

**Step 1: 替换 watermark 和 extract 节中的旧字段**

将 `configs/base_config.json` 的 `"watermark"` 节改为：

```json
"watermark": {
  "secret_key": "",
  "lm_model_path": "",
  "dataset": "humaneval",
  "dataset_path": "data/datasets",
  "output_dir": "data/watermarked",
  "encoder_model_path": "data/models/codet5-base",
  "encoder_embed_dim": 128,
  "encoder_device": "cuda",
  "margin_base": 0.1,
  "margin_alpha": 0.05,
  "max_retries": 5,
  "temperature": 0.8,
  "top_p": 0.95,
  "top_k": 50,
  "max_new_tokens": 512,
  "eos_token_id": null,
  "enable_fallback": true
}
```

将 `"extract"` 节改为：

```json
"extract": {
  "secret_key": "",
  "embed_dim": 128,
  "z_threshold": 3.0,
  "input_file": "",
  "output_dir": "data/results"
}
```

移除的旧字段：`"prompt"`, `"output_file"`, `"code_file"`.

**Step 2: 验证 JSON 格式合法**

```bash
conda run -n WFCLLM python -c "import json; json.load(open('configs/base_config.json')); print('OK')"
```
Expected: `OK`

**Step 3: 提交**

```bash
git add configs/base_config.json
git commit -m "chore: update base_config.json for dataset-driven watermark/extract pipeline"
```

---

### Task 6: 更新 `run.py` — watermark 阶段

**Files:**
- Modify: `run.py`

**Step 1: 修改 `build_parser()` 中的 watermark 参数**

在 `run.py` 的 `build_parser()` 中，找到注释 `# Watermark 参数` 这段：

```python
# Watermark 参数
parser.add_argument("--secret-key", default=None, help="水印密钥")
parser.add_argument("--lm-model-path", default=None, help="代码生成 LLM 路径")
parser.add_argument("--prompt", default=None, help="生成代码的输入提示")
parser.add_argument("--output-file", default=None, help="保存生成代码的路径")
# Extract 参数
parser.add_argument("--code-file", default=None, help="待检测代码文件路径")
parser.add_argument("--z-threshold", type=float, default=None, help="Z 分数阈值")
```

替换为：

```python
# Watermark 参数
parser.add_argument("--secret-key", default=None, help="水印密钥")
parser.add_argument("--lm-model-path", default=None, help="代码生成 LLM 路径")
parser.add_argument("--dataset", default=None, choices=["humaneval", "mbpp"],
                    help="水印嵌入数据集（humaneval 或 mbpp）")
parser.add_argument("--dataset-path", default=None, help="本地数据集根目录（默认: data/datasets）")
parser.add_argument("--output-dir", default=None, help="水印数据集输出目录（默认: data/watermarked）")
# Extract 参数
parser.add_argument("--input-file", default=None, help="待检测的水印 JSONL 文件路径")
parser.add_argument("--extract-output-dir", default=None, help="检测报告输出目录（默认: data/results）")
parser.add_argument("--z-threshold", type=float, default=None, help="Z 分数阈值")
```

**Step 2: 重写 `run_watermark()` 函数**

将现有的 `run_watermark()` 替换为：

```python
def run_watermark(args: argparse.Namespace, state: RunState) -> int:
    """阶段二：批量生成含水印代码（基于数据集）。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.watermark.config import WatermarkConfig
    from wfcllm.watermark.generator import WatermarkGenerator
    from wfcllm.watermark.pipeline import WatermarkPipeline, WatermarkPipelineConfig

    print("=== 阶段二：生成时水印嵌入 ===")

    # 前置检查
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1
    if not args.secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    if not args.lm_model_path:
        print("[错误] --lm-model-path 为必填参数", file=sys.stderr)
        return 1

    # Config 读取（优先 CLI，回退 config 文件）
    cfg = load_config(args.config)
    wm_cfg = cfg.get("watermark", {})
    dataset = args.dataset or wm_cfg.get("dataset", "humaneval")
    dataset_path = args.dataset_path or wm_cfg.get("dataset_path", "data/datasets")
    output_dir = args.output_dir or wm_cfg.get("output_dir", "data/watermarked")
    embed_dim = args.embed_dim or wm_cfg.get("encoder_embed_dim", 128)

    encoder_checkpoint = state.get("encoder", "checkpoint")

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    encoder_tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    # 加载代码生成 LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path).to(device)

    wm_config = WatermarkConfig(
        secret_key=args.secret_key,
        encoder_embed_dim=embed_dim,
        encoder_device=device,
    )
    generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, wm_config)

    pipeline_config = WatermarkPipelineConfig(
        dataset=dataset,
        output_dir=output_dir,
        dataset_path=dataset_path,
    )
    pipeline = WatermarkPipeline(generator=generator, config=pipeline_config)

    try:
        output_path = pipeline.run()
    except Exception as e:
        print(f"[错误] 水印生成失败：{e}", file=sys.stderr)
        return 1

    state.mark_done("watermark", output_file=output_path, dataset=dataset)
    print(f"[完成] 水印数据集已保存至 {output_path}")
    return 0
```

**Step 3: 验证语法无误**

```bash
conda run -n WFCLLM python -c "import ast; ast.parse(open('run.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

**Step 4: 提交**

```bash
git add run.py
git commit -m "feat: update run_watermark() to use WatermarkPipeline with dataset-driven batch processing"
```

---

### Task 7: 更新 `run.py` — extract 阶段

**Files:**
- Modify: `run.py`

**Step 1: 重写 `run_extract()` 函数**

将现有的 `run_extract()` 替换为：

```python
def run_extract(args: argparse.Namespace, state: RunState) -> int:
    """阶段三：批量检测水印（基于 JSONL 水印数据集）。"""
    import torch
    from transformers import AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.extract.config import ExtractConfig
    from wfcllm.extract.detector import WatermarkDetector
    from wfcllm.extract.pipeline import ExtractPipeline, ExtractPipelineConfig

    print("=== 阶段三：水印提取与验证 ===")

    # 前置检查
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1
    if not args.secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1

    # Config 读取（优先 CLI，回退 config 文件；input_file 也可从 run_state 中取）
    cfg = load_config(args.config)
    ext_cfg = cfg.get("extract", {})
    input_file = args.input_file or ext_cfg.get("input_file") or state.get("watermark", "output_file")
    if not input_file:
        print("[错误] --input-file 为必填参数（或先完成阶段二）", file=sys.stderr)
        return 1
    if not Path(input_file).exists():
        print(f"[错误] 文件不存在：{input_file}", file=sys.stderr)
        return 1

    output_dir = args.extract_output_dir or ext_cfg.get("output_dir", "data/results")
    embed_dim = args.embed_dim or ext_cfg.get("embed_dim", 128)
    z_threshold = args.z_threshold or ext_cfg.get("z_threshold", 3.0)

    encoder_checkpoint = state.get("encoder", "checkpoint")

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    extract_config = ExtractConfig(
        secret_key=args.secret_key,
        embed_dim=embed_dim,
        z_threshold=z_threshold,
    )
    detector = WatermarkDetector(extract_config, encoder, tokenizer, device=device)

    pipeline_config = ExtractPipelineConfig(
        input_file=input_file,
        output_dir=output_dir,
    )
    pipeline = ExtractPipeline(detector=detector, config=pipeline_config)

    try:
        report_path = pipeline.run()
    except Exception as e:
        print(f"[错误] 检测失败：{e}", file=sys.stderr)
        return 1

    import json as _json
    report = _json.loads(Path(report_path).read_text(encoding="utf-8"))
    summary = report["summary"]
    print(f"\n=== 检测结果摘要 ===")
    print(f"  样本总数:     {report['meta']['total_samples']}")
    print(f"  水印检测率:   {summary['watermark_rate']:.1%}  "
          f"95% CI [{summary['watermark_rate_ci_95'][0]:.3f}, {summary['watermark_rate_ci_95'][1]:.3f}]")
    print(f"  平均 Z 分数:  {summary['mean_z_score']:.4f} ± {summary['std_z_score']:.4f}")
    print(f"  平均 p 值:    {summary['mean_p_value']:.6f}")
    print(f"  报告已保存至: {report_path}")

    state.mark_done(
        "extract",
        report_file=report_path,
        watermark_rate=summary["watermark_rate"],
    )
    return 0
```

**Step 2: 验证语法无误**

```bash
conda run -n WFCLLM python -c "import ast; ast.parse(open('run.py').read()); print('syntax OK')"
```
Expected: `syntax OK`

**Step 3: 提交**

```bash
git add run.py
git commit -m "feat: update run_extract() to use ExtractPipeline with JSONL input and statistical report"
```

---

### Task 8: 全量测试验证

**Step 1: 运行全量测试套件**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --tb=short 2>&1 | tail -30
```
Expected: 所有测试通过，无 FAILED

**Step 2: 如有失败，逐一修复后再提交**

**Step 3: 最终提交**

```bash
git add -A
git commit -m "test: verify all tests pass after pipeline redesign"
```
