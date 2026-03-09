# Watermark & Extract Pipeline 重设计

**日期：** 2026-03-09
**状态：** 已批准

## 背景

当前 `run_watermark()` 接受手写的单条 `--prompt`，`run_extract()` 接受单个 `--code-file`。需要改为面向数据集的批量 pipeline，支持 HumanEval/MBPP，生成水印数据集 JSONL，并在 extract 阶段输出研究级统计指标。

## 架构方案（方案 B：新增 pipeline 层）

核心模块（`generator.py`、`detector.py`）不变，新增两个 pipeline 文件封装批量 I/O 逻辑。

### 新增文件

```
wfcllm/watermark/pipeline.py   # 数据集批量水印嵌入
wfcllm/extract/pipeline.py     # 批量水印检测与统计
```

### 数据流

```
watermark pipeline:
  dataset (humaneval/mbpp) → prompt → WatermarkGenerator.generate()
  → JSONL: data/watermarked/<dataset>_<timestamp>.jsonl

extract pipeline:
  JSONL → WatermarkDetector.detect()
  → 报告: data/results/<stem>_report.json
```

## 详细设计

### 1. WatermarkPipeline

```python
@dataclass
class WatermarkPipelineConfig:
    dataset: str          # "humaneval" or "mbpp"
    output_dir: str       # e.g. "data/watermarked"
    dataset_path: str     # 本地数据集路径，e.g. "data/datasets/"

class WatermarkPipeline:
    def __init__(self, generator: WatermarkGenerator, config: WatermarkPipelineConfig): ...
    def run(self) -> str:  # 返回输出 JSONL 路径
```

**输出 JSONL 每行格式：**

```json
{
  "id": "HumanEval/0",
  "dataset": "humaneval",
  "prompt": "...",
  "generated_code": "...",
  "total_blocks": 12,
  "embedded_blocks": 9,
  "failed_blocks": 2,
  "fallback_blocks": 1,
  "embed_rate": 0.75
}
```

**进度显示（双层）：**
```
Watermarking humaneval [50/164] ████████░░ 30%
  ✓ HumanEval/49 | blocks: 9/12 | failed: 2 | fallback: 1 | embed_rate: 75.0%
```

### 2. ExtractPipeline

```python
@dataclass
class ExtractPipelineConfig:
    input_file: str       # 输入 JSONL 路径
    output_dir: str       # e.g. "data/results"

class ExtractPipeline:
    def __init__(self, detector: WatermarkDetector, config: ExtractPipelineConfig): ...
    def run(self) -> str:  # 返回输出报告路径
```

**输出报告格式（研究级）：**

```json
{
  "meta": {
    "input_file": "data/watermarked/humaneval_20260309_120000.jsonl",
    "total_samples": 164,
    "z_threshold": 3.0
  },
  "summary": {
    "watermark_rate": 0.82,
    "watermark_rate_ci_95": [0.76, 0.88],
    "mean_z_score": 4.21,
    "std_z_score": 1.13,
    "mean_p_value": 0.0012,
    "mean_blocks": 10.3,
    "embed_rate_distribution": {
      "mean": 0.74, "std": 0.15, "p25": 0.62, "p50": 0.76, "p75": 0.88
    }
  },
  "per_sample": [
    {
      "id": "HumanEval/0",
      "is_watermarked": true,
      "z_score": 4.5,
      "p_value": 0.0003,
      "independent_blocks": 8,
      "hits": 7
    }
  ]
}
```

**进度显示（双层）：**
```
Extracting [164/164] ██████████ 100%
  ✓ HumanEval/163 | z=3.8 | blocks=9 | watermarked=True
```

### 3. configs/base_config.json 变化

```json
{
  "watermark": {
    "secret_key": "",
    "lm_model_path": "",
    "dataset": "humaneval",           // 新增，替代 --prompt
    "dataset_path": "data/datasets/", // 新增
    "output_dir": "data/watermarked", // 新增，替代 --output-file
    "margin_base": 0.1,
    "margin_alpha": 0.05,
    "max_retries": 5,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 512
  },
  "extract": {
    "secret_key": "",
    "embed_dim": 128,
    "z_threshold": 3.0,
    "input_file": "",                 // 新增，替代 --code-file
    "output_dir": "data/results"      // 新增
  }
}
```

### 4. run.py CLI 参数变化

| 阶段 | 移除 | 新增 |
|------|------|------|
| watermark | `--prompt`, `--output-file` | `--dataset` (humaneval/mbpp), `--dataset-path`, `--output-dir` |
| extract | `--code-file` | `--input-file` (JSONL路径), `--output-dir` |

CLI 参数优先级高于 config 文件（现有逻辑不变）。

## 测试要求

- `tests/watermark/test_pipeline.py`：mock `WatermarkGenerator`，验证 JSONL 输出格式与进度日志
- `tests/extract/test_pipeline.py`：mock `WatermarkDetector`，验证报告 JSON 格式与统计计算
