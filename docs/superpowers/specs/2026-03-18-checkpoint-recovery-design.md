# 样本级断点恢复功能设计

**日期：** 2026-03-18
**作者：** Claude
**状态：** 设计阶段

## 1. 概述

为 WatermarkPipeline 和 ExtractPipeline 添加样本级断点恢复能力，允许用户在任务中断后从上次处理的位置继续执行，避免重复计算。

### 1.1 目标

- 支持任务中断后的断点恢复
- 利用输出文件本身作为断点记录
- 提供灵活的恢复方式（最新文件或指定文件）
- 保持输出格式一致性

### 1.2 非目标

- 语句块级别的断点（仅支持样本级别）
- 分布式任务的断点同步
- 自动故障检测和恢复

## 2. 用户接口

### 2.1 命令行参数

新增 `--resume` 参数，支持三种取值：

1. **不指定（默认）**：正常执行，不恢复
2. **`--resume latest`**：自动查找输出目录中最新的 JSONL 文件并恢复
3. **`--resume <path>`**：恢复指定路径的 JSONL 文件

### 2.2 使用示例

**WatermarkPipeline：**

```bash
# 首次运行
python -m experiment.embed_extract_alignment.run \
    --secret_key my-key \
    --model_path /path/to/llm \
    --n_samples 100 \
    --output_dir data/watermarked

# 中断后恢复最新文件
python -m experiment.embed_extract_alignment.run \
    --secret_key my-key \
    --model_path /path/to/llm \
    --n_samples 100 \
    --output_dir data/watermarked \
    --resume latest

# 恢复指定文件
python -m experiment.embed_extract_alignment.run \
    --secret_key my-key \
    --model_path /path/to/llm \
    --n_samples 100 \
    --output_dir data/watermarked \
    --resume data/watermarked/humaneval_20260318_120000.jsonl
```

**ExtractPipeline：**

```bash
# 首次运行
python -m wfcllm.extract.pipeline \
    --input_file data/watermarked/humaneval_20260318_120000.jsonl \
    --output_dir data/extract_reports

# 恢复最新文件
python -m wfcllm.extract.pipeline \
    --input_file data/watermarked/humaneval_20260318_120000.jsonl \
    --output_dir data/extract_reports \
    --resume latest
```

## 3. 架构设计

### 3.1 输出格式变更

**ExtractPipeline 输出格式从 JSON 改为 JSONL：**

**变更前（JSON）：**
```json
{
  "meta": {...},
  "summary": {...},
  "per_sample": [...]
}
```

**变更后（JSONL + Summary）：**
- 主输出：`{stem}.jsonl` - 每行一个样本的检测结果
- 汇总文件：`{stem}_summary.json` - 统计信息

**JSONL 格式示例：**
```jsonl
{"id": "HumanEval/0", "is_watermarked": true, "z_score": 3.45, ...}
{"id": "HumanEval/1", "is_watermarked": false, "z_score": 0.12, ...}
```

**Summary JSON 格式示例：**
```json
{
  "meta": {
    "input_file": "...",
    "total_samples": 164
  },
  "summary": {
    "watermark_rate": 0.95,
    "mean_z_score": 3.21,
    ...
  }
}
```

### 3.2 核心组件

#### 3.2.1 配置扩展

**WatermarkPipelineConfig：**
```python
@dataclass
class WatermarkPipelineConfig:
    dataset: str
    output_dir: str
    dataset_path: str
    resume: str | None = None  # None, "latest", 或文件路径
```

**ExtractPipelineConfig：**
```python
@dataclass
class ExtractPipelineConfig:
    input_file: str
    output_dir: str
    resume: str | None = None  # None, "latest", 或文件路径
```

#### 3.2.2 断点工具模块

新增 `wfcllm/common/checkpoint.py`，提供通用的断点恢复工具：

```python
def find_latest_jsonl(directory: Path, pattern: str = "*.jsonl") -> Path | None:
    """查找目录中最新的 JSONL 文件（按修改时间）"""

def load_processed_ids(jsonl_path: Path) -> set[str]:
    """从 JSONL 文件中提取已处理的样本 ID 集合"""

def resolve_resume_path(
    resume: str | None,
    output_dir: Path,
    default_pattern: str = "*.jsonl"
) -> tuple[Path | None, bool]:
    """
    解析 resume 参数
    返回: (文件路径, 是否恢复模式)
    """
```

