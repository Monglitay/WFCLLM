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
- 主输出：`{stem}_details.jsonl` - 每行一个样本的检测结果
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
    """查找目录中最新的 JSONL 文件（按修改时间）

    如果目录为空或没有匹配文件，返回 None。
    如果文件大小为 0，发出警告但仍返回该文件。
    """

def load_processed_ids(jsonl_path: Path) -> set[str]:
    """从 JSONL 文件中提取已处理的样本 ID 集合

    错误处理：
    - 跳过空行
    - 捕获 JSON 解析错误，记录警告并跳过损坏的行
    - 检查 'id' 字段是否存在，缺失时记录警告并跳过
    - 如果损坏行数超过 10%，抛出 ValueError

    返回: 已处理样本的 ID 集合
    """

def resolve_resume_path(
    resume: str | None,
    output_dir: Path,
    default_pattern: str = "*.jsonl"
) -> tuple[Path | None, bool]:
    """解析 resume 参数

    参数:
        resume: None, "latest", 或文件路径
        output_dir: 输出目录
        default_pattern: 查找最新文件时使用的 glob 模式

    返回: (文件路径, 是否恢复模式)

    行为:
    - resume=None: 返回 (None, False)
    - resume="latest": 查找最新文件，未找到时返回 (None, False) 并警告
    - resume=<path>: 验证文件存在，不存在时抛出 FileNotFoundError
    """
```

### 3.3 Pipeline 实现流程

#### 3.3.1 WatermarkPipeline 断点恢复

**初始化阶段：**

```python
def run(self) -> str:
    # 1. 解析 resume 参数
    resume_path, is_resume = resolve_resume_path(
        self._config.resume,
        Path(self._config.output_dir),
        pattern=f"{self._config.dataset}_*.jsonl"
    )

    # 2. 加载已处理样本 ID（如果是恢复模式）
    processed_ids = set()
    if is_resume and resume_path:
        processed_ids = load_processed_ids(resume_path)
        print(f"Resume from {resume_path}, {len(processed_ids)} samples already processed")

    # 3. 加载并过滤 prompts
    all_prompts = self._load_prompts()
    prompts = [p for p in all_prompts if p["id"] not in processed_ids]

    if not prompts:
        print("All samples already processed")
        return str(resume_path)

    # 4. 确定输出文件路径
    if is_resume and resume_path:
        out_path = resume_path
    else:
        out_dir = Path(self._config.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"{self._config.dataset}_{timestamp}.jsonl"
```

**处理阶段：**

```python
    # 5. 以追加模式打开文件（恢复模式）或写入模式（新建模式）
    mode = "a" if is_resume else "w"
    with open(out_path, mode, encoding="utf-8") as f:
        for item in prompts:
            result = self._generator.generate(item["prompt"])
            record = {
                "id": item["id"],
                "dataset": self._config.dataset,
                "prompt": item["prompt"],
                "generated_code": result.code,
                "total_blocks": result.total_blocks,
                "embedded_blocks": result.embedded_blocks,
                "failed_blocks": result.failed_blocks,
                "fallback_blocks": result.fallback_blocks,
                "embed_rate": result.embedded_blocks / result.total_blocks if result.total_blocks > 0 else 0.0,
            }
            # 流式写入，每处理一条样本立即写入
            try:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()  # 确保立即写入磁盘
            except OSError as e:
                # 磁盘满或 I/O 错误，已写入的数据可以恢复
                print(f"Failed to write sample {item['id']}: {e}", file=sys.stderr)
                print(f"Progress saved to {out_path}, you can resume later", file=sys.stderr)
                raise

    return str(out_path)
```

#### 3.3.2 ExtractPipeline 断点恢复

**初始化阶段：**

```python
def run(self) -> str:
    # 1. 解析 resume 参数
    input_stem = Path(self._config.input_file).stem
    resume_path, is_resume = resolve_resume_path(
        self._config.resume,
        Path(self._config.output_dir),
        pattern=f"{input_stem}_details.jsonl"
    )

    # 2. 加载已处理样本 ID
    processed_ids = set()
    if is_resume and resume_path:
        processed_ids = load_processed_ids(resume_path)
        print(f"Resume from {resume_path}, {len(processed_ids)} samples already processed")

    # 3. 加载并过滤输入样本
    all_records = self._load_jsonl()
    records = [r for r in all_records if r["id"] not in processed_ids]

    if not records:
        print("All samples already processed")
        # 生成汇总文件
        self._generate_summary(resume_path)
        return str(resume_path)

    # 4. 确定输出文件路径
    out_dir = Path(self._config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if is_resume and resume_path:
        details_path = resume_path
    else:
        details_path = out_dir / f"{input_stem}_details.jsonl"
```

**处理阶段：**

```python
    # 5. 以追加模式或写入模式打开文件
    mode = "a" if is_resume else "w"
    with open(details_path, mode, encoding="utf-8") as f:
        for item in records:
            result = self._detector.detect(item["generated_code"])
            record = {
                "id": item["id"],
                "is_watermarked": result.is_watermarked,
                "z_score": result.z_score,
                "p_value": result.p_value,
                "independent_blocks": result.independent_blocks,
                "hits": result.hit_blocks,
            }
            # 流式写入
            try:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
            except OSError as e:
                print(f"Failed to write sample {item['id']}: {e}", file=sys.stderr)
                print(f"Progress saved to {details_path}, you can resume later", file=sys.stderr)
                raise

    # 6. 生成汇总文件
    summary_path = self._generate_summary(details_path)
    return str(details_path)
```

#### 3.3.3 行为差异说明

**WatermarkPipeline 和 ExtractPipeline 在"所有样本已处理"时的行为差异：**

- **WatermarkPipeline**：直接返回恢复文件路径，不做额外操作
  - 原因：JSONL 文件已经包含所有必要信息，无需额外处理

- **ExtractPipeline**：调用 `_generate_summary()` 重新生成汇总文件后返回
  - 原因：summary.json 包含统计信息，需要基于完整的 details.jsonl 重新计算
  - 场景：用户可能删除了 summary.json 但保留了 details.jsonl，此时需要重新生成

**汇总生成：**

```python
def _generate_summary(self, details_path: Path) -> Path:
    """从 details JSONL 生成 summary JSON"""
    # 验证文件存在
    if not details_path.exists():
        raise FileNotFoundError(f"Details file not found: {details_path}")

    # 重新读取完整的 details 文件
    all_results = []
    with open(details_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_results.append(json.loads(line))

    # 计算统计信息
    total = len(all_results)
    watermarked = sum(1 for r in all_results if r["is_watermarked"])
    # ... 其他统计计算

    summary = {
        "meta": {"input_file": self._config.input_file, "total_samples": total},
        "summary": {"watermark_rate": watermarked / total if total > 0 else 0.0, ...}
    }

    # 健壮的文件名生成逻辑
    base_stem = details_path.stem
    if base_stem.endswith("_details"):
        base_stem = base_stem[:-8]  # 移除 "_details" 后缀
    summary_path = details_path.parent / f"{base_stem}_summary.json"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary_path
```

## 4. 错误处理策略

### 4.1 文件不存在

**场景：** 用户指定的 resume 文件不存在

**处理：**
- 抛出 `FileNotFoundError` 并提示正确的文件路径
- 如果是 `--resume latest` 但目录为空，提示"未找到可恢复的文件，将以新任务模式运行"

### 4.2 文件格式错误

**场景：** JSONL 文件格式损坏或不符合预期

**处理：**
- 在 `load_processed_ids` 中捕获 JSON 解析错误
- 跳过损坏的行，记录警告日志
- 如果损坏行数超过 10%，抛出异常并建议用户检查文件

### 4.3 ID 冲突

**场景：** 恢复文件中的样本 ID 与当前数据集不匹配

**处理：**
- 在过滤阶段自动跳过不匹配的 ID
- 如果恢复文件中的 ID 完全不在当前数据集中，发出警告

### 4.4 配置不匹配

**场景：** 恢复文件的配置（如 dataset）与当前配置不一致

**处理：**
- WatermarkPipeline：检查恢复文件名中的 dataset 前缀是否匹配
- 如果不匹配，抛出 `ValueError` 并提示用户
- ExtractPipeline：检查 input_file 的 stem 是否与恢复文件匹配

### 4.5 磁盘空间不足

**场景：** 写入过程中磁盘空间不足

**处理：**
- 依赖操作系统的 `OSError` 异常
- 捕获后提示用户清理磁盘空间
- 已写入的数据保留，可以再次恢复

## 5. 测试策略

### 5.1 单元测试

**测试模块：** `tests/common/test_checkpoint.py`

**测试用例：**
1. `test_find_latest_jsonl_empty_dir` - 空目录返回 None
2. `test_find_latest_jsonl_multiple_files` - 多个文件返回最新的
3. `test_load_processed_ids_valid` - 正确提取 ID 集合
4. `test_load_processed_ids_malformed` - 处理损坏的 JSONL 行
5. `test_resolve_resume_path_none` - resume=None 返回 (None, False)
6. `test_resolve_resume_path_latest` - resume="latest" 查找最新文件
7. `test_resolve_resume_path_explicit` - resume=<path> 返回指定路径

### 5.2 集成测试

**测试模块：** `tests/watermark/test_pipeline_checkpoint.py`

**测试用例：**
1. `test_watermark_pipeline_resume_latest` - 中断后恢复最新文件
2. `test_watermark_pipeline_resume_explicit` - 恢复指定文件
3. `test_watermark_pipeline_resume_all_processed` - 所有样本已处理
4. `test_watermark_pipeline_resume_partial` - 部分样本已处理

**测试模块：** `tests/extract/test_pipeline_checkpoint.py`

**测试用例：**
1. `test_extract_pipeline_resume_latest` - 中断后恢复最新文件
2. `test_extract_pipeline_resume_explicit` - 恢复指定文件
3. `test_extract_pipeline_output_format` - 验证 JSONL + Summary 输出格式
4. `test_extract_pipeline_summary_generation` - 验证汇总文件正确性

### 5.3 边界情况测试

1. **空数据集** - 0 个样本的情况
2. **单样本** - 只有 1 个样本的情况
3. **文件权限** - 只读目录、无写权限的情况
4. **并发写入** - 多个进程同时写入同一文件（不支持，应报错）
5. **超大文件** - 10000+ 样本的恢复性能测试
6. **空 JSONL 文件** - 恢复一个 0 字节的文件
7. **损坏的 JSONL 文件** - 部分行格式错误或缺少 ID 字段
8. **数据集顺序变化** - 恢复时数据集样本顺序与原始不同
9. **文件名包含特殊字符** - 确保 `_details` 后缀处理的健壮性

## 6. 实现清单

### 6.1 新增文件

1. **`wfcllm/common/checkpoint.py`**
   - `find_latest_jsonl()` - 查找最新 JSONL 文件
   - `load_processed_ids()` - 加载已处理样本 ID
   - `resolve_resume_path()` - 解析 resume 参数

2. **`tests/common/test_checkpoint.py`**
   - 断点工具模块的单元测试

3. **`tests/watermark/test_pipeline_checkpoint.py`**
   - WatermarkPipeline 断点恢复的集成测试

4. **`tests/extract/test_pipeline_checkpoint.py`**
   - ExtractPipeline 断点恢复的集成测试

### 6.2 修改文件

1. **`wfcllm/watermark/pipeline.py`**
   - `WatermarkPipelineConfig` 添加 `resume` 字段
   - `WatermarkPipeline.run()` 实现断点恢复逻辑
   - 添加 `f.flush()` 确保流式写入

2. **`wfcllm/extract/pipeline.py`**
   - `ExtractPipelineConfig` 添加 `resume` 字段
   - `ExtractPipeline.run()` 改为 JSONL 输出 + 断点恢复
   - 新增 `_generate_summary()` 方法生成汇总文件
   - 输出文件名从 `{stem}_report.json` 改为 `{stem}_details.jsonl` + `{stem}_summary.json`

3. **`experiment/embed_extract_alignment/run.py`** (或相应的入口脚本)
   - 添加 `--resume` 命令行参数
   - 传递给 `WatermarkPipelineConfig`

4. **CLI 入口脚本** (如果 ExtractPipeline 有独立入口)
   - 添加 `--resume` 命令行参数
   - 传递给 `ExtractPipelineConfig`

### 6.3 文档更新

1. **`README.md`**
   - 添加断点恢复功能的使用说明

2. **`docs/` 相关文档**
   - 更新 Pipeline 使用文档
   - 说明 ExtractPipeline 输出格式变更

## 7. 实现顺序

1. **阶段一：基础工具**
   - 实现 `wfcllm/common/checkpoint.py`
   - 编写单元测试 `tests/common/test_checkpoint.py`

2. **阶段二：WatermarkPipeline 断点恢复**
   - 修改 `WatermarkPipelineConfig` 和 `WatermarkPipeline.run()`
   - 编写集成测试 `tests/watermark/test_pipeline_checkpoint.py`
   - 更新入口脚本添加 `--resume` 参数

3. **阶段三：ExtractPipeline 输出格式变更**
   - 修改 `ExtractPipeline.run()` 输出为 JSONL
   - 实现 `_generate_summary()` 方法
   - 更新相关测试以适配新格式

4. **阶段四：ExtractPipeline 断点恢复**
   - 在 ExtractPipeline 中集成断点恢复逻辑
   - 编写集成测试 `tests/extract/test_pipeline_checkpoint.py`
   - 更新入口脚本添加 `--resume` 参数

5. **阶段五：文档和边界测试**
   - 更新 README 和相关文档
   - 补充边界情况测试
   - 性能测试（大规模数据集）

## 8. 风险与限制

### 8.1 风险

1. **并发写入冲突**
   - 多个进程同时恢复同一文件会导致数据损坏
   - 缓解方案：
     - 在文档中明确说明不支持并发恢复
     - **可选实现**：使用文件锁机制（`fcntl.flock` 或 `filelock` 库）在打开文件时检测并发访问，如果文件已被锁定则抛出异常

2. **磁盘空间不足**
   - 大规模数据集可能耗尽磁盘空间
   - 缓解：在写入前检查可用空间，提前警告

3. **配置漂移**
   - 恢复时使用的配置（如 secret_key）与原始配置不一致
   - 缓解：在文档中强调必须使用相同配置

### 8.2 限制

1. **样本级断点**
   - 不支持语句块级别的断点
   - 如果单个样本处理时间很长，中断仍会损失该样本的进度

2. **无自动故障恢复**
   - 不会自动检测进程崩溃并重启
   - 用户需要手动执行 `--resume latest`

3. **单机限制**
   - 不支持分布式任务的断点同步
   - 仅适用于单机单进程场景

## 9. 未来扩展

1. **自动恢复模式**
   - 添加 `--auto-resume` 参数，自动检测并恢复最新文件

2. **进度条优化**
   - 在恢复模式下，进度条显示"已完成 X / 总共 Y"

3. **断点元数据**
   - 在 JSONL 文件头部添加元数据注释（如配置哈希）
   - 用于验证配置一致性

4. **增量汇总**
   - ExtractPipeline 在处理过程中增量更新 summary，而不是最后重新计算
