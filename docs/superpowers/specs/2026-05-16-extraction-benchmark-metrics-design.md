# 提取阶段 Benchmark 指标设计：Pass@1 / Pass@10 / AUROC

> 日期：2026-05-16
> 数据集：HumanEval、MBPP（后续扩展其他数据集）

## 背景

当前提取阶段（ExtractPipeline）只输出检测统计量（watermark_rate、mean_z_score 等），缺少两类关键评估指标：

1. **功能正确性保持**（Pass@1 / Pass@10）：水印代码是否仍能通过测试用例
2. **检测区分度**（AUROC）：检测器能否区分水印代码和非水印代码

现有 `compute_pass_at_k` 和 `compute_roc_auc` 函数已实现数学计算，但：
- Pass@k 的正确性判定只有 AST 匹配（不准确），缺少真正的测试用例执行
- AUROC 需要正/负样本对比，当前提取流程只处理单个输入文件

## 设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 正确性判定 | 执行测试用例 | 标准做法，准确 |
| 执行环境 | 直接 subprocess | 简单，当前场景够用 |
| 集成方式 | 离线评估脚本 | 不侵入主流程 |
| Pass@k 数据来源 | 支持已有结果 + 自动生成 | 灵活 |

## 架构

```
wfcllm/evaluation/
├── code_execution.py      # 已有：pass@k 数学公式、AUROC 计算
├── dual_channel.py        # 已有：保持不动
├── benchmark.py           # 新建：核心评估逻辑
└── __init__.py

scripts/evaluate.py        # 已有：新增 bench 子命令
```

## 组件设计

### 1. TestExecutor — 测试用例执行器

```python
@dataclass
class TestCase:
    task_id: str
    entry_point: str | None   # HumanEval 专用
    test_code: str            # HumanEval: test 字段; MBPP: "\n".join(test_list)

class TestExecutor:
    """执行 HumanEval/MBPP 测试用例，判定生成代码的功能正确性。"""

    def __init__(self, dataset: str, dataset_path: str, timeout: float = 5.0):
        self._test_cases: dict[str, TestCase] = load_test_cases(dataset, dataset_path)
        self._timeout = timeout

    def execute(self, task_id: str, generated_code: str) -> bool:
        """subprocess 执行测试，返回是否通过。"""
```

**HumanEval 执行逻辑：**
- 拼接：`generated_code + "\n" + test_code + f"\ncheck({entry_point})"`
- subprocess 执行，超时 5s，无异常 = 通过

**MBPP 执行逻辑：**
- 拼接：`generated_code + "\n" + test_code`
- subprocess 执行，超时 5s，无异常 = 通过

**安全控制：**
- 每次执行用独立子进程
- 超时强制 kill
- 捕获所有异常 → 返回 False

### 2. BenchmarkRunner — 评估编排器

```python
@dataclass
class BenchmarkConfig:
    dataset: str                          # "humaneval" | "mbpp"
    config_path: str                      # base_config.json 路径
    num_candidates: int = 10              # pass@k 的 k 上限
    timeout_per_test: float = 5.0         # 每个测试用例超时秒数

    # 模式一：传入已有结果
    watermarked_dirs: list[str] | None = None    # 已有水印 candidate 目录列表
    positive_details: str | None = None          # 已有正样本提取结果
    negative_details: str | None = None          # 已有负样本提取结果

    # 模式二：自动生成
    auto_generate: bool = False                  # 自动调用 watermark + extract
    negative_corpus: str | None = None           # 负样本语料库路径

    output_dir: str = "data/eval/benchmark"
```

**编排流程：**
1. 加载或生成水印 candidates（N 个 JSONL）
2. 对每个 candidate 的每个 task 执行测试用例 → 标注 `is_correct`
3. 计算 Pass@1、Pass@10
4. 加载或运行正样本提取 → positive z_scores
5. 加载或运行负样本提取 → negative z_scores
6. 计算 AUROC、TPR@1%FPR
7. 输出 BenchmarkReport JSON

### 3. 数据集加载器扩展

在 `wfcllm/datasets/loaders/local.py` 新增：

```python
def load_test_cases(dataset: str, dataset_path: str) -> dict[str, TestCase]:
    """加载 HumanEval/MBPP 的测试用例。"""
```

- HumanEval：从 dataset 中读取 `test` 和 `entry_point` 字段
- MBPP：从 dataset 中读取 `test_list` 字段

### 4. CLI 入口

在 `scripts/evaluate.py` 新增 `bench` 子命令：

```bash
# 模式一：传入已有结果
python scripts/evaluate.py bench \
  --dataset humaneval \
  --watermarked-dirs data/watermarked/run1 data/watermarked/run2 \
  --negative-details data/results/negative_details.jsonl \
  --output-dir data/eval/benchmark/humaneval

# 模式二：自动生成
python scripts/evaluate.py bench \
  --dataset humaneval \
  --config configs/base_config.json \
  --auto-generate \
  --num-candidates 10 \
  --output-dir data/eval/benchmark/humaneval
```

## AUROC 分数字段

提取结果中包含多个检测分数，AUROC 对所有可用字段分别计算：

| 字段 | 含义 | 何时可用 |
|------|------|----------|
| `z_score` | 语义通道检测分数 | 始终 |
| `lexical_z_score` | 词法通道检测分数 | token_channel.enabled=true |
| `joint_score` | 联合检测分数 | dual-channel 模式 |

输出 report 中对每个可用字段分别报告 AUROC 和 TPR@1%FPR。

## 输出格式

```json
{
  "dataset": "humaneval",
  "num_candidates": 10,
  "num_tasks": 164,
  "metrics": {
    "pass_at_1": 0.72,
    "pass_at_10": 0.89,
    "detection": {
      "semantic": { "auroc": 0.95, "tpr_at_1pct_fpr": 0.82 },
      "lexical": { "auroc": 0.88, "tpr_at_1pct_fpr": 0.65 },
      "joint": { "auroc": 0.97, "tpr_at_1pct_fpr": 0.90 }
    }
  },
  "details": {
    "execution_results_path": "data/eval/benchmark/humaneval_exec_results.jsonl",
    "positive_details_path": "...",
    "negative_details_path": "..."
  }
}
```

## 测试策略

- `tests/evaluation/test_benchmark.py`：
  - 测试 TestExecutor：用简单的正确/错误代码片段验证判定逻辑
  - 测试 BenchmarkRunner：mock subprocess 执行，验证端到端流程
  - 测试 CLI 参数解析
- 不在单元测试中加载模型或生成水印代码

## 不做的事情

- 不改动 ExtractPipeline
- 不改动 dual_channel.py
- 不引入 Docker 或额外依赖
- 不做并行执行优化（后续可加）
