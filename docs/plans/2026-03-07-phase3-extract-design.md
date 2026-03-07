# Phase 3 设计：提取与验证 —— 无参考水印检测流水线

## 目标

在不依赖原始代码的情况下，从待测代码中提取语句块样本，通过统计学检验确证水印的存在。

## 架构：流水线组件式

```
wfcllm/extract/
├── __init__.py          # 公共 API 导出（双层：高层 + 底层组件）
├── config.py            # ExtractConfig, DetectionResult, BlockScore
├── scorer.py            # BlockScorer - 语义特征打分
├── dp_selector.py       # DPSelector - DP 去重 + 独立样本选取
├── hypothesis.py        # HypothesisTester - Z-score 显著性检验
└── detector.py          # WatermarkDetector - 高层一体化入口
```

与 Phase 2 风格一致：每个步骤独立为一个类，高层入口串联整个流程。

## 组件间复用

直接 import Phase 2 的以下组件（不重新实现）：

- `WatermarkKeying`：从密钥 + AST 拓扑派生 (v, t)
- `ProjectionVerifier`：编码代码并计算余弦投影
- `SemanticEncoder`：Phase 1 的编码器

直接 import common 的：

- `extract_statement_blocks()`：AST 解析与语句块提取
- `StatementBlock`：语句块数据结构

## 数据结构

### ExtractConfig

```python
@dataclass
class ExtractConfig:
    secret_key: str           # 与生成时相同的密钥
    embed_dim: int = 128      # 编码器输出维度
    z_threshold: float = 3.0  # Z-score 判定阈值
```

### BlockScore

```python
@dataclass
class BlockScore:
    block_id: str
    score: int            # S_i ∈ {+1, -1}
    projection: float     # 余弦投影值 p
    target_sign: int      # 目标符号 t* ∈ {-1, +1}
    selected: bool        # 是否被 DP 选中为独立样本
```

### DetectionResult

```python
@dataclass
class DetectionResult:
    is_watermarked: bool
    z_score: float
    p_value: float
    total_blocks: int           # 代码中总语句块数
    independent_blocks: int     # DP 去重后独立块数 M
    hit_blocks: int             # 命中数 X
    block_details: list[BlockScore]  # 所有块的详情（selected 标记 DP 选中）
```

## 组件设计

### 1. BlockScorer（scorer.py）

对每个语句块计算水印命中得分。

- 输入：`StatementBlock` 列表
- 输出：`BlockScore` 列表
- 复用 `WatermarkKeying.derive()` 获取 (v, t)
- 复用 `ProjectionVerifier.verify()` 计算投影
- **提取阶段只看投影符号，不看裕度**（方案文档明确：仅验证 `sgn(p_i) == t_i*`）
- 调用 `verify()` 时传 `margin=0.0`，由投影符号决定 S_i
- 根级块的 parent_node_type 为 `"module"`

### 2. DPSelector（dp_selector.py）

DP 去重 + 回溯选取互不嵌套的独立样本集。Phase 3 唯一的核心新算法。

**自底向上计算 OPT(i)：**

```
OPT(i) = max(S_i, Σ OPT(j) for j in children(i))
```

- 按 depth 降序遍历（叶子先处理）
- 叶子节点：`OPT(i) = S_i`
- 非叶子：比较自身得分 vs 子节点得分之和，记录选择

**自顶向下回溯选取独立集：**

- 从根级块（`parent_id is None`）开始
- `use_self == True`：将该块加入独立集
- `use_self == False`：递归处理其子节点
- 结果是一组互不嵌套的块

### 3. HypothesisTester（hypothesis.py）

Z-score 单侧显著性检验。

```
M = 独立块数
X = 命中数 (score == +1)
Z = (X - M/2) / sqrt(M/4)
p_value = 1 - Φ(Z)  (scipy.stats.norm.sf)
is_watermarked = Z > z_threshold
```

- 单侧检验：只关心命中率显著高于 0.5
- M = 0 时返回 `is_watermarked=False, z_score=0.0`

### 4. WatermarkDetector（detector.py）

高层一体化入口。

```python
def detect(self, code: str) -> DetectionResult:
    blocks = extract_statement_blocks(code)
    scores = self._scorer.score_all(blocks)
    selected_ids = self._dp.select(blocks, scores)
    selected_scores = [s for s in scores if s.block_id in selected_ids]
    return self._tester.test(selected_scores, total_blocks=len(blocks))
```

## 公共 API

```python
# 高层 API
from wfcllm.extract import WatermarkDetector, ExtractConfig, DetectionResult

# 底层组件
from wfcllm.extract import BlockScorer, BlockScore
from wfcllm.extract import DPSelector
from wfcllm.extract import HypothesisTester
```

## 数据流

```
源代码 -> extract_statement_blocks() -> [StatementBlock]
       -> BlockScorer.score_all()     -> [BlockScore]
       -> DPSelector.select()         -> 独立 block_id 集合
       -> HypothesisTester.test()     -> DetectionResult
```
