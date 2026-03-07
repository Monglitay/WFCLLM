# 节点熵值计算实验设计

## 目标

计算 AST 节点的综合信息熵，公式如下：

$$H_{\text{Node}}(N) = \frac{1}{k} \sum_{i=1}^{k} \left( -\sum_{v \in C(p_i)} P(v|p_i) \log_2 P(v|p_i) \right)$$

其中 k 为节点包含的 token 数，P(v|p_i) 为第 i 个 token 在给定前缀下的条件概率分布。

**概率分布来源：** 基于代码变换变体的统计估计（非 LLM logprobs）。

## 数据流

**输入：** `experiment/statement_block_transform/results/mbpp_blocks_transformed.json`（974 个样本，每个样本含若干语句块，每个语句块含原始代码 + 变体列表）

**处理流程：**

```
对每个样本的每个语句块：
  1. 用 tree-sitter 解析原始代码 → AST，按节点类型分组提取节点列表
  2. 对每个变体解析 → AST，按节点类型 + 出现顺序对齐
     - 第 k 个原始节点 ↔ 变体中第 k 个同类型节点
     - 变体节点数 > 原始节点数：多出部分作为独立样本
     - 变体节点数 < 原始节点数：缺失位置跳过该变体对应节点
  3. 对每个节点实例，收集所有变体的 token 序列
  4. 按 token 位置 i 统计频率 → P(v|p_i)
  5. 计算每个位置的香农熵，取平均 → H_Node
  6. 按节点类型汇总所有 H_Node 值
```

**输出：** `experiment/node_entropy/results/node_entropy_results.json`

## 模块结构

```
experiment/node_entropy/
├── config.py          # 路径配置
├── ast_utils.py       # AST 解析、节点提取、token 提取
├── entropy.py         # 熵值计算核心逻辑
├── entropy_main.py    # 主流程 CLI 入口
└── results/
    └── node_entropy_results.json
```

### 各模块职责

- **config.py** — 输入输出路径、参数配置
- **ast_utils.py** — 用 tree-sitter 解析代码，按节点类型提取节点列表，从节点 span 提取 token 序列
- **entropy.py** — 对一组 token 序列（同类型节点的所有变体），按位置统计频率分布，计算香农熵，返回 H_Node
- **entropy_main.py** — 读取 JSON 数据，遍历样本和语句块，调用上述模块，按节点类型汇总结果，写入输出文件

## 数据结构

### 中间数据结构（内存中）

```python
# 每个节点类型收集的 token 序列组
{
  "for_statement": [
    ["for", "i", "in", "range", "(", "n", ")"],  # 原始代码某节点
    ["for", "i", "in", "range", "(", "0", ",", "n", ")"],  # 变体1
    ...
  ],
  "if_statement": [...],
}
```

### 输出 JSON 格式

```json
{
  "metadata": {
    "input_file": "mbpp_blocks_transformed.json",
    "total_samples": 974,
    "timestamp": "..."
  },
  "node_type_entropy": {
    "for_statement": {
      "mean_entropy": 1.23,
      "std_entropy": 0.45,
      "sample_count": 312,
      "entropy_values": [1.1, 1.3, "..."]
    },
    "if_statement": { "..." : "..." }
  }
}
```

**字段说明：**
- `sample_count`：该节点类型收集到的节点实例总数（跨所有样本和变体）
- `entropy_values`：每个节点实例的 H_Node 值列表
- `mean_entropy` / `std_entropy`：最终统计分析结果

## 节点对齐策略

按节点类型 + 出现顺序对齐。原始代码的第 k 个 `for_statement` 对应变体中的第 k 个 `for_statement`。

- 变体节点数 > 原始：多出的节点作为独立样本单独统计
- 变体节点数 < 原始：缺失位置的变体跳过，不参与该节点的概率计算
- 初始策略为方案 A（可后续改为收集所有同类型节点方案）
