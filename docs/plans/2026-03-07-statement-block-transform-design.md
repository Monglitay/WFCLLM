# Statement Block Transform Design

## Context

WFCLLM 项目的第二阶段实验：对已分割的语句块进行语义等价代码变换，生成多样化的代码变体作为 LLM 代码水印的训练数据。

- 输入：`experiment/statement_block_split/results/mbpp_blocks.json`（974 个 MBPP 样本的语句块）
- 变换规则：`docs/python代码变换规则.csv`（39 条规则，6 个类别）
- 工具：tree-sitter AST 模式匹配 + 节点替换
- 环境：conda WFCLLM, Python

## 目的

通过语义等价变换生成代码变体，用于训练代码 LLM 水印模型。每个语句块检测所有可行规则，对可行规则生成所有长度的排列组合（设上限截断），产生多样化的训练数据。

## Directory Structure

```
experiment/
└── statement_block_transform/
    ├── config.py                      # 配置（路径、截断上限等）
    ├── engine.py                      # TransformEngine 核心引擎
    ├── rules/                         # 规则模块
    │   ├── __init__.py                # 导出所有规则
    │   ├── base.py                    # Rule 基类
    │   ├── api_calls.py              # API与函数调用类规则 (11条)
    │   ├── syntax_init.py            # 语法与初始化类规则 (4条)
    │   ├── control_flow.py           # 控制流类规则 (4条)
    │   ├── expression_logic.py       # 表达式与逻辑类规则 (5条)
    │   ├── identifier.py             # 标识符操作类规则 (2条)
    │   └── formatting.py             # 代码格式化类规则 (3条)
    ├── transform.py                   # 主处理流程
    └── results/
        └── mbpp_blocks_transformed.json
```

## Dependencies

无新增依赖，复用 Phase 1 的 `tree-sitter` + `tree-sitter-python` + `datasets`。

## Core Architecture

### Rule Base Class

```python
class Rule:
    name: str           # 规则名称，如 "explicit_default_print"
    category: str       # 类别，如 "API与函数调用"
    description: str    # 描述

    def detect(self, source: str, tree: Tree) -> list[Match]:
        """检测语句块中所有可应用此规则的位置，返回匹配列表"""
        ...

    def apply(self, source: str, matches: list[Match]) -> str:
        """对源代码应用变换，返回变换后的代码"""
        ...
```

### TransformEngine

```python
class TransformEngine:
    rules: list[Rule]
    max_permutation_length: int   # 排列最大长度，默认 5
    max_variants_per_block: int   # 每个语句块最大变体数，默认 1000

    def get_applicable_rules(self, block: dict) -> list[Rule]:
        """对一个语句块检测所有可行规则"""

    def generate_variants(self, block: dict) -> list[dict]:
        """生成所有排列变体（受上限截断）"""
```

### Permutation Truncation Strategy

- `max_permutation_length`：排列最大长度（如设为 5，则不生成 6+ 规则的排列）
- `max_variants_per_block`：每个语句块最大变体数量
- 排列按长度从短到长生成，优先保留短排列

## Processing Flow

```
加载 mbpp_blocks.json
        ↓
遍历每个 sample 的每个 block
        ↓
对 block.source 用 tree-sitter 解析
        ↓
调用所有 Rule.detect() → 收集可行规则
        ↓
生成排列（受截断限制）
        ↓
对每个排列，依次 apply 规则 → 得到变体代码
  （每步重新解析 AST，若规则不可行则跳过该排列）
        ↓
记录: 变体代码 + 使用规则及顺序 + 所属 task_id
        ↓
汇总保存到 mbpp_blocks_transformed.json
```

**关键细节**：排列中依次应用规则时，每步都重新解析变换后的代码再做下一步变换。前一步变换可能改变 AST 结构，后续规则需要在新的 AST 上检测和应用。如果某步规则在变换后的代码上已不可行，则跳过该排列。

## Output JSON Structure

```json
{
  "metadata": {
    "source_file": "mbpp_blocks.json",
    "total_blocks": 12345,
    "transformed_blocks": 8000,
    "total_variants": 50000,
    "max_permutation_length": 5,
    "max_variants_per_block": 1000,
    "timestamp": "2026-03-07T..."
  },
  "samples": [
    {
      "task_id": 601,
      "blocks": [
        {
          "block_id": 0,
          "original_source": "print(x)",
          "applicable_rules": ["explicit_default_print", "operand_swap"],
          "variants": [
            {
              "variant_id": 0,
              "rules_applied": ["explicit_default_print"],
              "transformed_source": "print(x, flush=False)"
            },
            {
              "variant_id": 1,
              "rules_applied": ["explicit_default_print", "operand_swap"],
              "transformed_source": "..."
            }
          ]
        }
      ]
    }
  ]
}
```

## 39 Rules Implementation Classification

### Simple (~20 rules) — AST node match + parameter insertion/text replace

- API 显式默认参数系列：print, range, open, sorted, min/max, zip, random.seed, html.escape, round, json.dump (10 条)
- 库别名替换 LNA (1 条)
- 列表/字典初始化互换 (2 条)
- 格式化系列：缩进/空格/注释 (3 条)
- 一元操作简化 (1 条)
- 操作数交换 D1 (1 条)
- 比较运算符翻转 D2 (1 条)

### Medium (~10 rules) — Context-aware / multi-node operations

- 字符串格式化互换（% ↔ format）
- 类型检查互换（isinstance ↔ type）
- 推导式 ↔ map/lambda
- 分支翻转 D3
- 布尔逻辑等价 De Morgan
- 算术结合律
- 第三方函数替换 TPF

### Complex (~5 rules) — Deep AST restructuring

- for ↔ while 循环转换
- 直接迭代 ↔ 索引迭代转换
- 变量重命名（蛇形→驼峰）
- 名称混淆/语义等价替换

## Transformation Granularity

简单语句块和复合语句块都进行变换。

## Validation Strategy

- 每条规则 2-3 个单元测试用例（detect + apply）
- 先实现简单规则跑通整个流程，再逐步添加中等和复杂规则
- 用 MBPP 前 10 个样本做集成测试
- 对变换后的代码用 `compile()` 检查语法正确性
- 语义正确性通过规则设计保证（每条规则都是已知的等价变换）
