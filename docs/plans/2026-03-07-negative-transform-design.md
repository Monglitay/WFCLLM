# Negative Transform Design

## Context

在现有 `statement_block_transform` 实验基础上，增加负变换模式（negative mode），构造语义错误的负样本，与正变换的语义等价样本形成对比，用于 LLM 代码水印训练数据。

- 输入：`experiment/statement_block_split/results/mbpp_blocks.json`
- 负变换规则：`docs/python代码负变换规则.csv`（36 条规则，8 个类别）
- 实现位置：`experiment/statement_block_transform/`（方案 C，在现有目录内扩展）
- 运行方式：通过 `--mode` 参数区分正/负变换

## 目的

通过语义破坏变换生成负样本，每个 variant 用 `sample_type` 字段标记 `"positive"` 或 `"negative"`，两种样本格式完全统一，方便后续联合训练。

## Directory Structure Changes

```
experiment/statement_block_transform/
├── config.py               # 新增 OUTPUT_FILE_NEGATIVE 路径
├── engine.py               # 新增 mode 参数，variant 加 sample_type 字段
├── transform.py            # 新增 --mode CLI 参数，根据 mode 选择规则集和输出路径
└── rules/
    ├── __init__.py         # 已有，不改
    ├── base.py             # 已有，不改（负规则复用同一基类）
    ├── ... (已有规则文件，不改)
    └── negative/           # 新建
        ├── __init__.py     # get_all_negative_rules()
        ├── api_calls.py    # 7 条：Min/Max翻转、Any/All替换、Sorted方向、Open模式、Extend/Append、StartsWith/EndsWith、Ceil/Floor
        ├── syntax_init.py  # 4 条：空容器污染、布尔翻转、Isinstance类型替换、Copy vs Ref
        ├── control_flow.py # 5 条：Off-by-one、Break/Continue互换、If-Else对调、Membership取反、Yield/Return替换
        ├── expression_logic.py # 6 条：Eq/Neq反转、算术运算符替换、And/Or替换、Bounds收缩、赋值降级、移位反转
        ├── identifier.py   # 2 条：非对称参数交换、作用域变量混淆
        ├── data_structure.py # 2 条：切片方向反转、Dict视图错位
        ├── exception.py    # 1 条：异常吞噬
        └── system.py       # 1 条：退出状态码反转
```

## Core Changes

### engine.py

```python
class TransformEngine:
    def __init__(self, rules, max_perm_len=5, max_variants=1000, mode="positive"):
        self.mode = mode  # "positive" or "negative"
        ...

    def generate_variants(self, source):
        variants.append({
            "variant_id": ...,
            "rules_applied": ...,
            "transformed_source": ...,
            "sample_type": self.mode,  # 新增字段
        })
```

### config.py

```python
OUTPUT_FILE_NEGATIVE = RESULTS_DIR / "mbpp_blocks_negative_transformed.json"
```

### transform.py

```python
ap.add_argument("--mode", choices=["positive", "negative"], default="positive")

# 根据 mode 选择规则集和输出路径
if mode == "negative":
    rules = get_all_negative_rules()
    output = OUTPUT_FILE_NEGATIVE
else:
    rules = get_all_rules()
    output = OUTPUT_FILE

engine = TransformEngine(rules=rules, ..., mode=mode)
```

## Permutation Strategy

负变换与正变换采用相同的全排列组合策略：对所有可行规则生成从长度 1 到 max_perm_len 的全排列，受 max_variants 截断。每步重新解析 AST，若规则不可行则跳过该排列。

## Output

正变换输出：`results/mbpp_blocks_transformed.json`
负变换输出：`results/mbpp_blocks_negative_transformed.json`

Variant 格式（两者相同，只有 sample_type 不同）：
```json
{
  "variant_id": 0,
  "rules_applied": ["min_max_flip"],
  "transformed_source": "res = max(x)",
  "sample_type": "negative"
}
```

## 36 Negative Rules by Category

| 类别 | 条数 | 文件 |
|------|------|------|
| API与函数调用 | 7 | api_calls.py |
| 语法与初始化 | 4 | syntax_init.py |
| 控制流 | 5 | control_flow.py |
| 表达式与逻辑 | 6 | expression_logic.py |
| 标识符操作 | 2 | identifier.py |
| 数据结构操作 | 2 | data_structure.py |
| 异常处理 | 1 | exception.py |
| 系统交互 | 1 | system.py |
| **合计** | **28** | |

注：CSV 中共 36 条，其中部分规则（如 scope 混淆、非对称参数交换）实现复杂度较高，实现分类如下：
- Simple：Min/Max翻转、Any/All、Sorted方向、布尔翻转、Eq/Neq、And/Or、Bounds收缩、赋值降级、移位反转、切片反转、Dict视图、退出码、异常吞噬、Break/Continue、Membership
- Medium：Off-by-one、If-Else对调、Extend/Append、StartsWith/EndsWith、Ceil/Floor、Copy vs Ref、Open模式、算术运算符替换
- Complex：Isinstance类型替换、空容器污染、非对称参数交换、作用域变量混淆、Yield/Return

## Validation

- 每条规则 2-3 个单元测试（detect + apply）
- 先实现 Simple 规则跑通整个流程，再加 Medium 和 Complex
- MBPP 前 10 个样本做集成测试
- 用 `compile()` 检查语法正确性（负变换改变语义但不破坏语法）
