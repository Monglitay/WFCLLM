# LSH 水印改造设计文档

**日期：** 2026-03-10
**分支：** develop-LSH（从 develop 创建）
**参考文档：** `docs/基于LSH的改进方案.md`

---

## 背景与动机

原方案通过单一方向向量 v 和目标位 t 做投影水印，要求语义编码器在紧凑语义簇中生成符号相反的变体，存在"坍缩悖论"。

LSH 改造通过引入局部敏感哈希，将高维语义空间静态划分为 2^d 个宽广区域，LLM 只需在庞大候选空间中自然采样出落在指定有效区的代码块，从根本上解决坍缩问题。

---

## 设计决策

| 决策点 | 选择 | 理由 |
|---|---|---|
| 有效区种子上下文 | 只用 parent_node_type | 等价变换安全，不含当前节点特征 |
| lsh_d 参数 | 可配置，默认 3 | 灵活适配不同强度需求 |
| 全局超平面管理 | 独立 LSHSpace 类 | 职责清晰，embedding 和 extraction 侧共享 |
| 实施策略 | 原地替换（方案一） | 改动集中，符合 YAGNI 原则 |

---

## 架构概览

```
watermark/
├── lsh_space.py      # 新增：LSHSpace，全局超平面管理
├── keying.py         # 改造：derive() 返回有效区集合 G
├── verifier.py       # 改造：verify() 基于 LSH 签名 + 裕度检验
├── config.py         # 改造：新增 lsh_d, lsh_gamma 字段
└── generator.py      # 改造：初始化 LSHSpace，更新调用链

extract/
├── scorer.py         # 改造：传 valid_set 给 verifier
└── hypothesis.py     # 改造：gamma 参数化 Z-score 公式
```

---

## 模块详细设计

### 1. `watermark/lsh_space.py`（新建）

```python
class LSHSpace:
    def __init__(self, secret_key: str, embed_dim: int, d: int):
        # 用 HMAC-SHA256(secret_key, "lsh") 作为种子
        # 生成 d 个伪随机正态分布向量并 L2 归一化
        # shape: (d, embed_dim)，存为 self._planes

    def sign(self, u: Tensor) -> tuple[int, ...]:
        # 计算 u 与每个超平面的余弦投影符号
        # LSH_i(u) = 1 if cos(u, n[i]) > 0 else 0
        # 返回 d 维二进制签名 tuple，如 (1, 0, 1)

    def min_margin(self, u: Tensor) -> float:
        # 返回 min_i |cos(u, n[i])|
        # 用于裕度检验（距离所有超平面的最小绝对余弦距离）
```

### 2. `watermark/keying.py`

**接口变化：**
- `__init__` 新增 `d: int`, `gamma: float` 参数，移除 `embed_dim`
- `derive(parent_node_type: str) -> frozenset[tuple[int, ...]]`
  - 种子：`HMAC-SHA256(secret_key, parent_node_type)`
  - 从 `2^d` 个签名中随机选 `round(gamma * 2^d)` 个构成有效区集合 G
  - 返回 `frozenset`

### 3. `watermark/verifier.py`

**接口变化：**
- `__init__` 新增 `lsh_space: LSHSpace` 参数
- `verify(code_text: str, valid_set: frozenset[tuple[int,...]], margin: float) -> VerifyResult`
  - 编码 → u
  - `sign = lsh_space.sign(u)`
  - 落点检验：`sign ∈ valid_set`
  - 裕度检验：`lsh_space.min_margin(u) > margin`
  - `passed = 落点检验 AND 裕度检验`

**VerifyResult 字段变化：**
- `projection: float` → `min_margin: float`
- `target_sign: int` → 移除

### 4. `watermark/config.py`

新增字段：
```python
lsh_d: int = 3          # 超平面数量
lsh_gamma: float = 0.5  # 有效区比例
```

### 5. `watermark/generator.py`

初始化变化：
```python
self._lsh_space = LSHSpace(config.secret_key, config.encoder_embed_dim, config.lsh_d)
self._keying = WatermarkKeying(config.secret_key, config.lsh_d, config.lsh_gamma)
self._verifier = ProjectionVerifier(encoder, encoder_tokenizer, self._lsh_space, ...)
```

调用链变化：
- `keying.derive(parent, node_type)` → `keying.derive(parent_node_type)`
- `verifier.verify(text, v, t, margin)` → `verifier.verify(text, valid_set, margin)`

### 6. `extract/scorer.py`

- `score_block()` 中 `keying.derive()` 调用同步更新（只传 parent_node_type）
- `verifier.verify()` 调用同步更新（传 valid_set）
- 评分逻辑不变：命中 = +1，未命中 = -1

### 7. `extract/hypothesis.py`

Z-score 公式参数化：
```python
# 旧：
z_score = (x - m / 2) / math.sqrt(m / 4)

# 新：
z_score = (x - m * gamma) / math.sqrt(m * gamma * (1 - gamma))
```

`HypothesisTester.__init__` 新增 `gamma: float = 0.5` 参数。

---

## 数据流

```
生成阶段：
  parent_node_type
    → keying.derive() → valid_set G
    → verifier.verify(block_text, G, margin)
        → encoder(block_text) → u
        → lsh_space.sign(u) → c
        → c ∈ G? AND min_margin(u) > margin?
        → VerifyResult.passed

提取阶段：
  parent_node_type
    → keying.derive() → valid_set G（与生成时相同）
    → verifier.verify(block_text, G, 0.0)  # 提取时 margin=0
        → lsh_space.sign(u) → c
        → score = +1 if c ∈ G else -1
  → dp_selector → hypothesis_tester(gamma=0.5) → Z-score
```

---

## 测试策略

每个改造模块对应 `tests/watermark/` 和 `tests/extract/` 下的测试文件：

- `test_lsh_space.py`：超平面确定性、sign 正确性、min_margin 计算
- `test_keying.py`：相同 parent_node_type 返回相同 G，G 大小符合 gamma
- `test_verifier.py`：passed 逻辑、VerifyResult 字段
- `test_hypothesis.py`：gamma 参数化 Z-score 公式
- `test_scorer.py`：更新后接口调用正确

---

## 评测指标变化

原方案的"水印符号一致性"改为 **LSH 一致性（LSH Consistency）**：
等价变换后的代码块中，与原块具有相同 LSH 签名的比例。
