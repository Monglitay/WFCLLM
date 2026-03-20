# 基于动态熵值的自适应水印机制设计

**日期：** 2026-03-19  
**作者：** Codex  
**状态：** 设计已确认，待实现

## 1. 概述

本设计旨在为 `WFCLLM` 引入 **Entropy-Aware Watermarking**：不再对所有语句块使用全局固定的 `gamma`，而是根据每个语句块的动态熵值，为其分配自适应的块级有效区域比例 `gamma_i`，并在提取端使用泊松二项分布的正态近似进行检测。

该设计重点解决三个现有问题：

1. 现有系统中，熵值仅用于动态 `margin`，`lsh_gamma` 仍为全局固定值，无法根据块复杂度调整嵌入强度。
2. 提取端当前假设所有样本共享同一命中概率，统计模型与块级自适应嵌入不匹配。
3. 嵌入端与提取端虽然已有对齐实验，但“块切分一致、熵值一致、参数一致”尚未被提升为正式协议约束。

本 RFC 的核心是建立一套 **可复现、可校验、可版本化** 的块级契约协议，使嵌入端与提取端对同一份最终代码能够重建完全一致的 `Statement Block Contract`。

### 1.1 目标

- 基于历史日志构建版本化 `entropy_profile`
- 在嵌入端为每个 simple statement block 计算 `gamma_i`
- 采用 `piecewise_quantile` 作为默认映射策略，同时保留 `fixed` / `linear` / `bucket` 作为可切换基线
- 在提取端使用块级 `gamma_i` 重构统计检验
- 显式保障并测试“嵌入端与提取端块切分完全一致，熵值绝对相等”
- 更新 `run.py`、配置文件、输出 metadata 与 README
- 制定分阶段实施与 Git 提交规划

### 1.2 非目标

- V1 不重构现有 `margin = margin_base + alpha * entropy` 逻辑，仅为未来扩展预留接口
- V1 不实现精确泊松二项分布 PMF/CDF，仅实现正态近似
- V1 不引入跨语言共享 profile；profile 的粒度限定为 `language + model_family`
- V1 不要求旧版 watermark 输出重写，只要求提取端向后兼容

## 2. 已确认的关键决策

### 2.1 产品与协议决策

- **熵值基准统计来源：** 采用混合模式。默认读取版本化 `entropy_profile`，必要时允许用新日志重新校准生成 profile。
- **Profile 作用域：** 按 `language + model_family` 维度维护，不按具体数据集拆分。
- **低熵区上界策略：** 对 `gamma_i` 设置上界裁剪，推荐 `0.95` 或 `0.99`；V1 默认用 `0.95`。
- **动态参数范围：** V1 仅让动态熵驱动 `gamma_i`，`margin` 保持现状。
- **兼容策略：** 提取端优先识别 adaptive metadata；缺失时回退旧版固定 `gamma`。
- **真值来源：** 提取端以最终代码重算为主，同时比对嵌入端 metadata。
- **异常处理：**
  - 结构不一致：硬失败
  - 数值不一致：输出诊断并标记样本契约无效，不纳入可信 adaptive 汇总

### 2.2 映射策略决策

在对三类方案进行比较后，选择 **方案三：分位数锚点的分段线性映射** 作为默认实现：

1. **原始熵线性映射**
   - 优点：实现简单
   - 缺点：对长尾分布敏感，容易被极端值主导
2. **分位数阶梯映射**
   - 优点：稳健、易解释
   - 缺点：不连续，边界附近容易跳变
3. **分位数锚点的分段线性映射**
   - 优点：兼具稳健性、平滑性和可解释性
   - 结论：**作为默认方案**

基于样本日志 `logs/0318_204544_watermark.log` 的快速统计，当前分布呈现明显长尾：

- `count ≈ 10648`
- `mean ≈ 1.8058`
- `p50 ≈ 1.3889`
- `p75 ≈ 2.4725`
- `p90 ≈ 3.0382`
- `p95 ≈ 4.2893`
- `max ≈ 23.3690`

这进一步证明直接对原始熵做线性拟合并不稳妥。

## 3. 当前系统缺口

### 3.1 嵌入端现状

当前 `wfcllm/watermark/entropy.py` 中，熵值只用于计算：

```python
margin = margin_base + margin_alpha * block_entropy
```

当前 `wfcllm/watermark/keying.py` 仍使用全局固定 `lsh_gamma` 决定有效区域大小：

```python
k = round(self._gamma * len(all_sigs))
```

因此系统虽然已经引入熵值，但尚未真正实现“按块自适应水印强度”。

### 3.2 提取端现状

当前 `wfcllm/extract/hypothesis.py` 使用固定参数二项分布：

```python
z = (x - m * gamma) / sqrt(m * gamma * (1 - gamma))
```

该公式默认所有块共享相同成功概率，无法匹配块级 `gamma_i`。

### 3.3 对齐与诊断现状

仓库已存在 `experiment/embed_extract_alignment`，用于诊断嵌入端事件和提取端 `StatementBlock` 的对齐情况。但该逻辑仍停留在实验层：

- 未形成主流程契约
- 未固化为正式输出字段
- 未建立结构不一致与数值不一致的分级故障语义

## 4. 总体架构

本设计在现有三阶段流水线上新增一个 **块级协议层**，命名为：

**Entropy-Aware Adaptive Gamma Protocol**

### 4.1 架构分层

1. **Entropy Profile Layer**
   - 负责离线校准与版本化 profile 管理
2. **Gamma Schedule Layer**
   - 负责将 `entropy_i` 映射为 `gamma_i`
3. **Block Contract Layer**
   - 负责构建块级 canonical metadata
4. **Alignment Guard Layer**
   - 负责嵌入端 metadata 与提取端重算结果的双轨校验
5. **Adaptive Test Layer**
   - 负责使用块级 `gamma_i` 进行统计检验

### 4.2 运行时数据流

#### 嵌入端

1. 生成 simple statement block
2. 计算该块的 `entropy_i`
3. 根据 `entropy_profile + gamma_schedule` 得到 `gamma_target_i`
4. 将 `gamma_target_i` 量化为离散有效区域大小 `k_i`
5. 派生 `gamma_effective_i = k_i / 2^d`
6. 用 `G_i + margin_i` 完成验证、重试与级联回退
7. 生成结束后，对最终代码重新构建 canonical block contracts
8. 将 block-level metadata 写入 watermark 输出

#### 提取端

1. 读取最终代码
2. 重新切分 simple statement blocks
3. 重算 `entropy_i / gamma_target_i / k_i / gamma_effective_i`
4. 与嵌入端 metadata 做结构与数值校验
5. 若通过结构校验，则使用 `gamma_effective_i` 进入自适应 Z 检验

### 4.3 设计原则

- 嵌入端与提取端共享同一套 contract builder
- canonical truth 采用整数或离散值，避免浮点漂移
- profile、schedule、statistics 解耦，便于实验与回归
- 对齐失败必须显式暴露，不允许静默吞掉 drift

## 5. Entropy Profile 设计

### 5.1 设计目标

`entropy_profile` 是 adaptive gamma 的离线校准产物，其职责是：

- 保存给定 `language + model_family` 下的熵值基线
- 为 `piecewise_quantile` 提供稳定锚点
- 作为嵌入端与提取端共享的“同一把尺子”
- 支持版本化、审计、回滚与对比实验

### 5.2 文件粒度与命名

建议命名：

```text
configs/entropy_profiles/python__deepseek-coder-7b-base__v2026-03-19.json
```

字段粒度按 `language + model_family` 组织，不按数据集单独拆分。

### 5.3 建议的数据结构

以下示例为**结构示意**，其中 `*_units` 由校准脚本实际填充：

```json
{
  "profile_id": "python__deepseek-coder-7b-base__v2026-03-19",
  "version": 1,
  "language": "python",
  "model_family": "deepseek-coder-7b-base",
  "sources": [
    "logs/0318_204544_watermark.log"
  ],
  "sample_count": 10648,
  "entropy_scale": 10000,
  "summary": {
    "min_units": 3571,
    "max_units": 233690,
    "mean_units": 18058,
    "median_units": 13889
  },
  "quantiles": {
    "p05_units": "<int>",
    "p10_units": "<int>",
    "p25_units": "<int>",
    "p50_units": 13889,
    "p75_units": 24725,
    "p90_units": 30382,
    "p95_units": 42893
  }
}
```

说明：

- profile 中存储整数单位值 `*_units`
- `entropy_scale` 明确记录缩放因子
- `sources` 用于审计来源

### 5.4 校准流程

建议扩展 `scripts/calibrate.py`，支持生成或更新 profile：

1. 读取一个或多个 watermark 日志
2. 解析其中 simple block 的 `entropy=...`
3. 统一转换为 `entropy_units`
4. 计算汇总统计与分位点
5. 输出版本化 profile 文件

### 5.5 Profile 更新策略

- 新 profile 不覆盖旧 profile
- 所有输出样本记录 `profile_id`
- summary 报告区分不同 profile 的样本来源

## 6. Gamma Schedule 设计

### 6.1 统一接口

为后续实验与回归，设计统一的 schedule 抽象，支持：

- `fixed`
- `linear`
- `bucket`
- `piecewise_quantile`

其中 V1 默认使用 `piecewise_quantile`。

### 6.2 默认锚点

建议默认锚点为：

- `p10 -> 0.95`
- `p50 -> 0.75`
- `p75 -> 0.55`
- `p90 -> 0.35`
- `p95 -> 0.25`

含义：

- 低熵块优先保护语义
- 中位数附近保留适中嵌入强度
- 高熵区逐步增强水印强度
- 极高熵区落到最强嵌入区间

### 6.3 分段线性映射

设锚点熵值为 `e_0 < e_1 < ... < e_n`，对应 `gamma_0 > gamma_1 > ... > gamma_n`。

对于区间内某个块熵 `e`：

```text
gamma_target(e) = gamma_j +
    (gamma_{j+1} - gamma_j) * (e - e_j) / (e_{j+1} - e_j)
```

边界规则：

- `e <= e_0`：取 `gamma_max`
- `e >= e_n`：取 `gamma_min`
- 再统一 clip 到 `[gamma_min, gamma_max]`

### 6.4 离散量化

因为实际有效区域大小必须落在离散签名集合上，建议引入：

```text
k_i = clip(round(gamma_target_i * 2^d), 1, 2^d - 1)
gamma_effective_i = k_i / 2^d
```

统计检验与运行时有效区域生成都应使用 `gamma_effective_i`，而不是原始 `gamma_target_i`。

### 6.5 设计理由

使用 `k_i` 作为 canonical truth 有三点好处：

1. 与 `2^d` 个 LSH 区域严格对齐
2. 避免浮点比较歧义
3. 提取端统计假设与运行时真实区域大小一致

## 7. Canonical Block Contract 设计

### 7.1 设计目标

块级 contract 是嵌入端与提取端共享的正式协议。其作用是：

- 为每个 simple block 定义稳定身份
- 存储熵值与 schedule 映射结果
- 提供结构和数值的一致性校验基础

### 7.2 Canonical 真值策略

为满足“熵值绝对相等”的要求，建议：

- 熵值真值采用 `entropy_units` 整数
- `gamma_i` 真值采用 `k_i`
- `gamma_target` 仅作为调试显示字段
- `gamma_effective_i` 由 `k_i` 派生

### 7.3 熵值整数化

建议新增：

```python
ENTROPY_SCALE = 10000
```

熵表中的条目改为整数单位值，例如：

- `0.9590 -> 9590`
- `0.1539 -> 1539`

块熵不再用浮点求和，而是：

```text
entropy_units = Σ node_entropy_units
```

只在日志打印时派生展示值：

```text
entropy = entropy_units / ENTROPY_SCALE
```

### 7.4 建议的 BlockContract 字段

```python
@dataclass
class BlockContract:
    ordinal: int
    block_id: str
    node_type: str
    parent_node_type: str
    block_text_hash: str
    start_line: int
    end_line: int
    entropy_units: int
    gamma_target: float
    k: int
    gamma_effective: float
```

### 7.5 块身份定义

每个 simple block 的 canonical identity 至少由以下信息组成：

- `ordinal`
- `node_type`
- `parent_node_type`
- `block_text_hash`

其中：

- `ordinal + block_text_hash + parent_node_type` 构成主身份
- `start_line / end_line` 用于辅助诊断
- 不使用运行时 AST 临时 ID 作为跨阶段主键

### 7.6 共享构建器

建议新增 `wfcllm/common/block_contract.py`，提供：

```python
build_block_contracts(
    code: str,
    profile: EntropyProfile,
    schedule: GammaSchedule,
    lsh_d: int,
) -> list[BlockContract]
```

嵌入端与提取端都必须调用该共享函数，禁止各自拼装独立逻辑。

## 8. 嵌入端改造设计

### 8.1 运行时控制层

当 interceptor 产出 simple block 时，执行：

1. 计算 `entropy_i`
2. 通过 schedule 计算 `gamma_target_i`
3. 量化得到 `k_i`
4. 派生 `gamma_effective_i`
5. 调用新的 `WatermarkKeying.derive(parent_node_type, k_i)`
6. 用 `G_i + margin_i` 完成验证

### 8.2 WatermarkKeying 改造

现有构造函数依赖全局 `gamma`：

```python
WatermarkKeying(secret_key, d, gamma)
```

建议改为：

```python
WatermarkKeying(secret_key, d)
derive(parent_node_type: str, k: int) -> frozenset[tuple[int, ...]]
```

有效区域大小由调用者按块传入。

### 8.3 最终契约层

样本生成结束后，不直接把运行时事件当作检测真值，而是：

1. 对最终 `generated_code` 重新调用 `build_block_contracts`
2. 得到最终 authoritative contracts
3. 将其写入输出记录
4. 同时对运行时事件与最终块做对齐，形成调试报告

### 8.4 Watermark 输出扩展

每条记录建议新增：

```json
{
  "adaptive_mode": "piecewise_quantile",
  "profile_id": "python__deepseek-coder-7b-base__v2026-03-19",
  "schedule_version": 1,
  "lsh_d": 4,
  "blocks": [
    {
      "ordinal": 0,
      "node_type": "return_statement",
      "parent_node_type": "if_statement",
      "block_text_hash": "...",
      "entropy_units": 10019,
      "gamma_target": 0.73,
      "k": 11,
      "gamma_effective": 0.6875
    }
  ],
  "alignment_summary": {
    "matched": 12,
    "mismatched": 0,
    "failed_reason": null
  }
}
```

### 8.5 嵌入端失败策略

- 若最终 contract 构建失败，样本标记为 `alignment_failed`
- 若运行时事件与最终块无法建立完整对齐，仍可保留生成代码，但该样本不得视为可信 adaptive 样本

## 9. 提取端改造设计

### 9.1 模式判定

提取端支持两种模式：

- `fixed`
- `adaptive`

默认策略为：

1. 若样本包含 adaptive metadata，则优先走 adaptive
2. 若 metadata 缺失，则回退 fixed

### 9.2 Contract 重算与比对

在 adaptive 模式下，提取端执行：

1. 对最终代码调用 `build_block_contracts`
2. 与嵌入端 `blocks` metadata 做逐块比对
3. 生成 `AlignmentReport`

### 9.3 AlignmentReport 建议字段

```python
@dataclass
class AlignmentReport:
    structure_match: bool
    numeric_match: bool
    mismatch_count: int
    first_mismatch_reason: str | None
    first_mismatch_block_ordinal: int | None
    embed_only_blocks: list[int]
    extract_only_blocks: list[int]
```

### 9.4 异常语义

#### 硬失败条件

- simple block 数量不一致
- block 顺序不一致
- `block_text_hash` 不一致
- `node_type / parent_node_type` 不一致

#### 高优先级数值异常

- `entropy_units` 不一致
- `k_i` 不一致
- `gamma_effective_i` 由 `k_i` 推导后不一致

处理策略：

- 结构不一致：立即返回 `alignment_failed`
- 数值不一致：输出诊断结果，并将样本标记为 `adaptive_contract_invalid`
- `adaptive_contract_invalid` 默认不纳入可信 adaptive 汇总

### 9.5 命中定义

对每个 simple block `i`：

- 若其语义向量落入 `G_i`，则 `hit_i = 1`
- 否则 `hit_i = 0`

其中 `G_i` 由 `k_i` 决定，而不是全局固定 `gamma`。

## 10. 自适应统计检验设计

### 10.1 固定模式

固定模式保持现有单参数二项检验逻辑，用于旧样本兼容与回归对比。

### 10.2 Adaptive 模式

自适应模式中，设：

- `X = Σ hit_i`
- `mu = Σ gamma_i`
- `sigma2 = Σ gamma_i * (1 - gamma_i)`

其中 `gamma_i = gamma_effective_i`

则正态近似 Z 分数为：

```text
Z = (X - mu) / sqrt(sigma2)
```

### 10.3 选择正态近似而非精确泊松二项的原因

- 工程复杂度明显更低
- 当前核心风险在契约一致性，而非分布计算精度
- 在块数充足时近似已足够支持实验比较

### 10.4 数值退化处理

实现中建议：

- 若 `sigma2 < 1e-12`，返回 `degenerate_test_result`
- 检测结果显式记录 `variance`
- 避免将退化结果误当作高置信度水印

### 10.5 DetectionResult 扩展

建议新增：

- `mode`
- `alignment_ok`
- `contract_valid`
- `alignment_failed_reason`
- `expected_hits`
- `variance`
- `profile_id`
- `schedule_version`
- `gamma_summary`

## 11. 配置、CLI 与输出格式设计

### 11.1 模块边界

建议新增或重构为以下模块：

- `wfcllm/watermark/entropy_profile.py`
- `wfcllm/watermark/gamma_schedule.py`
- `wfcllm/common/block_contract.py`
- `wfcllm/extract/alignment.py`
- `wfcllm/extract/hypothesis.py`（扩展支持 fixed / adaptive）
- `scripts/calibrate.py`（扩展支持 profile 校准）

### 11.2 配置结构

建议在 `watermark` 下新增：

```json
{
  "adaptive_gamma": {
    "enabled": true,
    "strategy": "piecewise_quantile",
    "profile_path": "configs/entropy_profiles/python__deepseek-coder-7b-base__v2026-03-19.json",
    "profile_id": "python__deepseek-coder-7b-base__v2026-03-19",
    "gamma_min": 0.25,
    "gamma_max": 0.95,
    "anchor_quantiles": ["p10", "p50", "p75", "p90", "p95"],
    "anchor_gammas": [0.95, 0.75, 0.55, 0.35, 0.25],
    "quantize_to_effective_gamma": true
  }
}
```

建议在 `extract` 下新增：

```json
{
  "adaptive_detection": {
    "mode": "prefer-adaptive",
    "require_contract_match": true,
    "fail_on_structure_mismatch": true,
    "warn_on_numeric_mismatch": true,
    "exclude_invalid_samples_from_summary": true
  }
}
```

### 11.3 CLI 扩展

建议在 `run.py` 中新增参数：

- `--gamma-strategy`
- `--entropy-profile`
- `--profile-id`
- `--adaptive-detection-mode`
- `--strict-contract`

### 11.4 校准脚本扩展

建议扩展 `scripts/calibrate.py` 支持：

- `build-entropy-profile`
- `calibrate-threshold`

### 11.5 README 更新重点

README 需新增：

- adaptive gamma 原理简介
- 如何构建 `entropy_profile`
- watermark / extract 新参数示例
- contract mismatch 的排障说明

## 12. 测试与质量保障

### 12.1 质量门槛

在宣布该功能完成前，至少满足：

- adaptive 模式可端到端跑通
- fixed 模式回归不破
- contract round-trip 测试稳定通过
- 至少一个真实 watermark 日志成功生成 profile
- mismatch 注入测试能正确触发异常或告警

### 12.2 测试分层

#### 1）单元测试：熵计算

- 相同代码多次构建 contract，`entropy_units` 完全一致
- 已知 snippet 断言精确 `entropy_units`

#### 2）单元测试：schedule 映射

- 锚点命中测试
- 区间内插值测试
- 上下界 clip 测试
- `gamma_target -> k_i -> gamma_effective_i` 量化边界测试

#### 3）契约测试：嵌入/提取 round-trip

- 相同最终代码分别走 embed finalize 与 extract rebuild
- 断言 block 顺序、`block_text_hash`、`entropy_units`、`k_i` 全等

#### 4）故障注入测试

- 篡改单块 `entropy_units`
- 篡改单块 `k_i`
- 删除一个 block
- 修改 parent type
- 断言系统输出 `alignment_failed` 或 `adaptive_contract_invalid`

#### 5）集成回归测试

- watermark -> extract 全链路验证 adaptive 契约一致
- 保留 fixed 模式回归
- 校验 summary 中 fixed / adaptive 不混算

### 12.3 核心难点的正式解决方案

为严格保障“嵌入端与提取端切分出的语句块完全一致，且计算出的熵值绝对相等”，本设计要求：

1. 引入共享 `block_contract` 构建器
2. 使用整数化 `entropy_units` 作为 canonical truth
3. 使用离散 `k_i` 作为块级 gamma 真值

这三项是整个 adaptive watermark 成败的关键地基。

## 13. 风险与缓解

### 13.1 高风险点

1. **Canonical contract 引入**
   - 风险：块提取边界与现有流程存在暗含差异
   - 缓解：优先落地 contract round-trip 测试
2. **运行时按块 `k_i` 改造 keying**
   - 风险：嵌入端行为变化影响旧测试
   - 缓解：保留 fixed 模式并做并行回归
3. **提取端 alignment 正式化**
   - 风险：历史样本中 latent drift 被暴露
   - 缓解：区分 fixed / adaptive 汇总，避免污染旧实验结论

### 13.2 中风险点

- profile 校准数据不足导致锚点偏移
- 小 `d` 场景下离散量化导致 `gamma_target` 与 `gamma_effective` 偏差较大

缓解方式：

- 要求 profile 记录样本数与来源
- 在报告中同时输出 `gamma_target` 与 `gamma_effective`
- 对 `lsh_d` 较小的实验显式记录量化偏差

## 14. 分步实施计划与 Git 提交规划

建议拆为 7 个可回滚提交：

### 提交 1：引入共享契约层

- 新增 canonical `BlockContract`
- 统一 simple block 提取、hash、parent type
- 引入 `entropy_units`

建议提交信息：

```text
feat: add canonical block contract builder
```

### 提交 2：引入 profile 与 gamma schedule

- 新增 `EntropyProfile`
- 新增 `GammaSchedule`
- 实现 `piecewise_quantile`
- 实现 `k_i` 量化

建议提交信息：

```text
feat: add entropy profile and adaptive gamma schedule
```

### 提交 3：嵌入端接入 adaptive gamma

- WatermarkKeying 支持按块 `k_i`
- WatermarkGenerator / RetryLoop 接入 adaptive gamma
- 输出 canonical block metadata

建议提交信息：

```text
feat: wire adaptive gamma into watermark generator
```

### 提交 4：提取端接入 contract 校验

- 新增 alignment 模块
- 执行 metadata 与重算结果双轨比对
- 实现结构/数值异常分级

建议提交信息：

```text
feat: add adaptive contract alignment checks
```

### 提交 5：重构 HypothesisTester

- 支持 fixed / adaptive 两种统计模式
- 输出 `expected_hits` 与 `variance`

建议提交信息：

```text
feat: support adaptive hypothesis testing
```

### 提交 6：CLI、配置与文档

- 更新 `run.py`
- 更新 `configs/base_config.json`
- 扩展 `scripts/calibrate.py`
- 更新 `README.md`

建议提交信息：

```text
feat: expose adaptive gamma via cli and configs
```

### 提交 7：端到端回归与诊断增强

- 新增 round-trip 测试
- 新增 mismatch 注入测试
- 固化 adaptive 诊断输出

建议提交信息：

```text
test: add adaptive watermark end-to-end coverage
```

## 15. 实施顺序说明

推荐顺序如下：

1. 先落地 canonical contract
2. 再落地 profile 与 schedule
3. 再接入嵌入端运行时 `k_i`
4. 然后正式化提取端 alignment
5. 最后替换统计检验、CLI 与文档

原因：

- 若无 canonical contract，则后续所有 adaptive 统计都没有可信输入
- 若无 profile 与 schedule，则嵌入端无法稳定生成 `k_i`
- 若无 alignment，则统计检验无法区分真实信号与上下文漂移

## 16. 验收标准

满足以下条件时，可认为该设计实现完成：

- `build-entropy-profile` 能从真实日志生成版本化 profile
- 嵌入端输出包含 canonical block metadata
- 提取端能自动识别 adaptive / fixed 样本
- adaptive 检测公式基于 `gamma_effective_i` 正常工作
- 至少一组端到端测试证明嵌入端与提取端块切分一致、`entropy_units` 全等、`k_i` 全等
- mismatch 注入测试能稳定触发预期故障语义

---

本设计将动态熵水印从“局部启发式改动”提升为一套正式协议：用版本化 profile 管理统计基线，用共享 contract builder 保证嵌入/提取一致性，用离散 `k_i` 保证统计假设与运行时行为一致，并在提取端以 adaptive hypothesis testing 完成闭环。这使得后续无论替换映射曲线、调整 `lsh_d`，还是扩展到不同模型族，都能在一致的工程边界内演进。
