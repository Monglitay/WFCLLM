# Anchor 有效性验证实验方案

**日期：** 2026-06-02  
**状态：** 机制验证计划，执行前草案  
**对象：** CASD-WFCLLM 中的 deterministic code anchor 与 Anchor-Orthogonal LSH  
**核心问题：** Anchor 是否真的缓解低熵代码生成中的 semantic region collapse，而不是只引入随机扰动或额外复杂度  

## 0. 结论先行

我不能确定 Anchor 一定有效。当前它只是一个由 SeqMark 启发、但更适合代码水印约束的机制假设。这个实验方案的目标不是证明 Anchor 必然成功，而是设计一组可以推翻它的实验：

> 如果 deterministic code anchor 不能显著提升候选 block 的 region entropy、不能把生成期嵌入成功转化为检测期 semantic z，或者随机 anchor 也能达到同样效果，那么 CASD 主线必须降级或更换方案。

验证重点有三个：

1. **机制层面：** Anchor 是否让同一代码上下文下的高质量候选从 LSH region collapse 中分散出来。
2. **检测层面：** Anchor 是否让 semantic hit/miss 更接近可校准的 Bernoulli 统计，而不是只改变签名但不增加有效证据。
3. **工程层面：** Anchor 是否在不引入 SeqMark 级在线多候选成本的情况下，达到接近 SeqMark oracle 的诊断收益。

## 1. 背景与怀疑点

### 1.1 当前 WFCLLM 的失败事实

cap9 结果显示，当前语义信道的问题不是完全没有嵌入，而是嵌入没有转化为检测证据：

| 指标 | 观察值 | 含义 |
|---|---:|---|
| mean embed_rate | 约 0.576 | 生成期确实有不少 block 被标记为嵌入成功 |
| semantic positive | 15/141 = 10.6% | 检测期语义阳性极低 |
| mean semantic z | 约 -0.119 | 平均语义统计量没有正向证据 |
| corr(embed_rate, semantic_z) | 约 0.048 | 嵌入成功率几乎不能解释检测分数 |
| joint positive | 119/141 = 84.4% | joint 主要由 lexical 信道拉高 |

这说明普通 block embedding + LSH 的统计结构可能不适合低熵代码生成。SeqMark 提出的 region collapse 正好解释了这个现象：同一上下文下的高质量候选语义相近，即使 rejection sampling 选中了目标 region，检测端也难以积累独立稳定的命中。

### 1.2 Anchor 的假设

CASD 不是直接照搬 SeqMark 的在线候选中心：

$$
\mu_c=\frac{1}{K}\sum_{j=1}^{K}E(\tilde b_j),
\quad
h=\operatorname{LSH}(E(b)-\mu_c)
$$

而是用代码中检测端可复现的 anchor：

$$
a_i=A(\text{signature}, \text{AST path}, \text{parent context}, \text{masked skeleton})
$$

再让 LSH 作用在相对实现差异上：

$$
z_i=P_iE(b_i), \quad P_i=I-\frac{a_i a_i^\top}{\lVert a_i\rVert_2^2+\epsilon}
$$

核心假设是：

> 代码 anchor 可以近似表示任务/结构共同语义方向；把该方向从 block embedding 中剥离后，LSH 会更多关注局部实现差异，从而缓解 region collapse。

这不是必然成立。它需要实验验证。

### 1.3 必须排除的替代解释

本方案必须排除以下反解释：

| 反解释 | 如果成立意味着什么 | 必要控制实验 |
|---|---|---|
| Anchor 只是随机扰动 | CASD 没有机制贡献 | random anchor negative control |
| Anchor 只利用 prompt 泄漏 | 检测端依赖 prompt，不够稳健 | prompt-free vs prompt-aware 消融 |
| Anchor 提高 region entropy 但不提高 semantic z | 只改善几何诊断，不改善水印统计 | end-to-end semantic detection |
| Anchor 提高检测但降低 pass rate | 选候选更激进，损伤代码质量 | pass@1 / syntax / execution |
| Anchor 只有在 oracle candidate center 下有效 | 只是 SeqMark 的高成本方案有效 | SeqMark oracle vs deterministic anchor |
| Anchor 只对某些 AST 节点有效 | 需要节点类型 gate，而不是全局使用 | node-type stratification |

## 2. Claim Map

| Claim | 为什么重要 | 最低可信证据 | 对应实验块 |
|---|---|---|---|
| C1: deterministic code anchor 缓解 region collapse | 这是 CASD 相对普通 LSH 与 SeqMark 的核心机制 | normalized region entropy 明显高于 vanilla LSH 和 random anchor，并达到 SeqMark oracle 增益的一定比例 | B1, B2 |
| C2: anchored semantic partition 能转化为检测证据 | 只改善候选空间几何还不够，必须提高 watermark detection power | semantic z、AUROC、TPR@5%FPR 在真实水印输出上提升，且 negative FPR 可控 | B3, B4, B5 |
| C3: anchor 的收益不依赖不可部署条件 | 方法不能依赖检测端调用原模型重采样 | prompt-free slot/context anchor 有有效收益；prompt-aware 只能作为上界或扩展 | B2, B6 |

## 3. 总体实验设计

实验分两条线并行推进：

1. **诊断线：** 不先改生产水印生成器，离线构造候选 block pool，直接测 Anchor 是否改变 region collapse。这个阶段成本低，能快速判断机制是否有希望。
2. **端到端线：** 在小规模 HumanEval/MBPP 子集上接入 anchored partition，验证 semantic detection 和 pass rate 是否真实改善。

如果诊断线失败，就不应继续投入端到端工程。

## 4. 核心指标

### 4.1 Region Entropy

对一个上下文 $c$，采样或收集 $K$ 个高质量候选 block：

$$
C_c=\{b_1,\dots,b_K\}
$$

用某个 partition 方法 $m$ 把候选映射到 region：

$$
r_j^{(m)} = h_m(b_j,c)
$$

经验分布：

$$
p_m(r\mid c)=\frac{1}{K}\sum_{j=1}^{K}\mathbf{1}[r_j^{(m)}=r]
$$

region entropy：

$$
H_m(c)=-\sum_r p_m(r\mid c)\log p_m(r\mid c)
$$

归一化 entropy：

$$
\bar H_m(c)=\frac{H_m(c)}{\log \min(K,R)}
$$

其中 $R$ 是可用 region 数。越接近 1，说明候选越分散；越接近 0，说明 collapse 越严重。

### 4.2 Collapse Ratio

$$
CR_m(c)=1-\bar H_m(c)
$$

Anchor 有效的最低要求是：

$$
CR_{\text{anchor}}(c) < CR_{\text{vanilla}}(c)
$$

并且不是 random anchor 也能做到同样程度。

### 4.3 Effective Region Count

$$
N_{\text{eff},m}(c)=\exp(H_m(c))
$$

它比 entropy 更直观：候选实际上覆盖了多少个有效 region。

### 4.4 Signature Diversity

对 LSH bit signature 计算平均 pairwise Hamming distance：

$$
D_m(c)=\frac{2}{K(K-1)}\sum_{j<k}
\frac{1}{d}\operatorname{Ham}(s_j^{(m)},s_k^{(m)})
$$

如果 Anchor 有效，$D_{\text{anchor}}$ 应该高于 vanilla，但不能高到像随机噪声一样完全破坏语义一致性。

### 4.5 Valid-Hit Balance

给定目标 hit 概率 $\gamma$ 和 secret key 派生的 valid region set $V_c$：

$$
q_m(c)=\frac{1}{K}\sum_{j=1}^{K}\mathbf{1}[h_m(b_j,c)\in V_c]
$$

理想情况下，非水印候选池中的 $q_m(c)$ 应该接近 $\gamma$。如果 Anchor 让 $q_m(c)$ 大量偏离 $\gamma$，检测端 null distribution 会失真。

### 4.6 Semantic Detection Power

端到端水印输出上，比较：

$$
z=\frac{\sum_i w_i(x_i-\gamma_i)}
{\sqrt{\sum_i w_i^2\gamma_i(1-\gamma_i)}}
$$

其中 $x_i$ 是 block 是否命中，$w_i$ 是可选的 capacity weight。Anchor 的收益必须体现在：

- mean semantic z 提升。
- semantic positive rate 提升。
- AUROC 提升。
- TPR@5%FPR 提升。
- `corr(embed_rate, semantic_z)` 从接近 0 变为正相关。

### 4.7 Code Quality

必须同时报告：

- syntax pass rate。
- benchmark pass@1。
- pass@10，如果候选文件和评测预算允许。
- generated code length。
- fallback/retry exhausted rate。

如果 Anchor 提升检测但显著损伤 pass rate，它不应作为主方案。

## 5. 比较系统

### 5.1 必须比较的 partition 方法

| ID | 方法 | 目的 | 是否主线 |
|---|---|---|---|
| M0 | vanilla LSH: `LSH(E(b))` | 当前语义信道基线 | 是 |
| M1 | random anchor residual | 排除随机扰动解释 | 是 |
| M2 | slot anchor | 使用 function signature + AST path + node type | 是 |
| M3 | context anchor | 使用 parent context + masked local context | 是 |
| M4 | skeleton anchor | 使用 masked block skeleton | 是 |
| M5 | slot + context anchor | 推荐 deterministic anchor 主候选 | 是 |
| M6 | slot + context + skeleton anchor | 更强但更复杂的 anchor | 是 |
| M7 | prompt-aware anchor | 诊断 prompt 是否必要 | 附录/上界 |
| M8 | SeqMark oracle center | 在线候选均值，上界诊断 | 是，但只做诊断 |

### 5.2 Anchor 构造说明

#### Slot Anchor

输入字段：

- function signature。
- dataset task id。
- AST path，例如 `FunctionDef > If > For > Assign`。
- node type。
- parent node type。

用途：捕捉结构槽位，不直接依赖当前 block 的实现内容。

#### Context Anchor

输入字段：

- block 前后固定窗口的非目标代码。
- parent statement 的 masked version。
- imports / helper signature。

用途：近似同一上下文下的共同语义方向。

#### Skeleton Anchor

输入字段：

- 当前 block 的 masked skeleton。
- 标识符、字面量、局部表达式替换为占位符。

用途：保留代码形状，去掉具体实现语义。

注意：skeleton anchor 有潜在泄漏风险，因为它来自当前 block。它适合做诊断和可选增强，但不应先作为唯一主 anchor。

#### Prompt-Aware Anchor

输入字段：

- benchmark prompt。
- function signature。
- natural language problem statement。

用途：判断 prompt 信息是否对 anchor 有决定性作用。若只有 prompt-aware anchor 有效，则最终方法必须明确要求 prompt-known 检测，或改用别的方案。

#### Random Anchor

构造：

$$
a_i^{\text{rand}}=\operatorname{Normalize}(\operatorname{PRNG}(key, block\_id))
$$

用途：负控制。若 random anchor 与 deterministic anchor 效果相同，说明收益可能只是投影扰动或维度变化，不是代码 anchor 的机制贡献。

#### SeqMark Oracle Center

构造：

$$
\mu_c=\frac{1}{K}\sum_{j=1}^{K}E(\tilde b_j)
$$

用途：上界。它不作为最终方案，因为生成端和检测端成本高，但可以回答：

> 如果使用真正的候选中心，region collapse 是否能被解决？

如果 SeqMark oracle 也无效，则 Anchor 主线没有意义，问题可能不在 region collapse。

## 6. 实验块

### B1: Candidate Pool Region Collapse 诊断

**Claim tested:** C1  
**Priority:** MUST-RUN  
**目标：** 在不改生产生成器的情况下，验证同一上下文的候选 block 是否真的发生 collapse，以及 anchor 是否缓解 collapse。  

#### 数据

- HumanEval 任务子集：优先 50 个任务；如果预算有限先取 25 个。
- MBPP 任务子集：优先 50 个任务。
- 每个任务抽取可替换 block context，按 node type 分层：
  - assignment / expression。
  - if / loop。
  - return。
  - helper call。
  - short block。

#### 候选生成

每个 context 生成 $K$ 个候选 block：

- quick setting: $K=16$。
- standard setting: $K=32$。
- upper setting: $K=64$，只用于少量 context。

候选生成只用于诊断，不作为最终部署成本。生成策略：

- 使用同一模型和 prompt。
- 温度 sweep: `0.2`, `0.4`, `0.7`。
- 保留语法合法候选。
- 可选执行测试过滤，若成本过高则先用 syntax + AST parse + length filter。

#### 比较方法

M0-M8 全部比较，但 M8 只在有 candidate pool 时计算。

#### 指标

- normalized region entropy $\bar H$。
- collapse ratio $CR$。
- effective region count $N_{\text{eff}}$。
- pairwise Hamming diversity $D$。
- node-type stratified $\bar H$。
- temperature stratified $\bar H$。

#### 成功标准

Anchor 进入下一阶段需要满足：

1. M5 或 M6 的平均 $\bar H$ 相比 M0 提升至少 `+0.10`。
2. 在低熵 context 上提升至少 `+0.15`。
3. M5/M6 的提升至少达到 SeqMark oracle 增益的 `50%`。
4. M5/M6 明显优于 random anchor M1；若差距小于 `+0.03`，判定机制不可信。
5. 至少两个主要 node type 上有效，而不是只对一种节点有效。

#### 失败解释

- M8 有效但 M5/M6 无效：deterministic anchor 不能近似候选中心，需考虑 amortized center 或 learned anchor。
- M8 也无效：region collapse 不是主要原因，回到 detector weighting、encoder alignment 或 lexical calibration。
- M1 与 M5/M6 同样有效：Anchor 机制假设不成立，只是随机投影扰动带来分散。
- 只有 M7 有效：方法依赖 prompt，必须改变检测威胁模型或限制适用场景。

#### 图表目标

- Figure 1: vanilla / random / deterministic / SeqMark oracle 的 normalized region entropy 箱线图。
- Figure 2: 按 node type 分组的 collapse ratio。
- Table 1: anchor ablation 诊断总表。

### B2: Anchor 成分消融与泄漏检查

**Claim tested:** C1, C3  
**Priority:** MUST-RUN  
**目标：** 判断哪些 anchor 成分真正有效，并检查是否依赖 prompt 或当前 block 内容泄漏。  

#### 比较系统

- M2 slot anchor。
- M3 context anchor。
- M4 skeleton anchor。
- M5 slot + context。
- M6 slot + context + skeleton。
- M7 prompt-aware。
- M1 random anchor。

#### 关键检查

1. **Prompt-free vs prompt-aware：** 如果 M7 明显强于 M5/M6，说明 prompt 信息是核心，prompt-free 检测会弱。
2. **Skeleton leakage：** 如果 M4/M6 显著强于 M2/M3/M5，但检测端改写或攻击后失败，skeleton 可能过拟合当前 block。
3. **Random anchor gap：** deterministic anchor 必须稳定优于 random anchor。
4. **Key independence：** anchor 本身不得使用 secret key；secret key 只能用于 valid region 选择。

#### 指标

- $\bar H$。
- $N_{\text{eff}}$。
- valid-hit balance $q_m(c)$。
- null hit probability deviation:

$$
\Delta_\gamma = \left|\mathbb{E}_c[q_m(c)]-\gamma\right|
$$

- across-key variance:

$$
\operatorname{Var}_{key}(q_m(c))
$$

#### 成功标准

- 推荐主 anchor M5 在 prompt-free 下相对 M0 有稳定提升。
- M5 不需要 skeleton 也有收益；M6 可以作为增强，但不能是唯一有效配置。
- $\Delta_\gamma \le 0.05$，否则检测端 null distribution 需要重新建模。
- random anchor 不得接近 M5/M6。

#### 失败解释

- 如果只靠 skeleton 有效，CASD 需要重新定义为 block-shape residual，不应声称是 context anchor。
- 如果 prompt-aware 明显必要，则应把方法改为 prompt-conditioned watermark，检测接口也必须保存或恢复 prompt。
- 如果 valid-hit balance 偏离严重，Anchor 会破坏 calibration，不能进入端到端。

#### 图表目标

- Table 2: anchor component ablation。
- Figure 3: prompt-free / prompt-aware / random control 的 $\Delta_\gamma$ 和 $\bar H$ 双轴图。

### B3: Offline Watermark Selection Simulation

**Claim tested:** C2  
**Priority:** MUST-RUN  
**目标：** 在 candidate pool 上模拟 rejection sampling，判断 Anchor 是否能在固定 retry budget 下选择到 valid region，同时不牺牲质量代理指标。  

#### 模拟过程

对每个 context $c$ 和候选池 $C_c$：

1. 用 secret key 和 block metadata 派生 valid region set。
2. 对每个 partition 方法 $m$ 计算每个候选是否命中。
3. 模拟生成期选择：
   - 若前 $B$ 个候选中有命中，选择第一个命中候选。
   - 若无命中，选择原模型最高概率或第一个候选作为 fallback。
4. 记录 accepted candidate 的质量代理指标。

retry budget：

- `B=1`：无 rejection。
- `B=4`：低成本。
- `B=8`：中等成本。
- `B=16`：接近诊断上限。

#### 质量代理指标

- syntax valid。
- AST parse valid。
- length ratio。
- indentation valid。
- optional unit test pass。
- perplexity/rank proxy，如果日志或模型概率可用。

#### 指标

- hit acquisition rate。
- fallback rate。
- retry exhausted rate。
- accepted quality proxy。
- semantic z proxy：

$$
z_{\text{proxy}}=\frac{\sum_i (x_i-\gamma_i)}
{\sqrt{\sum_i \gamma_i(1-\gamma_i)}}
$$

- block-level capacity vs hit correlation。

#### 成功标准

- 在 `B=4` 或 `B=8` 时，M5/M6 的 hit acquisition 明显高于 M0。
- fallback rate 低于当前 cap9 中失败模式的可接受上限。
- accepted candidate 的 syntax/parse/quality proxy 不低于 M0。
- M5/M6 的 z proxy 高于 M0，且 M1 random anchor 不能同等提升。

#### 失败解释

- region entropy 提升但 hit acquisition 不提升：anchor 分散候选，但 valid set 选择或 gamma 设计不匹配。
- hit acquisition 提升但 quality proxy 下降：需要 capacity gate 或 quality-aware retry，不宜直接部署。
- random anchor 同样提升：不是代码 anchor 机制，不能作为论文主贡献。

#### 图表目标

- Figure 4: retry budget vs hit acquisition / fallback。
- Table 3: offline simulation 的 z proxy 与质量代理。

### B4: 小规模端到端语义水印验证

**Claim tested:** C2  
**Priority:** MUST-RUN after B1-B3 pass  
**目标：** 将最佳 anchor 配置接入 generation-time semantic channel，验证真实输出上的 semantic detection 是否改善。  

#### 数据

- HumanEval 25 或 50 task subset。
- MBPP 50 task subset。
- 每个设置 3 seeds。

#### 比较系统

| 系统 | 说明 |
|---|---|
| S0 current WFCLLM semantic | 当前 vanilla semantic channel |
| S1 current WFCLLM + capacity-weighted detector only | 判断问题是否只在 detector |
| S2 AO-LSH M5 + current detector | 判断 anchor partition 本身是否有用 |
| S3 AO-LSH M5 + capacity-weighted detector | 推荐 CASD-lite |
| S4 AO-LSH M6 + capacity-weighted detector | 更强 anchor |
| S5 SeqMark oracle diagnostic | 小样本上界，不作为最终部署 baseline |

#### 控制变量

- 相同模型。
- 相同 prompts。
- 相同 max_new_tokens。
- 相同 temperature/top-p。
- 相同 gamma schedule。
- 相同 retry budget。
- 相同 secret key 派生协议，除 partition 方法外不变。

#### 指标

- semantic mean z。
- semantic positive rate。
- semantic AUROC。
- TPR@5%FPR。
- embed_rate。
- `corr(embed_rate, semantic_z)`。
- total_blocks / embedded_blocks / failed_blocks / fallback_blocks。
- pass@1 或 benchmark correctness。
- generation latency。
- encoder calls。

#### 成功标准

进入完整实验前，S3 至少满足：

1. semantic mean z 从 cap9 的负值或接近 0 提升到明显正值。
2. semantic positive rate 相比 S0 至少提升 `+15` 个百分点，或 AUROC 提升至少 `+0.10`。
3. `corr(embed_rate, semantic_z)` 明显为正，目标 `>= 0.25`。
4. pass@1 相对 S0 下降不超过 `2` 个百分点。
5. negative calibration 下 FPR 接近目标阈值。

#### 失败解释

- B1-B3 成功但 B4 失败：离线候选池不代表真实生成轨迹，或生成器选择过程引入偏差。
- S1 成功但 S2/S3 不明显：主要问题是 detector weighting，不是 Anchor。
- S2 成功但 S3 不成功：weighting 设计不当。
- S3 检测强但 pass 掉得明显：需要 safe gate，不能直接采用。

#### 图表目标

- Table 4: small end-to-end detection and pass。
- Figure 5: embed_rate vs semantic_z scatter，比较 S0 与 S3。

### B5: Negative Corpus Calibration 与 Null Distribution 检查

**Claim tested:** C2, C3  
**Priority:** MUST-RUN  
**目标：** 检查 Anchor 是否破坏水印检测的 null distribution，防止只在 positive 上涨但 FPR 不可控。  

#### 数据

- 非水印 LLM 生成代码。
- Human-written reference solution，如 HumanEval canonical solution。
- 现有 negative corpus，但需要确保 semantic scoring 字段完整。
- 如果 joint/lexical 参与，negative corpus 必须有真实 scored positions，不能再出现 lexical positions 为 0 却用于 joint calibration 的情况。

#### 检查项

- block hit rate under null 是否接近 $\gamma$。
- semantic z under null 是否近似标准正态或经验可校准。
- 按 node type、长度、entropy 分组的 FPR。
- prompt-known / prompt-free 两种检测条件。
- 多 secret key 重复，避免某个 key 偶然有利。

#### 指标

- FPR@threshold。
- empirical threshold for 1%, 5%, 10% FPR。
- KS test / QQ plot，检查 z 分布。
- key-wise threshold variance。
- subgroup FPR max gap。

#### 成功标准

- 使用经验校准后，5% 目标 FPR 的实际 FPR 在 `4%-6%` 区间，或有足够样本时置信区间覆盖 5%。
- subgroup FPR 不出现明显异常，例如某些 node type 超过全局 FPR 的两倍。
- key-wise threshold 不剧烈漂移。

#### 失败解释

- null hit rate 偏离 $\gamma$：anchor partition 改变了基础概率，必须重新建模或做 empirical calibration。
- subgroup FPR 高：需要 node-type gate 或分层阈值。
- threshold 随 key 漂移：key 派生或 region assignment 不稳定。

#### 图表目标

- Figure 6: null z QQ plot。
- Table 5: FPR calibration by subgroup。

### B6: Robustness and Attack-Side Diagnostic

**Claim tested:** C2, C3  
**Priority:** NICE-TO-HAVE before paper claim, not first gate  
**目标：** 验证 Anchor 不只是对原始生成代码有效，至少在轻度编辑后仍有语义统计信号。  

#### 攻击/扰动

- identifier rename。
- whitespace / formatting。
- local statement reorder when semantics safe。
- comment insertion/removal。
- simple variable introduction。
- light LLM paraphrase/refactor，预算允许时再做。

#### 指标

- semantic z retention。
- AUROC retention。
- pass retention。
- block alignment success rate。

#### 成功标准

- identifier/formatting 下 semantic AUROC 基本不下降。
- light refactor 下仍优于 vanilla semantic channel。

#### 失败解释

- 格式/rename 就失败：anchor 或 block alignment 过度依赖表面形式。
- refactor 失败但轻扰动成功：可作为当前方法边界，不影响第一版主张。

## 7. 执行顺序与 Stop/Go Gate

| Milestone | 目标 | 主要 runs | Stop/Go Gate | 预计成本 | 风险 |
|---|---|---|---|---:|---|
| M0 | 确认 candidate pool 和 metrics 正确 | 10 contexts, K=8, M0/M1/M5/M8 | 指标能稳定计算，M8 不全为 NaN | 低 | 候选池质量太差 |
| M1 | Region collapse 主诊断 | HumanEval 25, K=16/32, M0-M8 | M5/M6 比 M0 提升 >= +0.10，且优于 M1 | 中 | Anchor 无效 |
| M2 | Anchor 成分与 null balance | M1 数据 + 多 key | $\Delta_\gamma \le 0.05$，random control 不接近 | 中 | calibration 失真 |
| M3 | Offline selection simulation | retry B=1/4/8/16 | hit acquisition 提升且 quality proxy 不降 | 中 | 分散但选不到好候选 |
| M4 | 小规模端到端 | HumanEval 25/50, 3 seeds | semantic z/positive/AUROC 提升，pass 下降 <= 2pt | 高 | 工程接入耗时 |
| M5 | Negative calibration | non-watermarked + reference | empirical FPR 可控 | 中 | null 分布偏斜 |
| M6 | Robustness 诊断 | rename/format/light refactor | 轻扰动下信号保留 | 中 | alignment 失败 |

第一阶段只跑 M0-M3。只有 M1-M3 通过，才值得实现端到端 AO-LSH。

## 8. 推荐第一批实验

### R001: Toy Candidate Pool Sanity

- **目的：** 确认指标和 candidate pool 构造没有问题。
- **数据：** HumanEval 10 个任务，每个任务 3 个 block context。
- **K：** 8。
- **方法：** M0, M1, M5, M8。
- **输出：** region entropy、effective region count、random control gap。
- **通过条件：** 指标可计算；M8 有合理上界；M1 不异常支配全部方法。

### R002: HumanEval Anchor Collapse Main

- **目的：** 快速判断 deterministic anchor 是否有机制价值。
- **数据：** HumanEval 25 个任务，按 node type 分层。
- **K：** 16 和 32。
- **方法：** M0-M8。
- **输出：** anchor ablation table。
- **通过条件：** M5/M6 明显优于 M0 和 M1。

### R003: Valid-Hit Balance Multi-Key

- **目的：** 检查 Anchor 是否破坏 null hit probability。
- **数据：** R002 的 candidate pool。
- **keys：** 10 个 secret key。
- **gamma：** 0.25, 0.5, 0.75。
- **输出：** $\Delta_\gamma$、key-wise variance。
- **通过条件：** deterministic anchor 的 null hit rate 接近目标 gamma。

### R004: Offline Retry Simulation

- **目的：** 估算接入生成器后的收益和 retry 成本。
- **数据：** R002 candidate pool。
- **retry budgets：** 1, 4, 8, 16。
- **方法：** M0, M1, M5, M6, M8。
- **输出：** hit acquisition、fallback、quality proxy。
- **通过条件：** M5/M6 在 B=4 或 B=8 时有明显收益。

### R005: Small End-to-End Semantic Only

- **目的：** 验证真实生成输出上的 semantic detection。
- **数据：** HumanEval 25，3 seeds。
- **方法：** S0-S4。
- **输出：** semantic z、AUROC、TPR@5%FPR、pass@1。
- **通过条件：** S3 比 S0 检测提升且 pass 不显著下降。

## 9. 数据与实现需求

### 9.1 可复用现有 artifacts

优先复用：

- `data/watermarked/humaneval_full_cap9.jsonl`
- `data/results/humaneval_full_cap9_details.jsonl`
- `data/results/humaneval_full_cap9_summary.json`
- block diagnostics sidecar，如果包含 retry/candidate 轨迹。

但需要注意：cap9 artifact 很可能没有保存每个 context 的完整 candidate pool。如果没有 candidate pool，就需要离线生成诊断候选，不要把它误认为生产部署成本。

### 9.2 需要新增的离线诊断产物

建议输出到 `data/diagnostics/anchor_validation/`，但不默认纳入 git：

| 文件 | 内容 |
|---|---|
| `candidate_pools.jsonl` | 每行一个 context 的 K 个候选 block 和 metadata |
| `anchor_features.jsonl` | 每个候选的 anchor input、embedding id、node type |
| `region_metrics.jsonl` | 每个 context/method/key/gamma 的 entropy 和 balance |
| `selection_simulation.jsonl` | retry simulation 的选择结果 |
| `anchor_validation_summary.json` | 聚合指标与 stop/go 结论 |

### 9.3 需要的最小实现组件

仅诊断阶段需要：

1. `build_candidate_pools`：从已有生成日志或重新采样构造候选池。
2. `compute_anchor_embeddings`：给 slot/context/skeleton/prompt anchor 编码。
3. `compute_region_metrics`：计算 M0-M8 的 region entropy、collapse ratio、valid-hit balance。
4. `simulate_retry_selection`：在候选池上模拟水印选择。
5. `summarize_anchor_validation`：生成表格和图。

端到端阶段才需要改 `wfcllm.watermark` 中的 partition 逻辑。

## 10. 统计分析

### 10.1 配对比较

所有主要比较使用同一 context 上的 paired difference：

$$
\Delta \bar H(c)=\bar H_{\text{anchor}}(c)-\bar H_{\text{vanilla}}(c)
$$

报告：

- mean difference。
- median difference。
- bootstrap 95% CI。
- win rate：

$$
\operatorname{WinRate}=\frac{1}{N}\sum_c \mathbf{1}[\Delta \bar H(c)>0]
$$

### 10.2 分层分析

必须分层报告：

- dataset: HumanEval / MBPP。
- node type。
- block length bucket。
- entropy bucket。
- temperature。
- number of available candidates。

如果 Anchor 只在高 entropy block 有效，它不能解决当前 low-entropy bottleneck。

### 10.3 多 key 检查

每个 partition 方法用多个 secret key 计算 valid region set。关注：

$$
\mathbb{E}_{key,c}[q_m(c)]
$$

和：

$$
\operatorname{Var}_{key,c}[q_m(c)]
$$

如果 key variance 很高，单次实验可能只是 lucky key。

## 11. 明确的 Go / No-Go 标准

### GO

满足以下条件，可以继续把 Anchor 写进 CASD 主方案：

1. M5 或 M6 在 B1 中比 vanilla LSH 的 normalized region entropy 提升 `>= +0.10`。
2. 在低熵 context 上提升 `>= +0.15`。
3. deterministic anchor 至少达到 SeqMark oracle 增益的 `50%`。
4. deterministic anchor 比 random anchor 至少高 `+0.05` normalized entropy 或有显著 bootstrap CI。
5. valid-hit balance 偏差 $\Delta_\gamma \le 0.05$。
6. 小规模端到端 semantic AUROC 或 positive rate 明显提升。
7. pass@1 下降不超过 `2` 个百分点。

### PARTIAL GO

满足以下情况只能作为受限方案：

| 情况 | 结论 |
|---|---|
| 只有 prompt-aware anchor 有效 | 方法改为 prompt-known 检测设定；不能声称 prompt-free |
| 只有 skeleton anchor 有效 | 方法改为 shape-residual LSH；需额外验证鲁棒性 |
| region entropy 提升但端到端 detection 不提升 | Anchor 只能作为几何诊断，主修复转向 detector/selection |
| detection 提升但 pass 下降 2-5pt | 需要 safe gate 和更保守 retry，不直接采用 |

### NO-GO

出现以下任一情况，应停止 Anchor 主线：

1. SeqMark oracle center 也不能改善 region entropy。
2. deterministic anchor 与 random anchor 无明显差别。
3. Anchor 让 null hit probability 明显偏离 $\gamma$ 且难以经验校准。
4. Anchor 在低熵 context 上无效，只在高熵 context 上有效。
5. 小规模端到端中 semantic z 仍接近 0 或负值。
6. pass@1 显著下降，且 safe gate 不能修复。

## 12. 与 SeqMark 的关系

本验证方案中，SeqMark 不是要被直接复现为最终方法，而是作为 oracle diagnostic：

| 角色 | 解释 |
|---|---|
| 机制上界 | 如果真实候选均值都不能解 collapse，则 Anchor 没必要继续 |
| 成本参照 | 对比 deterministic anchor 是否以低成本获得部分收益 |
| 创新边界 | 证明 CASD 不是简单照搬 SeqMark，而是用代码可复现结构替代在线候选中心 |

最终论文叙事应该是：

> SeqMark 指出低熵约束生成存在 semantic region collapse；我们先用 oracle center 证明该问题在代码 block 上存在，再证明 deterministic code anchor 可以低成本近似解决其中一部分。

如果实验不能支撑这句话，就不应该把 Anchor 写成主贡献。

## 13. 风险与缓解

| 风险 | 可能原因 | 缓解 |
|---|---|---|
| Anchor 无效 | 代码 encoder 没有线性分离 task/common direction 和 implementation direction | 尝试 learned projection 或转向 detector weighting |
| Random anchor 同样有效 | 投影扰动而非代码结构起作用 | 放弃 Anchor 机制 claim，仅保留随机投影作为工程 trick 也要谨慎 |
| Prompt anchor 才有效 | 结构信息不足以恢复任务语义 | 明确 prompt-known 检测，或引入 prompt embedding side metadata |
| Null distribution 失真 | anchor 与 node type/length 相关，导致 hit probability 偏斜 | empirical calibration、node-type threshold、capacity weights |
| Pass rate 下降 | anchored valid region 选择与代码质量冲突 | safe gate、quality-aware retry、降低 gamma/delta |
| 候选池诊断与真实生成不一致 | 离线采样分布不同于生产 rejection loop | 小规模端到端作为硬 gate |

## 14. 最终应产出的表和图

| 编号 | 内容 | 对应 claim | 位置 |
|---|---|---|---|
| Table 1 | Anchor region entropy ablation | C1 | 主文 |
| Table 2 | Random / prompt / skeleton leakage control | C1, C3 | 主文或附录 |
| Table 3 | Offline retry simulation | C2 | 主文 |
| Table 4 | Small end-to-end detection/pass | C2 | 主文 |
| Table 5 | Negative calibration/FPR | C2 | 主文 |
| Figure 1 | Region entropy boxplot | C1 | 主文 |
| Figure 2 | Node-type collapse heatmap | C1 | 附录 |
| Figure 3 | embed_rate vs semantic_z scatter | C2 | 主文 |
| Figure 4 | retry budget vs hit acquisition | C2 | 主文 |
| Figure 5 | Null z QQ plot | C2 | 附录 |

## 15. 最小执行清单

先不要直接改生产水印逻辑。最小执行路径是：

1. 构造 HumanEval 25-task candidate pool。
2. 实现 M0/M1/M5/M8 的 region metric。
3. 跑 R001 sanity。
4. 扩展到 M0-M8，跑 R002。
5. 做 multi-key valid-hit balance，跑 R003。
6. 做 offline retry simulation，跑 R004。
7. 如果 R002-R004 通过，再接入端到端 AO-LSH，跑 R005。

## 16. 计划的关键判断

这份验证计划的核心判断是：

> Anchor 是否有效，不能由最终 detection 数字单独决定；必须同时看 region entropy、random-anchor gap、SeqMark oracle gap、null hit balance、pass rate 和端到端 semantic z。

如果这些指标同时支持 Anchor，CASD 的 Anchor-Orthogonal LSH 才有资格进入最终路线。否则，最诚实的修复方向应该回到 capacity-weighted detector、safe lexical auxiliary、或更接近 SeqMark 的 amortized center，而不是继续堆 Anchor 复杂度。
