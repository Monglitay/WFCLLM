# 双通道强 ACW 水印设计

**日期：** 2026-04-10  
**作者：** OpenCode  
**状态：** 设计已确认，待实现

## 1. 概述

本设计旨在为 `WFCLLM` 引入一条参考强 ACW 思路的 **token 级词法水印通道**，并与现有 **block 级语义水印通道** 组成双通道系统。

当前系统已经具备以下核心能力：

1. 基于 `AST simple statement block` 的块切分与共享契约。
2. 基于语义编码器、LSH、重试与级联回退的 block 级主通道。
3. 基于最终代码重算的提取与统计检验流程。

本设计不替换现有主通道，而是在其之上新增一条 **可训练、可生成、可检测、可联合判决** 的 token 级第二通道。该第二通道参考论文与原仓库中的强 ACW 思路，但在 `WFCLLM` 中按当前架构和离线约束重新设计，不直接搬运原仓库代码、模块或命名。

## 2. 目标与非目标

### 2.1 目标

- 保留现有 block 级语义水印通道作为主通道。
- 新增一条强 ACW 风格的 token 级词法通道。
- 在生成阶段实现双通道共存：token 通道逐 token 施加偏置，block 通道负责最终验收与回滚。
- 在检测阶段只依赖 `最终代码 + secret key + 本地模型资产` 重放两条通道。
- 输出 `semantic_result`、`lexical_result`、`joint_result` 三套检测结果。
- 保持项目离线可运行、可训练、可测试、可回归。

### 2.2 非目标

- V1 不替换现有 block contract、语义编码器、LSH 或 block rollback 主线。
- V1 不将 token 通道提升为主导通道。
- V1 不训练额外的学习型 fusion classifier。
- V1 不承诺 token 通道对强格式化、强改写或深度语法等价变换仍保持高鲁棒性。
- V1 不扩展到多语言；实现范围限定为 Python。

## 3. 已确认的关键决策

### 3.1 产品与协议决策

- **双通道结构：** `语义主通道 + 词法辅通道`。
- **优先级：** token 通道负责细粒度偏置；block 通道负责最终提交与回滚。
- **检测输入约束：** 检测不得依赖 prompt、生成轨迹、原始 logits 或在线 LLM 状态。
- **词法通道形态：** 使用强 ACW 风格的 token 级策略模型，而不是纯规则 slot 方案。
- **联合输出：** 必须同时保留单通道分数和联合分数，不能只输出单个黑箱结果。
- **兼容原则：** 第二通道必须可以独立开关，便于对照实验与回归。

### 3.2 方案比较与结论

讨论过的三类方案如下：

1. **块内规则词法通道**
   - 优点：改造风险低，检测最容易重建。
   - 缺点：离强 ACW 较远，token 级表达能力有限。
2. **弱 ACW 风格 token 通道**
   - 优点：比纯规则方案更接近论文，工程复杂度低于强 ACW。
   - 缺点：候选空间受限，检测上限有限。
3. **强 ACW 风格 token 通道**
   - 优点：更接近论文原始思路，可形成真正的双尺度双通道。
   - 缺点：需要新训练资产与更深的生成/检测接入。

结论：V1 采用 **方案三**，但严格服从 `WFCLLM` 现有 block 级主干，不将其实现为平行项目或独立推理框架。

## 4. 总体架构

### 4.1 架构分层

1. **Shared Runtime Layer**
   - 复用现有 LLM、prompt 流程、watermark pipeline、extract pipeline。
2. **Semantic Channel Layer**
   - 现有 block 级语义水印通道。
3. **Lexical Channel Layer**
   - 新增 token 级词法策略模型、bias 逻辑与检测重放逻辑。
4. **Fusion Layer**
   - 负责三套结果的标准化、加权和解释性输出。

### 4.2 运行时主原则

- token 通道逐 token 介入，但不单独拥有回滚权。
- simple block 闭合后，仍由语义通道做最终验收。
- 若 block 语义验收失败，则整块回滚，block 内的 token 通道信号一并撤销。
- 检测端两条通道并行运行，再进入联合层。

## 5. 词法通道模型设计

### 5.1 模型职责

词法通道模型是一个轻量双头模型，职责只有两项：

1. **Gate Head**
   - 预测当前位置是否适合进行 token 级水印嵌入。
2. **Preference Head**
   - 为当前位置生成条件化 token 排序依据，供 keyed green/red 分流与 bias 使用。

该模型不承担完整语言建模职责，不替代主 LLM，仅服务第二通道。

### 5.2 输入特征

V1 采用 `token 为主，轻量结构特征为辅` 的输入设计：

- 最近 `context_width` 个 token 上下文。
- 当前是否处于代码正文区域。
- 当前所属 block 的轻量结构属性：如 `node_type`、`parent_node_type`、block 内相对偏移等。
- 语言标签，V1 固定为 Python，但保留字段。

### 5.3 输出语义

- `switch_logit`：当前位置是否值得嵌入。
- `token_preference_logits`：当前位置的 token 排序依据。

生成时，`switch_logit` 决定是否启用 token 通道；`token_preference_logits` 结合 key 生成 green/red 分流并施加 bias。检测时，二者被重新重放。

## 6. 训练设计

### 6.1 参考原则

参考原仓库后，保留其核心训练抽象：

- 先构造语义等价训练样本。
- 再对样本离线提取 teacher logits 与 entropy。
- 再将样本组织成 `prefix -> next token / logits / switch target` 的监督学习问题。

但在 `WFCLLM` 中必须按现有项目边界重写，不直接复用原仓库训练模块。

### 6.2 训练数据来源

训练数据由两层构成：

1. **基础代码样本**
   - 来自当前项目已有离线任务与本地语料来源，如 HumanEval、MBPP 与项目现有生成语料。
2. **语义等价增强样本**
   - 复用并扩展 `wfcllm/common/transform/positive/*` 规则，生成语义等价代码变体。

### 6.3 Teacher 信号

对原始样本和增强样本统一离线运行主 LLM，提取每个位置的：

- next-token teacher logits
- entropy
- token 序列与 prefix 对齐信息

这些产物作为 token 通道训练监督，不进入最终检测依赖。

### 6.4 训练样本组织

训练集采用固定上下文滑窗，将样本组织为：

- `prefix_tokens`
- `next_token`
- `teacher_logits`
- `entropy`
- `continuation_diversity`
- `structure_mask`

其中 `continuation_diversity` 表示同一 prefix 在增强样本集中可导向多少种 continuation，用于补强 gate 监督。

### 6.5 Switch Target 设计

V1 中 `switch target` 不只依赖单一熵阈值，而是由三部分共同决定：

1. **Entropy Signal**
   - 高 entropy 位置更可能具备嵌入自由度。
2. **Continuation Diversity Signal**
   - 同一 prefix 下存在多个 continuation 时，更适合做分流。
3. **Structure Mask**
   - 只允许代码正文与结构稳定区域参与训练和重放。

### 6.6 损失函数

建议采用三项主损失：

1. **Distillation Loss**
   - 对齐主 LLM 的 next-token logits，保持自然性。
2. **Cross-Entropy Loss**
   - 对齐真实 next token，保持局部正确性。
3. **Switch BCE Loss**
   - 学习当前位置是否应打开 token 通道。

可选保留一个后续扩展接口：

4. **Diversity-Aware Ranking Regularizer**
   - 用于强化多 continuation 前缀上的排序能力；V1 不要求实现。

## 7. 生成阶段设计

### 7.1 双层控制流程

生成阶段采用 `token 内层 + block 外层` 的控制结构：

1. 主 LLM 逐 token 输出 logits。
2. token 通道模型读取当前位置 prefix，输出 `gate + preference`。
3. 若 gate 打开，则基于 `preference + secret key` 构建 green/red 分流，并对 green 侧加 bias。
4. 采样得到当前 token。
5. 当现有系统检测到 simple block 闭合时，交由语义通道做 block 级验收。

其中 token 通道的 green/red 分流协议在 V1 中固定定义为：

1. 使用 token 通道模型输出的 `token_preference_logits` 对**完整 tokenizer 词表**排序。
2. 对排序后的 token 序列按相邻两两分组：`(t0, t1), (t2, t3), ...`。
3. 以 `secret key + prefix` 为种子生成确定性随机位，对每一对中选择一个 token 放入 green 集，另一个自动归入 red 集。
4. 若词表大小为奇数，则最后一个 token 由同一随机流独立决定是否纳入 green。
5. 生成时对 green 集中 token 施加 bias；检测时用相同协议重放 green/red 集。

该协议对生成和检测同时生效，不允许一端使用 top-k 子集、另一端使用全词表。采样阶段虽然仍受主 LLM 的 `top_k`、`top_p` 等过滤影响，但 green/red 的定义始终以完整词表为准。

### 7.2 优先级与回滚

- token 通道可以影响当前 token 的采样分布。
- token 通道不能单独决定 block 是否提交。
- block 语义验收失败时，整块回滚，block 内 token 通道信号全部失效。
- block 重试和可选级联回退继续由现有语义主通道负责。

### 7.3 运行时保护策略

为避免 token 通道破坏生成稳定性，V1 的运行时保护策略不是启发式建议，而是第 12 节定义的**正式协议规则**。生成实现必须至少支持：

- 短 block 关闭规则
- 高重试收敛规则
- 低 gate 比例关闭规则
- 结构敏感位置屏蔽规则

## 8. 检测阶段设计

### 8.1 基本原则

检测端只依赖：

- 最终代码
- tokenizer
- secret key
- token 通道模型
- 语义编码器与现有提取配置

检测不得依赖生成轨迹、prompt、原始 logits 或生成时缓存。

### 8.2 词法检测流程

对最终代码逐位置重放 token 通道：

1. 重新 tokenization。
2. 对每个位置构造 prefix。
3. 重放 token 通道模型，得到 `gate + preference`。
4. 用相同 key 和相同分流规则重建当前位置的 green/red 集。
5. 若 gate 打开，则检查最终 token 是否命中 green。
6. 累计统计得到 `num_positions_scored`、`num_green_hits`、`green_fraction`、`lexical_z_score`、`lexical_p_value`。

其中第 4 步的分流规则必须严格复用第 7.1 节定义的完整词表排序与相邻配对协议。

### 8.3 检测稳定性约束

为确保重放成立，必须满足：

- 生成端和检测端共享同一 tokenizer 版本。
- 后处理链条必须可控；会改变 token 序列的后处理必须禁用或纳入协议。
- 是否忽略重复 n-gram 或重复 prefix 必须在配置中显式定义。

V1 建议同时提供两个独立配置项：

- `ignore_repeated_ngrams`
- `ignore_repeated_prefixes`

## 9. 语义通道与词法通道的联合策略

### 9.1 角色分工

- **语义通道：** 主证据，负责鲁棒性和 block 级可解释性。
- **词法通道：** 辅证据，负责更高密度的 token 级统计增益。

### 9.2 联合输出

检测结果必须同时保留：

- `semantic_result`
- `lexical_result`
- `joint_result`

V1 联合层采用显式加权统计融合，而不训练单独 fusion 模型。

### 9.3 联合判决建议

推荐计算：

```text
joint_score = w_sem_eff * z_sem + w_lex_eff * z_lex
```

其中：

- `w_sem_eff = joint_semantic_weight`
- `w_lex_eff = joint_lexical_weight * lexical_support_factor`
- `lexical_support_factor` 默认定义为 `min(1.0, num_positions_scored / lexical_full_weight_min_positions)`
- 默认要求 `joint_semantic_weight > joint_lexical_weight`

V1 默认联合判决规则建议为：

```text
prediction = joint_score >= joint_threshold
confidence = 1 - p_joint
```

其中：

- `joint_threshold` 默认取 `4.0`
- `p_joint` 由标准正态分布上尾概率计算：`p_joint = 1 - Phi(joint_score)`
- 以上规则为默认实现，具体权重允许配置化

同时应输出解释性标签，例如：

- `semantic strong, lexical weak`
- `semantic borderline, lexical supportive`
- `semantic weak, lexical unsupported`

### 9.4 三种运行模式的正式含义

为避免实现歧义，V1 中三种模式定义如下：

1. **semantic-only**
   - 使用现有 block 级语义通道进行生成与检测。
   - token 通道完全关闭。
2. **lexical-only**
   - 作为**实验与消融模式**存在。
   - 生成时启用 token 通道 bias，但关闭语义水印验收、语义重试与语义统计输出。
   - 该模式不作为默认生产模式，仅用于评估第二通道的独立效果。
3. **dual-channel**
   - 默认目标模式。
   - 生成时 token 通道逐 token 施压，语义通道负责 block 级提交、重试与回滚。
   - 检测时同时输出语义、词法与联合结果。

因此，`语义通道是主通道` 仅适用于默认的双通道运行模式，不否定 `lexical-only` 作为受控消融模式的存在。

## 10. 配置与产物设计

### 10.1 新增产物类别

V1 需要新增以下本地资产：

1. **token 通道训练语料缓存**
2. **teacher 提取缓存**
3. **token 通道模型 checkpoint**
4. **模型元数据文件**
   - 记录 tokenizer 版本、context width、训练配置、特征版本等

### 10.2 配置约束

建议新增 token 通道配置组，至少包含：

- `enabled`
- `model_path`
- `context_width`
- `switch_threshold`
- `delta`
- `ignore_repeated_ngrams`
- `ignore_repeated_prefixes`
- `joint_semantic_weight`
- `joint_lexical_weight`
- `lexical_full_weight_min_positions`
- `joint_threshold`
- `debug_mode`

同时要求配置能单独开关：

- 仅语义通道
- 仅词法通道
- 双通道联合

其中 `仅词法通道` 明确指上文定义的 **实验与消融模式**，而不是默认生产路径。

### 10.3 模块落点约束

为降低后续 planning drift，新增组件应优先落在 `wfcllm/` 内部，而不是新建平行根级项目。建议模块分区如下：

- `wfcllm/watermark/token_channel/`
  - 训练语料构建
  - teacher 提取
  - 词法策略模型
  - 训练入口与产物加载
- `wfcllm/watermark/`
  - 生成阶段 token 通道接入
- `wfcllm/extract/`
  - token 通道检测重放与联合输出

`experiment/` 仍只作为参考区，不进入生产依赖。

## 11. 风险与缓解

### 11.1 主要风险

1. **token 通道训练不收敛或分流质量不佳**
   - 可能导致检测增益有限。
2. **生成质量下降**
   - bias 过强可能降低代码正确率，增加 block 重试成本。
3. **与 block rollback 耦合后推理开销上升**
   - token 通道越激进，整体重试成本可能越高。
4. **tokenizer 或后处理漂移破坏检测重放**
   - 是 token 通道的结构性风险。
5. **联合增益不明显**
   - 可能出现语义通道已足够强、第二通道提升有限的结果。

### 11.2 缓解策略

- 保留第二通道的独立开关和对照实验入口。
- 默认由语义通道主导提交和回滚。
- 对 token 通道引入 gate、结构过滤和高重试时自动收敛机制。
- 将 tokenizer 与模型元数据版本化。
- 在最终报告中分别展示 semantic-only、lexical-only、joint 三组结果。

## 12. 运行时保护规则

V1 将运行时保护规则定义为**正式协议的一部分**，但所有阈值均配置化，允许实验调整。以下默认值作为实现起点：

### 12.1 短 block 关闭规则

- 若一个 simple block 的可计分代码 token 少于 `8` 个，则该 block 不启用 token 通道。
- 该阈值配置项建议命名为 `lexical_min_block_tokens`。

### 12.2 高重试收敛规则

- 在 `dual-channel` 模式中，若同一个 simple block 的语义验收失败次数达到 `2` 次，则 token 通道 bias 强度衰减为 `0.5 * delta`。
- 若失败次数达到 `4` 次，则该 block 剩余尝试全部关闭 token 通道。
- 阈值配置项建议命名为 `lexical_retry_decay_start` 与 `lexical_retry_disable_after`。

### 12.3 低 gate 比例关闭规则

- 对 block 内前 `16` 个可计分 token 统计 gate 打开比例。
- 若比例低于 `0.10`，则判定该 block 上 token 通道证据过稀，剩余 token 不再施加 bias。
- 配置项建议命名为 `lexical_gate_probe_tokens` 与 `lexical_gate_min_fraction`。

### 12.4 结构敏感位置屏蔽规则

- 以下位置在 V1 中默认不参与 token 通道训练与运行时 bias：
  - `import_statement`
  - `import_from_statement`
  - `function_definition` 的 signature 区域
  - `class_definition` 的 header 区域
  - decorators
- 这些限制通过结构 mask 实现，视为协议约束而非临时启发式。

## 13. 验证标准

### 13.1 训练级验证

- switch 正负样本分布合理。
- token 排序能够拟合 teacher 结构。
- 训练/验证 loss 稳定收敛。

### 13.2 生成级验证

- 在同一固定评测集上，相比 `semantic-only` 基线：
  - `pass@1` 绝对下降不超过 `2` 个百分点；
  - `pass@10` 绝对下降不超过 `3` 个百分点；
  - simple block 平均重试次数上升不超过 `25%`；
  - 平均单样本生成耗时上升不超过 `35%`。

### 13.3 检测级验证

必须分别评估：

- semantic-only
- lexical-only
- joint

指标至少包含：

- z-score 分布
- TPR / FPR
- ROC AUC

并以以下阈值作为 V1 检测成功标准：

- `lexical-only` 的 ROC AUC 不低于 `0.65`；
- `lexical-only` 在 `FPR <= 1%` 处的 TPR 不低于 `0.20`；
- `joint` 相比 `semantic-only` 至少满足以下二者之一：
  - ROC AUC 提升不少于 `0.02`；
  - 在 `FPR <= 1%` 处 TPR 提升不少于 `0.05` 绝对值。

### 13.4 攻击级验证

至少覆盖以下轻量扰动：

- 格式化变化
- 注释变化
- 局部变量重命名
- 轻度等价改写

预期为：

- 语义通道更稳
- 词法通道下降更多
- 联合结果在至少一部分场景下优于语义单通道

## 14. V1 成功标准

V1 视为成功，应满足以下三条：

1. **不明显伤害生成质量**
   - 满足第 13.2 节中的全部生成级阈值。
2. **词法通道可独立检出**
   - 满足第 13.3 节中的 `lexical-only` 检测阈值。
3. **联合检测优于语义单通道**
   - 满足第 13.3 节中的 `joint` 提升阈值。

## 15. 结论

本设计将 `WFCLLM` 的未来方向明确为：

**以现有 block 级语义水印通道为主干，引入一条参考强 ACW 思路的 token 级词法水印通道，形成真正的双尺度双通道系统。**

V1 的重点不是复刻论文仓库本身，而是把其核心思想转换为与 `WFCLLM` 当前架构相容的训练、生成、检测和联合判决协议。在该前提下，第二通道既能提供更细粒度的统计信号，又不会破坏现有 block 级主干和离线工程约束。
