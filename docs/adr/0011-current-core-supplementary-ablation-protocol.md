---
status: accepted
---

# 补充消融以当前 Reproduction Core 为基准

补充材料把当前 `gated_semantic_window_v1` 实现视为论文方法的权威实现，不追溯
论文 PDF 与当前配置的历史差异。补充消融只研究论文正文未覆盖的机制因素，并
使用当前完整生产后端、完整 Python/HumanEval 数据和完整八阶段 fresh-run 链；
它们是可引用的受控比较，不是新方法、参数搜索、测试后端或历史复现入口。

## 排除范围

以下属于 Paper-Covered Analysis，不重复执行：

- proposal budget `B`；
- keyed valid-region ratio `gamma`；
- OpenCoder-1.5B 到 OpenCoder-8B 的模型规模迁移；
- DIPPER 与变量重命名鲁棒性分析。

context length、learning rate、batch size 和 suitability loss weight 属于 Gate
训练选参，不进入本轮机制消融。

## 实验矩阵

所有比较采用 one-factor-at-a-time：每个变体只改变表中一个因素，其余字段
保持当前默认值。默认点只运行一次，并同时作为六条曲线的共同对照。矩阵共有
15 个唯一 full run：

| Factor | Levels | Current default |
|---|---|---|
| Semantic signature dimension `d` | `8, 12, 14` | `12` |
| Stable semantic margin `delta` | `0.000, 0.005, 0.010, 0.020` | `0.000` |
| Suitability threshold `tau_suit` | `0.30, 0.50, 0.70` | `0.50` |
| Closure-band center `tau_close` | `0.40, 0.50, 0.60` | `0.50` |
| Minimum reliable windows `n_min` | `1, 2, 3, 4` | `2` |
| Maximum carrier extent `max_units` | `1, 2, 3` | `3` |

`d` 的上界取 14，而不是 16，因为当前 keyed-region 构造会显式枚举 `2**d`
个签名；14 已能观察高维代价，同时避免把一次参数分析变成不受控的指数级资源
实验。每个 `d` 点必须记录 nominal `gamma`、实际区域数 `k` 和实现得到的
`k / 2**d`，但不得改变 `gamma`。

`delta` 使用严格判定 `min_margin > delta`，并在 gate-data 标签、生成候选
接受和检测可靠性中保持一致。任何一处仍硬编码 `0.0` 都应由合同测试拒绝。

`tau_suit` 只改变冻结 Gate 输出被接受为 Suitable Window 的阈值，不改变
suitability 标签或 loss。

`tau_close` 定义为 closure uncertainty band 的中心。band width 固定为
`0.10`，因此三个点分别使用 `[0.35, 0.45]`、`[0.45, 0.55]` 和
`[0.55, 0.65]`。生成端与检测端必须使用同一 band；artifact 同时记录中心、
下界、上界和宽度。

`n_min` 只改变一个程序形成可评分检测所需的 Reliable Window 数量。虽然它
可以从已有 score 后处理重算，每个点仍执行完整 fresh-run 链，以避免增加恢复
入口或旁路实验身份。

`max_units` 是端到端协议因素。相应变体的 gate-data 只构造
`1..max_units` 窗口，closure 标签的最大边界、Gate Bundle、生成端和检测端都
使用同一个值；它不是仅在检测时截断已有窗口。

## 执行和随机性

实验只覆盖 `python/humaneval/full`，使用完整 164-task 数据集。跨语言与
MBPP 已由主结果承担，不在参数消融中重复，避免把机制效应和语言/数据集效应
混在一起。

每个矩阵点必须：

1. 从新的 variant root、新的 pilot/full run root 和新的状态文件开始；
2. 执行
   `encoder → gate-data → gate-train → generate → calibrate → detect → report → audit`；
3. 训练并只消费本 variant 的 semantic encoder 和 Gate Bundle；
4. 不读取其他 variant 的 Bundle、checkpoint、candidate、report 或状态；
5. 使用同一 Ablation Family 的 pilot/full source catalogs、Training Key
   Bank、Holdout Key Bank、Deployment Key、校准负样本和测试负样本；
6. 保持当前 encoder、Gate、生成和候选采样种子不变。

共享私有材料用于控制变量，不允许把原始 key 写入任何公开 artifact。公开身份
只记录现有 Key Identifier、key-bank manifest identity 和 hash。每个参数点
只运行一个固定种子；因此报告不得声称多种子置信区间或统计显著性。

## 校准和测试负样本

每个变体独立校准 `target_fpr=0.05`，不能沿用默认点的阈值。校准使用同一个
严格四字段 held-out unwatermarked corpus。

actual FPR 必须在另一份独立、严格四字段的 unwatermarked test-negative
corpus 上计算；该 corpus 不得参与阈值选择，并须覆盖与 164 个正样本相同的
目标任务身份。校准负样本和测试负样本必须按 ID、prompt 和 normalized code
检查不相交。任一输入缺失、数量不足或隔离失败都应在创建 run tree 前失败。

正式 detector positive input 仍然只能是 `inputs/final_code.jsonl`，并保持
`id,dataset,prompt,final_code` 四字段。负样本检测明细、generation sidecar、
正确性结果和消融汇总不得进入 positive input。

## 结果合同

每个参数点至少报告：

- `Passed/Total` 与 posthoc Pass@1；
- AUROC、`Detected/Total`、全样本 TPR；
- target FPR、独立测试集 actual FPR 及其分子/分母；
- watermarked/unwatermarked insufficient-evidence 数量；
- scoreable coverage、平均 Reliable Window 数和检测级 abstention；
- 每程序 carrier 数、Suitable Window 通过率、candidate-zero 接受率；
- candidate attempts、duplicate、结构合法、detector recoverable、
  stable-margin、keyed-hit、最终接受和 budget-exhausted 的数量与比例；
- 每样本端到端生成延迟、总秒数、中位数、p95 和相对默认点 multiplier。

延迟从单个 prompt 的 base generation 开始计时，到最终水印代码交付结束；
包括 finalizer、Gate、rewrite 与候选验证，不包括模型加载、encoder/Gate 训练、
校准、检测或报告。所有参数点使用相同硬件与串行样本协议。

AUROC 使用 watermarked positives 与独立 test negatives 的检测统计量。TPR
分母包含 insufficient-evidence positives；actual FPR 分母包含
insufficient-evidence negatives。insufficient evidence 不得从分母中删除。

## 身份和入口

Supplementary Ablation 保持方法名 `gated_semantic_window_v1` 和 profile
`full`，不新增方法 preset、fast/smoke profile、生产后端或旧 artifact reader。
现有 canonical full 配置文件保持不变。实验入口只能在现有
`run.py`/`wfcllm.cli.entry` 与
`scripts/experiments/run_gated_experiment.sh` 上接受一个经过白名单验证的
Supplementary Ablation Spec；任意参数覆盖、同时改变两个因素、请求
Paper-Covered Analysis 或在非 Python/HumanEval profile 上运行都必须失败。

每个 run 的 config hash、状态、Gate artifacts、generation manifest、
calibration、detection、report 和 audit 都绑定：

- `study_kind=supplementary_ablation`；
- factor；
- canonical level；
- default level；
- canonical baseline config hash；
- Ablation Family identifier；
- 本 variant 的完整 resolved-config hash。

这些 full-production 变体保持 `formal_eligible=true`、
`diagnostic_only=false` 和 `not_official_method=false`，但不得替代 canonical
Full Reproduction Profile 的默认结果。fixture 或注入后端产生的测试产物仍
必须使用完整非正式标记。

## 验收

实现完成的最低条件是：

- 五个现有 Full Reproduction Profile 在未提供消融 spec 时行为和 identity
  不变；
- 白名单矩阵的每个点能形成不同且稳定的 config hash；
- arbitrary override、多个 factor、paper-covered factor、错误默认值和旧
  artifact 输入均被拒绝；
- `delta`、Gate thresholds、`n_min` 与 `max_units` 在训练、生成、检测和
  artifact hash 之间没有静默硬编码分叉；
- actual FPR 来自独立 test negatives；
- detector positive input 和秘密隔离审计继续通过；
- compileall、离线 collect-only、完整测试、shell syntax、状态命令和
  removed-concept residue scan 全部通过。

代码验证只能证明消融执行合同，不得把 fixture 测试称为真实 GPU 消融结果。
