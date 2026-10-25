# WFCLLM Supplementary Ablation 半数据量实验结果表

## 1. 实验范围

本表汇总 `gated_semantic_window_v1` 在以下冻结协议下完成的 15 个
one-factor-at-a-time（OFAT）正式实验：

- 语言与数据集：Python / HumanEval；
- profile：`full`；
- study ID：`issue-3-half-20260730-gate-health-scope-fix`；
- 固定种子：单一固定 seed，仅报告描述性结果，不作多种子显著性声称；
- 生成正样本：HumanEval 前半部分 `82/164`；
- calibration negatives：`50/100`；
- independent test negatives：`82/164`；
- Gate 训练：保持方法合同要求的 300 个独立 source groups；
- 每个正式实验均从新的 run root 开始，独立执行：
  `encoder → gate-data → gate-train → generate → calibrate → detect → report → audit`；
- 每个实验均独立生成自己的 encoder artifact 和 Gate Bundle；
- 所有实验均通过 Formal Eligibility、Metric Contract 和 artifact audit；
- Pass@1 仅在生成及检测完成后计算，不参与生成、重试、选择、校准或检测。

机器可读独立审计：

```text
/root/autodl-tmp/wfcllm-issue3-half-control-20260730T151500Z/final_independent_audit.json
SHA-256: 26c511dd514a239fb748426b78d71da07ff66e0512262a4e3ac1652413e13733
```

## 2. Baseline 冻结配置

除表格中明确列出的唯一实验因素外，其他参数均保持 baseline 不变。

| 参数 | Baseline 值 | 具体含义 |
|---|---:|---|
| `d` | 12 | LSH 区域编码维度；控制语义空间哈希分区的粒度 |
| `delta` | 0.0 | Reliable Window 判定使用的精度/稳定性余量 |
| `tau_suit` | 0.5 | Gate suitable head 的接受阈值 |
| `tau_close` | 0.5 | Gate close/agreement head 的接受阈值 |
| `n_min` | 2 | detector 给出可评分结论所需的最少 Reliable Windows 数 |
| `max_units` | 3 | statement window 最多包含的语句单元数 |
| nominal `gamma` | 0.45 | 固定，不属于本次消融因素 |
| generation model | OpenCoder-1.5B-Instruct | 所有实验使用相同冻结模型和 tokenizer |

## 3. 指标说明

| 指标 | 含义 |
|---|---|
| Pass@1 | 生成代码的 posthoc 正确率；本研究中所有实验均为 `66/82 = 0.804878` |
| AUROC | positive detection statistic 与 independent test-negative statistic 的排序区分能力 |
| TPR | 82 条正式 positive 中被判定为水印代码的比例 |
| actual FPR | 82 条 independent test negatives 中被误判为水印代码的比例 |
| Positive / Negative Coverage | 具备不少于 `n_min` 个 Reliable Windows、因而可以正式评分的正/负样本比例 |
| Gate Coverage | generation 机制中具有 Gate 可用窗口的覆盖率；它与 detector scoreable coverage 不是同一指标 |
| Final Accepted | 生成机制漏斗中最终成功接受的 keyed/watermarked rewrite 窗口总数 |
| Mean Latency | 从基础生成开始到最终程序交付的平均单样本延迟 |

## 4. 15 个实验的具体含义与结果

| # | 实验 | 相对 baseline 的唯一改动 | 实验具体含义 | Pass@1 | AUROC | TPR | actual FPR | Positive / Negative Coverage | Gate Coverage | Final Accepted | Mean Latency | 结果解释 |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | `baseline` | 无；使用 `d=12, delta=0, tau_suit=0.5, tau_close=0.5, n_min=2, max_units=3` | 提供所有 OFAT 变体的正式参考点，并验证修复后的默认机制是否恢复健康 | 0.804878 | 0.838415 | 0.243902（20/82） | 0.012195（1/82） | 91.46%（75/82） / 92.68%（76/82） | 97.02% | 123 | 12.080 s | 机制已从旧实验的近零 coverage 崩塌恢复；AUROC、coverage 和 Reliable Window 数均非退化 |
| 1 | `d-8` | `d: 12 → 8` | 使用更低维、更粗粒度的 LSH 区域编码，检验较少哈希维度能否保留可靠水印窗口 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 11.102 s | Gate 概率仍有正分离，但全部窗口在冻结阈值下不可用；按 ADR 0011 记录真实空背景，而非降低阈值 |
| 2 | `d-14` | `d: 12 → 14` | 使用更高维、更细粒度的 LSH 区域编码，检验更细哈希分区对覆盖率和检测能力的影响 | 0.804878 | 0.706945 | 0.085366（7/82） | 0.024390（2/82） | 90.24%（74/82） / 92.68%（76/82） | 96.12% | 77 | 11.883 s | 保持了高 coverage，但 TPR 和 AUROC 低于 baseline，说明更细分区没有带来更强的最终检测能力 |
| 3 | `delta-0p005` | `delta: 0 → 0.005` | 为 Reliable Window 增加很小的稳定性余量，检验轻微收紧可靠性条件的影响 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 11.391 s | 即使很小的正 `delta` 也使所有窗口无法通过冻结机制；这是该因素的真实敏感性结果 |
| 4 | `delta-0p01` | `delta: 0 → 0.01` | 进一步收紧 Reliable Window 的精度/稳定性余量 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 10.802 s | 与 `delta=0.005` 一致，冻结模型下没有窗口达到正式可靠性条件 |
| 5 | `delta-0p02` | `delta: 0 → 0.02` | 使用本矩阵中最严格的 `delta`，检验高稳定性余量对 Gate 标签和最终覆盖率的影响 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 11.218 s | Gate-data 仍有 4 个 suitable-positive 和 296 个 suitable-negative groups，且每个 split 均为两类；零 coverage 是显式 OFAT 效应，不是单类训练崩溃 |
| 6 | `tau-suit-0p3` | `tau_suit: 0.5 → 0.3` | 放宽 suitable head 的接受阈值，检验允许更多候选窗口进入水印机制后的效果 | 0.804878 | 0.913965 | 0.573171（47/82） | 0.024390（2/82） | 97.56%（80/82） / 97.56%（80/82） | 97.75% | 193 | 12.031 s | 本矩阵中 AUROC 和 TPR 最高、coverage 也最高；它是预先定义的消融观测，不据此反向改写 baseline 默认值 |
| 7 | `tau-suit-0p7` | `tau_suit: 0.5 → 0.7` | 收紧 suitable head 的接受阈值，检验高置信 suitable 条件是否仍能保留足够窗口 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 11.005 s | Gate 训练本身非退化，但没有窗口达到冻结的 `0.7` 正式阈值，形成合法空 calibration background |
| 8 | `tau-close-0p4` | `tau_close: 0.5 → 0.4` | 放宽 close/agreement head 阈值，检验较宽松的 close 判定对可靠窗口的影响 | 0.804878 | 0.746802 | 0.073171（6/82） | 0.012195（1/82） | 90.24%（74/82） / 92.68%（76/82） | 95.78% | 63 | 11.966 s | coverage 仍高，但最终检测能力低于 baseline |
| 9 | `tau-close-0p6` | `tau_close: 0.5 → 0.6` | 收紧 close/agreement head 阈值，检验更高 close 置信要求的影响 | 0.804878 | 0.746802 | 0.073171（6/82） | 0.012195（1/82） | 90.24%（74/82） / 92.68%（76/82） | 95.78% | 63 | 12.927 s | 与 `tau_close=0.4` 得到相同检测统计，说明该批样本的有效概率没有跨越 0.4/0.6 所形成的关键决策边界；两个 run 的 config、Gate Bundle 和 artifacts 仍各自独立 |
| 10 | `n-min-1` | `n_min: 2 → 1` | 仅需 1 个 Reliable Window 即允许 detector 评分，检验最低 evidence 要求对 coverage 与误报的影响 | 0.804878 | 0.849420 | 0.207317（17/82） | 0.012195（1/82） | 97.56%（80/82） / 100.00%（82/82） | 97.06% | 118 | 11.596 s | 得到最高的 negative coverage 和很高的 positive coverage；TPR 略低于 baseline，actual FPR 仍为 1/82 |
| 11 | `n-min-3` | `n_min: 2 → 3` | 要求至少 3 个 Reliable Windows，检验增加 evidence 数量对覆盖率和检测可靠性的影响 | 0.804878 | 0.851651 | 0.256098（21/82） | 0.012195（1/82） | 70.73%（58/82） / 70.73%（58/82） | 97.06% | 127 | 11.700 s | AUROC 和 TPR 略高于 baseline，但 scoreable coverage 明显下降；体现更严格 evidence 门槛的预期权衡 |
| 12 | `n-min-4` | `n_min: 2 → 4` | 要求至少 4 个 Reliable Windows，进一步检验严格 evidence 门槛 | 0.804878 | 0.835812 | 0.231707（19/82） | 0.012195（1/82） | 60.98%（50/82） / 62.20%（51/82） | 96.98% | 119 | 12.697 s | coverage 比 `n_min=3` 继续下降，符合 minimum Reliable Windows 增大时更多样本 abstain 的机制预期 |
| 13 | `max-units-1` | `max_units: 3 → 1` | 将 statement window 限制为单语句，检验没有跨语句上下文时 Gate 和检测机制能否工作 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 11.644 s | 单语句窗口下概率仍有正分离，但没有正式可用窗口；`comparable_group_count=0, decision_consistency=1.0` 按协议记录，不伪造跨 context 比较 |
| 14 | `max-units-2` | `max_units: 3 → 2` | 将 statement window 限制为最多两条语句，检验减少上下文长度的影响 | 0.804878 | 0.500000 | 0.000000（0/82） | 0.000000（0/82） | 0.00%（0/82） / 0.00%（0/82） | 0.00% | 0 | 11.320 s | 两语句上下文仍不足以在冻结阈值下形成 Reliable Windows，显示当前方法依赖 baseline 的三语句上下文 |

## 5. 结果总结

### 5.1 修复后的 baseline

修复后的 baseline 不再出现旧实验的检测覆盖崩塌：

- AUROC：`0.838415`；
- TPR：`20/82 = 0.243902`；
- actual FPR：`1/82 = 0.012195`；
- positive scoreable coverage：`75/82 = 91.46%`；
- negative scoreable coverage：`76/82 = 92.68%`；
- positive / negative mean Reliable Windows：
  `4.768293 / 4.780488`；
- Gate suitable probability separation：`0.065633`；
- generation 机制最终接受窗口数：`123`。

这与旧 baseline 的 TPR `0`、AUROC 约 `0.494`、正负 coverage 约
`0.61%/1.22%` 有本质区别，说明恒定学习率和 Gate 健康检查修复了真实的训练机制
退化。

### 5.2 参数敏感性

- `tau_suit` 是本矩阵中最敏感且方向最清楚的因素：
  `0.3` 显著增加机制覆盖和检测能力，`0.7` 则使覆盖降为零。
- 当前冻结模型对任意正 `delta` 都非常敏感：
  `0.005/0.01/0.02` 均产生零 coverage。
- `d=14` 保留了 coverage，但检测能力低于 baseline；
  `d=8` 则完全失去可评分窗口。
- `n_min` 表现出符合机制预期的 evidence/coverage 权衡：
  `n_min=1` coverage 最高，`n_min=3/4` 随门槛提高而 coverage 下降。
- `max_units=1/2` 均为零 coverage，表明该固定资源和 Gate 合同下需要
  baseline 的三语句上下文才能形成可靠窗口。
- `tau_close=0.4/0.6` 的最终检测统计相同，说明在本半数据量固定样本上没有产生
  可见的决策边界变化；这不表示两个实验被合并或复用了 artifact。

### 5.3 零 coverage 的解释

以下 7 个实验得到零 coverage：

```text
d-8
delta-0p005
delta-0p01
delta-0p02
tau-suit-0p7
max-units-1
max-units-2
```

这些结果没有通过调低阈值、改变 `n_min`、补样或重复运行进行“修饰”。对应 Gate
训练均满足：

- train/validation/test split 非单类；
- suitable probability 为有限值且正负均值存在正分离；
- encoder、Gate Bundle、配置和 run identity 正确绑定；
- calibration negatives 与 independent test negatives 严格独立；
- 空 pooled calibration background 符合 ADR 0011；
- detector 对全部记录保守返回 `insufficient_evidence`，p-value 为 null；
- Formal Eligibility、Metric Contract 和 artifact audit 均通过。

因此，零 coverage 是这些预先冻结 OFAT 因素在当前资源和合同下的真实方法局限，
不能作为继续调参直到结果“好看”的理由。

## 6. 可比性与局限

- 15 个实验共享相同的半数据量 ID、顺序、generation model、source resources、
  key identities 和单固定 seed；
- 每个实验独立训练 encoder 和 Gate Bundle，并完整运行八阶段；
- 所有选择标准在读取正式 independent test-negative 指标之前已经冻结；
- actual FPR 的最小非零步长为 `1/82 ≈ 0.012195`；
- 本实验为单 seed、HumanEval 前半 82 条，因此不能据此作多种子显著性或完整
  HumanEval 总体性能声称；
- `tau-suit-0p3` 虽然在本矩阵中结果最好，但它只是预先定义消融点的观测结果，
  不能据此将其事后选为新的默认参数；
- 不同实验的细小 latency 差异可能受单机瞬时负载影响，应作为描述性结果理解。
