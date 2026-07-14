# Watermark Mechanism V3：第二次动态语义 Reset 诊断

## 1. 决策

从正式V2 commit `eac59bfbdc80a91bd5eb5d1332dcace75442d9ad`重建独立分支。停止但保留：

- `codex/watermark-mechanism-v3-literature-pivot`：explicit syntax/codebook V3；
- `codex/watermark-mechanism-v3-semantic-lsh-reset`：complete-final-code semantic rejection-selection V3。

正式主机制为 **Streaming-SharedPool Dynamic Semantic V3**：20条raw trajectories在生成到deterministic statement/block boundary时，对current canonical block加至多一个preceding committed block构成bounded causal context；同wave contexts合并进frozen CodeT5-family encoder batch；public whitening/int quantization后做secret-keyed random-hyperplane LSH与content-ID target；trajectory在commit时累计exact signed evidence；EOS在quality hard gate后用累计correct-key evidence排名。Detector只从final code重建contexts并batch replay。

## 2. 与旧要求冲突及处理

上一轮文档把“每个完整candidate运行同一final detector再做rejection selection”写成正式修复。用户本轮最高优先级明确否决该路线；因此旧结论只保留为cost/correctness prototype，不进入新branch ancestry或主实现。

`brainstorming`流程通常要求设计后等待人工批准；用户同时明确要求不要停留在计划阶段、不要询问可从文献/仓库/实验确定的问题，并指定Candidate A/方案1为默认推荐。故本轮把该直接指令视为对Streaming-SharedPool主臂的授权，设计与权衡写入preregistration后自主继续。这一流程冲突不改变科学硬门。

## 3. 冻结历史事实

| Arm | Pass@1 | positive detection | held-out FPR | AUROC | sufficient |
|---|---:|---:|---:|---:|---:|
| Current-R20 | 84/164 | 42/164 | 4/41 = 9.76% | ~0.6226 | 135/164 |
| V2-R20 | 82/164 | 38/164 | 3/41 = 7.32% | ~0.7313 | 163/164 |

这些panel已打开，只是development evidence。V2提高coverage/AUROC但未提高冻结阈值detection，且FPR仍>5%。新V3必须重新冻结calibration与fresh held-out negatives。

## 4. 为什么不放弃 semantic encoder

V1的accepted snippets大量仍存在于final code，但generation mutable window与detector exhaustive window不等价，score correlation很弱。V2/complete-final prototype证明相同scorer可以exact replay。证据指向context observability和objective isomorphism，而不是“encoder没有信息”。

正式symbol必须满足因果链：

```text
final-observable canonical block context
-> frozen semantic encoder
-> public whitening/L2
-> deterministic int16 quantization
-> secret-derived int8 hyperplanes/int64 dot
-> observed semantic bits
-> content-ID key target/Hamming evidence
```

Constant/bypass encoder如果仍产生相同decision，则机制失败。AST role只能进入context identity/serializer，不直接计hit。

## 5. 为什么bounded causal K=1

- generation在block closure时可见；
- detector从final code唯一重建；
- 不需要future suffix、prompt、task ID、ledger或hidden state；
- 每context token cost有上界，整体近似线性；
- closure-wave可以batch；
- 删除一个block最多影响被删unit与紧随其后的一个context；
- 相比whole-prefix避免`O(B²)` token growth和全局永久错位。

超过unit token budget不静默截断：该unit成为erasure。Context无法唯一恢复也成为erasure。

## 6. Primary arm 与公平性

正式只报告两个paired arms：

1. Current-R20；
2. Streaming-SharedPool-V3-R20。

同task的20 raw trajectories及其hash必须完全相同。V3只在生成过程中增量计算每条trajectory的semantic evidence，并在EOS按冻结quality-first规则从同pool选择。由于没有中途key-conditioned prefix分叉，exact same-pool声明成立。

Block rejection/rollback和semantic beam会让prefix在第一处key decision后分叉，只能在未来作为matched-budget arm报告。本轮若主臂失败，不得偷偷切换而继续使用同一preregistration。

## 7. 主要风险与观测量

1. **Region collapse**：20条高质量code trajectory可能落入相同semantic buckets。测signature entropy、pairwise cosine、best-of-20 attainable gain。
2. **Surface sensitivity**：format/identifier可能主导embedding。测format、rename、AST-unparse稳定性，不能把危险变换当等价真值。
3. **Clean occupancy bias**：用calibration、fresh held-out、wrong-key、per-bit/bucket occupancy与outlier analysis。
4. **Short-code capacity**：报告sufficient blocks、erasures、no embedding；ECC不能制造容量。
5. **Quality conflict**：固定lexicographic quality gate；无安全candidate时选择quality最优并标记erasure/no_embedding。
6. **Checkpoint leakage**：记录HumanEval/MBPP训练provenance；不使用hidden tests调policy/encoder。
7. **Cost**：rejected route实测约0.2014 semantic seconds/task，目标至少下降30%；generation无EOS全量20-way rescore。
8. **Rollback contamination**：正式主臂不做semantic rollback；state machine仍需测试rejected/rolled-back evidence removal，为未来arm与contract防护。

## 8. 停止条件

以下任一出现即候选失败并闭合负结果：encoder不是正式evidence；secret不改变generation ranking；R3<100%；需要prompt/ID/ledger/sidecar；context/token mismatch；rollback污染；wrong-key异常；fresh held-out FPR>5%；pilot Pass<15/30或比Current低>1；TPR不高于Current；只靠outlier；schema/secret泄漏；成本下降<30%；或实现仍在EOS对20 full finals全量重评分。

失败不授权syntax codebook、threshold leakage、换split/seed/encoder/LSH/statistic或伪称same-pool。只允许一次冻结范围内单组件修复；仍失败则不运行full 164。

## 9. 成功声明边界

只有contract、R3、安全、schema、profile与30-task pilot全门通过才运行HumanEval 164。Full还需Pass≥82/164、actual held-out FPR≤5%、V3 TPR>Current、R3=100%、correct-key非outlier正移、wrong-key/clean合法、artifact/hash闭合、secret/file-access audit通过。否则结果明确标记experimental negative，旧默认配置不变。
