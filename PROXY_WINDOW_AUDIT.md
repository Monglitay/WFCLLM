# Dynamic Semantic V3：旧 Proxy 与 Complete-Final 路线审计

审计日期：2026-07-14（Asia/Shanghai）  
正式基线：`eac59bfbdc80a91bd5eb5d1332dcace75442d9ad`  
审计对象：历史 incremental proxy semantic route、被否决 complete-final-code semantic rejection-selection route。

## 1. 结论

旧 incremental proxy 不是 final code 的确定函数：mutable window membership 取决于 generation-time hit、window clear、rollback、当前可见 prefix、event ordinal 和 layer close；保存的 ledger/hash 只能证明部分文本存活，不能证明 generation 与 detector 编码了相同 context。它最多是 R1/R2 development evidence。

上一轮 complete-final semantic route 修复了大部分 objective mismatch，但正式 selection 逐一对 20 个完整 final code 调用 `score_code()`，每次重新 parse、extract、tokenize，并进行一个 candidate-level batch encoder forward，选中后再 replay。实测实现不是“每 block 一个 GPU forward”，而是“每 complete candidate 一个 GPU forward”；因此每题约 20+1 个 encoder forward、约 `20×B` 个 tokenized contexts。它作为 correctness/R3 prototype 有价值，但被用户否决为正式 generation 主循环。

新正式路线必须在 deterministic statement/block closure 时调用同一 context serializer 和 frozen encoder，跨 active trajectories 按 closure wave batch，commit 时增量累计 evidence；EOS 只读取累计 score；只对最终选中的 code 做一次独立 R3 replay。

## 2. Replay 等级

| Level | 输入 | 历史 incremental proxy | complete-final route | dynamic reset 要求 |
|---|---|---|---|---|
| R0 | artifact/schema 可读取 | pass | pass | pass |
| R1 | saved ledger/window/embedding 重算 | 部分 pass | selection-time score 可重算 | 仅诊断 |
| R2 | final code 但仍需要 prompt/ID/ordinal/ledger | pass | 旧 transport/ordinal 版本部分属于 R2 | 禁止 |
| R3 | final code + public config/artifacts + frozen encoder + secret | fail | scorer/core 已接近并形成可复用 prototype | 必须 100% |

R3 detector 的业务 payload 必须恰好为 `{ "final_code": string }`。Secret 通过 private process boundary提供；public artifacts不得包含 key fingerprint、private planes或targets。

## 3. 路线 A：历史 incremental proxy

### 3.1 Generation definition

历史 `PromptAwareBoundaryDetector`、state machine 与 semantic LSH rule 的行为是：

1. incremental parser event从当前 prefix构造 candidate；
2. per-layer mutable window保存一到两个 direct statements；
3. hit会清空 window，miss可能触发 rollback，full window只保留 suffix；
4. compound layer按 close时可见 text评分；
5. encoder输入是当前 candidate group的 join，不是 final-code serializer；
6. event ordinal、position/layer path和prompt-aware target function参与或影响状态。

Rejected candidate、rollback后消失prefix和未来suffix都会改变 subsequent parsing、parentage和window membership，因此 generation context无法从final code唯一恢复。

### 3.2 Detector definition

历史 detector从complete code重解析，选择目标函数，递归枚举owned statements，构造所有1/2-statement sliding windows和compound windows，再赋 detector-local ordinals并用calibration buckets聚合。这不是generation的hit-dependent window machine。

### 3.3 量化development evidence

- 旧 no4-long 中138/164样本有accepted sidecar hashes，138个至少一个hash在final detector proxy中出现。
- mean unique-hash/accepted-event ratio为`0.7725`，重复rule-decision candidate rate为`0.4398`。
- accepted-event count与final detector score Pearson `r=0.196`；recovered accepted-hash count与score `r=0.315`。
- 93个样本同时有accepted、hash match和sufficient detector evidence却仍低于threshold。
- 历史单任务cProfile记录1209次semantic rule evaluation/encoder path调用，encoder累计约10.84 s；这说明逐event串行GPU调用是明显工程瓶颈。

这些结果支持“recoverable text不等于replayable semantic context”，不支持放弃semantic encoder或改成syntax codebook。

### 3.4 Field-by-field mismatch

| Surface | Generation proxy | Final detector | 结果 |
|---|---|---|---|
| visibility | incremental prefix | complete final code | future suffix mismatch |
| boundary | parser event | completed AST owned statements | recovery/reparenting |
| window | mutable hit/rollback-dependent | exhaustive sliding | 不可重建 |
| rollback | 改变prefix与retry | 不可见 | hidden history |
| function | prompt-aware | final-code heuristic | target mismatch |
| identity | event position/ordinal | detector-local ordinal | shift/collision |
| tokenizer | dynamic contexts | reconstructed contexts | IDs未逐context闭合 |
| statistic | per-window accept | calibrated aggregate | objective不等价 |
| ledger |保存accepted windows |正式detector禁止 | 可能掩盖mismatch |

## 4. 路线 B：complete-final semantic rejection selection

### 4.1 Static code audit

被否决分支 `codex/watermark-mechanism-v3-semantic-lsh-reset` 的 `SemanticV3RetryAttemptSelector.select()` 对每个attempt执行：

```text
quality(final_code)
score_code(final_code)
```

`score_code()`每次执行：

```text
parse full final code
extract all semantic contexts
token-count units
batch all eligible contexts within this one candidate
frozen encoder forward
whitening/quantization
keyed LSH + Hamming target
aggregate score
```

完成20个candidate后才选出best，并对selected final code再次`score_code()`验证replay。优点是同一scorer使selection/replay易闭合；缺点是没有generation-time unit state，所有语义工作都在20份完整输出结束后发生。

### 4.2 真实成本 microprofile

只读profile使用该否决分支现有code、CodeT5/LoRA checkpoint、whitening、secret private file和V2既有retry final codes。选取连续5 tasks×20 retry=100 complete codes；模型加载与一次warmup排除；不生成新code、不打开新held-out。

| Metric | Measured |
|---|---:|
| complete candidates | 100 |
| distinct tasks | 5 |
| encoder calls（加一次replay） | 101 |
| encoded contexts（加一次replay） | 471 |
| total semantic scoring wall | 1.0071 s |
| median/candidate | 9.62 ms |
| p10/p90 candidate | 9.06 / 11.56 ms |
| mean units/candidate | 4.69 |
| measured peak allocated VRAM增量 | 511.5 MiB |
| estimated semantic wall/task | 0.2014 s |

重要限定：当前实现已经对“一个candidate内的blocks”做batch，所以不是`20×B`个GPU forwards；它仍是约20个顺序candidate forwards/task，并重复20次full parse/tokenization。上述profile只计semantic scoring，不含causal LM generation。

### 4.3 可复用与必须废弃

可复用correctness components：final-code context extractor的测试思想、frozen encoder hash closure、public whitening、deterministic quantization、HMAC-derived int8 hyperplanes、int64 dot product、Hamming target、exact rational ordering、strict one-field payload、batch-per-final detector、independent replay与file-access audit。

必须废弃的generation design：EOS后循环20 complete finals、per-final parse/tokenize/encode、以final replay反向参与20-way主搜索、依赖candidate index/retry ledger的target、公开key fingerprint。

## 5. 新 dynamic route 的成本模型

令`R=20` trajectories，`B_r`为trajectory r闭合units数，`W`为closure waves，`M`为encoder microbatch数。

- rejected route：parse约`R`次，tokenize约`sum B_r` contexts，encoder forward约`R`次，再加selected replay。
- dynamic batched route：incremental public boundary updates；context serializer只在closure时运行；tokenize仍约`sum B_r`，encoder forward约`M<=W`（同wave可合并）；EOS encoder forward为0，再加selected R3 replay一次。
- detector：parse一次、收集B contexts、encoder forward为`ceil(B/batch_size)`，禁止per-block forward。

成本硬门冻结为相对rejected route的median semantic wall/task至少下降30%，即在同机同artifact/debug profile中目标低于约`0.1410 s/task`。若debug profile在查看pilot outcome前证明该基准统计不稳定，只允许一次书面修订；之后不得改。

## 6. Counterfactual R3 contract

1. 空临时目录只放final code与显式public artifacts。
2. 新进程只接受single-field record或direct string API。
3. 不复制prompt、ID、ledger、candidate、saved score/embedding/context。
4. 比较unit IDs、context hashes/text、token IDs/masks、embedding tolerance、quantized vectors、LSH symbols、score/p-value/decision。
5. file-open allowlist阻止experiment/ledger/sidecar访问。
6. CPU/GPU integer symbols、statistic与decision相同。
7. 删除rejected/rollback blocks后generation accumulated evidence与final replay一致。
8. final detector把全部contexts合成batch/microbatch。

R3低于100%、需要hidden input、或encoder bypass不改变evidence时，禁止pilot。

## 7. 审计结论

历史proxy失败是context/identity/objective mismatch；complete-final route是正确性原型但违反本轮成本与dynamic-generation主路径。正式修复不是threshold tuning，也不是syntax codebook，而是统一serializer、bounded causal context、closure-wave encoder batching、commit-time exact accumulation与single selected-final R3 replay。
