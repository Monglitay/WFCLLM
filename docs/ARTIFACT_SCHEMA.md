# Current Artifact Schema

只支持 Fresh Reproduction Run 的当前布局：

```text
<experiment-root>/
├── pilot/                       # 独立 pilot encoder + gate-data
├── pilot-private/               # pilot keys（本地私有）
├── private/                     # full keys（本地私有）
├── gate-cache/                  # 本地 cache
├── pilot_state.json
├── run_state.json
└── run/
    ├── encoder/
    ├── gate-data/
    ├── gate-train/
    │   ├── candidate_bundle/
    │   └── candidate_bundle_manifest.json
    ├── inputs/final_code.jsonl
    ├── generation/
    ├── calibration/
    ├── detection/
    ├── reports/
    └── audit/
```

## 关键 artifact

| 路径 | 生产阶段 | 消费阶段 |
|---|---|---|
| `encoder/best_model.pt` | encoder | gate-data、gate-train、generate、detect |
| `gate-data/manifest.json` | gate-data | gate-train、Metric Contract、audit |
| `gate-data/window_groups.jsonl` | gate-data | gate-train、audit |
| `gate-data/feasibility_summary.json` | gate-data | gate-train、audit |
| `gate-train/candidate_bundle_manifest.json` | gate-train | generate、calibrate、detect、audit |
| `gate-train/candidate_bundle/` | gate-train | generate、calibrate、detect |
| `inputs/final_code.jsonl` | generate | detect、posthoc evaluation、audit |
| `generation/audit.jsonl` | generate | report、audit；绝不是 detector input |
| `generation/finalizer.jsonl` | generate（启用 finalizer 时） | audit |
| `generation/raw_program_outputs.jsonl` | generate（无 finalizer 时） | audit、生成后配对正确性控制；绝不是 detector input |
| `generation/progress.json` | generate | audit |
| `calibration/negative_corpus.jsonl` | calibrate | calibrate、audit |
| `calibration/reference_calibration.json` | calibrate | detect、report、Metric Contract |
| `detection/positive_details.jsonl` | detect | report、Metric Contract |
| `reports/reference_report.json` | report | Metric Contract、audit |
| `reports/pass_report_posthoc.json` | posthoc | Metric Contract |
| `reports/metric_contract.jsonl` | audit 后统一汇总 | 当前 Full Reproduction 结果 |
| `audit/*.json` | audit | 独立完整性检查 |

## 身份与绑定

内部 `RewriteRequest.completed_suffix` 保存 C++ 窗口之后未改动的源码，
只供候选语法解析恢复完整作用域。HF 重写模型调用不接收此字段，候选单元只从
替换区间提取；detector input 仍严格遵循四字段合同。

当前 artifact 使用当前 schema version，并以 SHA-256 绑定配置、source catalog、
key identifier、Gate Bundle、tokenizer、semantic encoder、window contract、
negative corpus manifest、generation config 与内容绑定的 generation model
identity。私有 key bytes 不进入公开 artifact。

C++ 固定轨迹使用 `cpp-structural-local-alpha/v3:<candidate_index>` 标识改写实现。
前 12 次表达式尝试保留；尾部可提供固定组合或仅作用于窗口内绑定的局部改名候选。
整条轨迹仍为 18 项，semantic validation 继续由原有运行时守卫完成。

JavaScript 固定表达式轨迹使用 `js-structural-expressions/v1:<candidate_index>`。
它不发起窗口模型调用，不读取 key 或评分，也不为候选颁发语义证书。正则、模板、
多行和存在分隔歧义的表达式保持不可用；原程序及所有失败候选仍保留。

Gate cache 的 `semantic-input-spans-<scale>-<config_hash>.jsonl` 记录每个结构合格候选的
`semantic_span_in_candidate`、原文 hash、区间文本 hash 与规范化编码器输入 hash。
候选原文仍在候选记录中；原文区间之外的注释不进入语义探针。
`selection_summary.semantic_input_span_audit` 记录该文件名、hash 和候选数。
这些记录带 `audit_only=true`、`not_detector_input=true`，不进入 detector。

Java 的 `java_structural_equivalent` 使用固定 18 次表达式轨迹；
`rewrite_config_id` 为 `java-structural-expressions/v1:<index>`，不添加注释或
语义证书。同一段候选原文的真实解析区间用于 Gate 数据探针和在线评分。

`generation/manifest.json` 与 run state 同时记录 `final_code_sha256` 和
`final_code_row_count`；calibrate、detect、report、audit 逐阶段复核。
calibration、positive detection details、reference report 也由 run state
记录 SHA-256。`audit/artifact_integrity.json` 汇总这些核心 hash，Metric
Contract 在读取 report 前再次验证，防止 audit 后篡改。

`candidate_bundle_manifest.json` 必须来自同一 run 的 `gate-train`，携带
`candidate_bundle_sha256` 和当前 experiment contract hash。后续阶段重新计算
树 hash；不接受外部路径或旧 candidate layout。

Gate 训练内部 checkpoint、训练指标与开发摘要使用 v3 合同。
`best_objective` 为 `[close_loss, suitable_false_positive_rate]`，按字典序最小化；
边界验证损失同时驱动检查点选择和早停。Suitable 损失及危险负样本惩罚仍参与
训练。当前配置的 Suitable 阈值上限为 0，因此不以分类头在 0.5 阈值下的
零误报优先于边界预测质量。这不改变最终 detector 的分数、阈值或指标口径。

生成器每次提交后从当前代码重新分窗。候选须在已提交前文、候选及后文组成的
完整代码中恢复为同一语句窗口，并满足当前 Gate 与 parent descriptor 合同。
候选评分仍使用该窗口的 UTF-8 原文字节区间。窗口 audit 的 `original_span`
和 `rollback_anchor` 以当前步骤执行前的代码为坐标，`selected_span` 以该步
提交后的代码为坐标；后续提交只修改尚未提交的后文。

MBPP finalizer 的输出必须先与可信 wrapper 的预期结果完全一致。对于完全
未改变的输出，字节一致本身构成来源证明，即使其语法错误、无法统计 AST
语句，也须保留为原样失败样本。此时 AST 计数字段可为 `null`，新增语句数为
0；语法及正确性标准保持不变，所有任务仍进入完整评测分母。

MBPP completion 模式保留完整的 prompt 与模型续写，交给可信 finalizer；
后续 Markdown 围栏不得覆盖已经生成的函数体。聊天模式规范化公开形参时，
通过同时赋值保留生成函数体所用的原局部名称，保持嵌套作用域不变。固定位置
形参可绑定公开接口的位置前缀，其余公开位置参数不注入模型生成的计算。
需要猜测缺失参数值、含可变参数或递归调用参数个数不匹配时保留原代码，
继续计入后验评测。新增的接口绑定语句不属于水印改写，正负两侧处理一致。

MBPP 目标之后的声明按静态全局引用及其依赖链保留，包括必要的辅助函数、类、
导入与赋值声明。词法作用域和延迟注解用于排除局部绑定及无须执行的类型引用。
此过程不执行生成代码或读取标准断言；尾部无关定义、示例调用和断言不引入输出。
动态名称查找及顶层控制流不作依赖推断。正负两侧必须使用相同的 finalizer 实现。

默认校准按负样本 `hit_rate` 分位数确定阈值。可选校准策略包含负样本经验分布的
standardized hit surplus：背景命中概率从全部
负样本通道计数估计，分数按程序证据量标准化；阈值和 right-tail-plus-one
检验均保持目标 FPR=0.05，无证据程序保留且不判阳性。这只调整检测决策的
校准策略。Metric Contract 的 AUROC 仍使用全部程序的原始 `hit_rate`，
不得以标准化分数的 AUROC 替代统一比较口径。

## Test-only artifact

Java/JavaScript 的生成后 correctness 行记录 `evaluation_protocol`、
`original_test_sha256`、`check_program_sha256`、真实 runtime/compiler 路径和
`timeout_seconds`，以绑定原始测试及实际执行程序。编译错误、运行错误与超时
均输出 `is_correct=false` 和错误信息；这些字段只用于后评测，不能进入
`inputs/final_code.jsonl` 或 detector。Java harness 的空 `Main` 占位清理及
JavaScript 对输出流的失败判定不改变任何原始测试或生成代码产物。

`humanevalpack_chat` 的 `generation/raw_program_outputs.jsonl` 额外保留
`model_generation`，字段为 `sample_id`、`prompt_mode`、`model_input`、
`model_response`、`model_input_token_ids`、`model_continuation_token_ids`。
`output_source` 是响应拆出代码围栏后的本轮原始程序，原响应仍完整保留，
失败代码同样保留。此 trace 只属于 generation sidecar，不能进入 detector。

受控测试可注入轻量依赖。只要测试产生 Gate artifact，就必须携带一致的
`diagnostic_test_backend=true`、`formal_eligible=false`、`diagnostic_only=true`
和 `not_official_method=true` 身份。生产 preflight 不接受这些 artifact。

## Metric Contract

`scripts/wfcllm_metric_contract.py` 只读取上述当前布局，输出
`wfcllm-metric-contract/v3`。每个已 audit 的有效当前 run 始终得到一行；
样本不足和缺少 posthoc Pass@1 以 `null` 与 caveat 表达，而不是省略整行。
其他 Pass@k 可生成当前 posthoc 报告，但不进入 Metric Contract。

AUROC 的正负程序分数均为 `hit_rate`。负样本必须读取已审计且绑定 SHA-256
的 `detection/negative_details.jsonl`，包括无证据程序；不得以校准拟合桶代替
完整负样本记录。行内同时记录 `negative_sample_count`、
`negative_insufficient_evidence_count` 与 `auroc_score_field`，使分母可以复核。
必要负样本 artifact 缺失、无效或被修改时失败。
