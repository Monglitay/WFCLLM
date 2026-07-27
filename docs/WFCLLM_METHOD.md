# WFCLLM Gate 方法

当前唯一方法是 `gated_semantic_window_v1`。

1. `encoder` 使用目标语言/数据集对应的 Gate source catalog 训练 semantic
   projection。Python 使用 AST 正变换，C++、Java 使用公开语句等价变体，
   JavaScript 使用无需额外 tree-sitter wheel 的词法 trivia 等价变体。
2. `gate-data` 用独立的 pilot/full source catalog、Training Key Bank 和
   Holdout Key Bank 构造 W1/W2/W3 窗口、候选改写和标签；Python 使用 AST
   改写，其余三种语言使用运行时 rewrite model。
3. `gate-train` 训练 close/suitable Gate，并在当前 run 中发布带 hash 绑定的
   Gate Bundle。不存在独立 `gate-validate` 阶段。
4. `generate` 由 Gate 决定窗口闭合与 suitability。候选 0 永远是原窗口；
   最多生成 3 个 key-blind 改写，只有结构、语义稳定和目标 LSH 命中全部满足
   时才选择改写，否则恢复原窗口。
5. `calibrate` 在 held-out 未加水印生成上按目标 FPR 标定阈值。外部负样本
   不足时，用同一 generation model 在与正样本不重叠的提示上补齐。
6. `detect` 只读取严格四字段 `final_code.jsonl`，重新提取可靠窗口并计算
   `reliable_window_hit_rate`；可靠窗口不足时给出
   `insufficient_evidence`。
7. `report` 与 `audit` 记录当前 schema 指标、四字段完整性、秘密隔离和
   no-quality-gate 证据。

固定参数合同包括：LSH `d=12`、`gamma=0.45`、最大窗口 3 个 statement
unit、最大 rewrite 3 次、Gate input 上限 256 tokens。不同语言只改变窗口
合同和改写策略，不复制模型配置。

Pass@1 是唯一的 Reproduction Evaluation，只在生成后计算。它以及测试结果、
正确性代理都不得参与窗口判定、候选生成、重试、选择、校准或检测。
