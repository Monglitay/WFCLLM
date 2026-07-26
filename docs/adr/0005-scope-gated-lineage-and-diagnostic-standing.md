---
status: accepted
---

# 本轮范围锁定 Gated Lineage，并记录多语言结果的 Diagnostic 资格

「让服务器 A 跑过的实验可复现」这一目标，其实际落点是 Gated Lineage 而非 `codex/server-sync-js`。后者的 `wfcllm/lang/cpp/__init__.py` 与 `java/__init__.py` 各只有一行占位 docstring，`wfcllm/datasets/humanevalpack.py` 的 `iter_samples` 直接抛 `NotImplementedError` —— 这份代码物理上跑不出 C++/Java/JS。HumanEval 多语言、MBPP 与换模型对比全部出自 Gated Lineage。因此本轮只收敛 Gated Lineage，Semantic-V3 Lineage 的 v3 语义实验推到后续轮次。

## Consequences

被指定的 Reproduction Target 中，多语言那几条按流水线自己的记账都是 [[Diagnostic Only]]：C++ 全部 arm、Java `full164` 与 `smoke16`、JS `full164` 的 gate 数据 manifest 均为 `formal_eligible: false` / `diagnostic_only: true` / `experimental_only: true`。JS 那条还带 [[Experimental Bypass]]（`target_fpr` 放宽到 0.31、`minimum_reliable_windows` 降到 1，理由是 gate 数据的 Suitable Window 正样本数为 0，而准入门槛要求至少 30）。

这不影响「可复现」这一目标，但决定了这批结果在论文中能承担的角色 —— 复现它们等于复现一批 Diagnostic Only 结果，不能当作方法结论的证据。相关运行还存在若干需要单独查证的信号：JS 那条 164 个样本中有 84 个是事后用另一模型重生成的，且 `selected_rewrite_count` 为 0；`attempt14` 的 Gate 的 [[Suitable False Positive Rate]] 为 0.9886，而准入上限是 0.05，其 `validated: true` 仅表示混淆矩阵非退化、并不检查准入阈值。
