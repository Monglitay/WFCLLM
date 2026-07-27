---
status: accepted
---

# Gate Bundle 由每个 Fresh Reproduction Run 重训

Gate Bundle 是代码、当前配置、Gate 数据、基座模型、key banks 与随机种子的
产物，不是仓库分发的输入。模型权重和历史 Bundle 不进入 Git。

每个 Full Reproduction Profile 都从 `encoder`、`gate-data`、`gate-train`
开始，并只消费同一 run 发布且 hash 绑定的当前 Bundle。外部历史 Bundle
不能缩短阶段链。相同种子不保证不同 CUDA kernel 的权重逐字节相同，因此复现
比较以合同和指标为准，不以历史 Bundle hash 相等为准。
