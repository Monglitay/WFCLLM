---
status: accepted
---

# 收敛结果落在 main，整理工作走独立分支

三份散落副本（本地、服务器 A、服务器 B）要收敛成唯一真相之源，而这个源应当是 `main`。整理工作在从 `main` 拉出的独立分支上进行，完工后正常 merge 回 `main`，不改写 `origin/main` 历史、不 squash 中间步骤。

关键前提是一个容易误判的事实：Gated Lineage 的基点 `0d29949` 与 `main`（`9ebd4b6`）共享祖先 `6e94ab2`，`main` 仅比该祖先领先 2 个提交、落后 0 个 —— Gated Lineage 是 `main` 的 1-commit 兄弟，不是异种分叉。之前看起来像分叉，是因为服务器 A 上 `WFCLLM-nocarrier-gated-semantic` 仓库自己的 `main` 走到了 `0d29949`，而 `origin/main` 走到了 `9ebd4b6` —— 两个同名 `main` 指向不同历史。因此让 `main` 成为收敛版本不需要任何历史手术。

## Considered Options

- **从 `0d29949` 起分支、最后把 `main` 指过去**：整理后的代码直接坐在实验代码态之上，但会丢掉 `main` 那 2 个提交并需要 force-push 改写 `origin/main`。
- **最后 squash 成单个提交并入 `main`**：`main` 历史最干净，但「先收了什么、后重构了什么」在 `main` 上不可见，日后出问题难以二分定位。

## Consequences

[[Archival Tag]] 挂在 `0d29949` 之上，与 `main` 并列，不进入 `main` 的历史 —— `main` 承载整理后的代码，Archival Tag 承载历史真相，两者职责分离。
