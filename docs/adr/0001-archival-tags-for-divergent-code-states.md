---
status: accepted
---

# 多个 Code State 用同基点并列 Archival Tag 表达

服务器 A 上产出实验结果的代码分散在四个互不包含的 Code State（`WFCLLM-nocarrier-gated-semantic` 工作区、其 `codex/opencoder-wfcllm-repair-20260723` worktree、无 git 的 `WFCLLM-cpp-basic-run`、以及 `WFCLLM-gated`），它们共享 commit `0d29949` 或其祖先，却各自带着大量未提交改动 —— 因此「一个 commit = 跑出这批结果的代码」在事实上不成立。我们把 `0d29949` 取回作为共同基点，在其上并列若干不可变 Archival Tag，每个 tag 固定一个 Code State 的原样内容（含 `nocarrier` 工作区里 66 个 `.before-*` 手工快照）。

## Considered Options

- **用 66 个 `.before-*` 文件线性重建真实时间线**：最忠实，且能回答「某个 MBPP 变体用的哪版代码」。但 `.before-*` 只是单文件快照而非完整树，无法保证任何中间态可运行；且丢弃全部实验产物后，快照名与 run 目录名的对照价值大幅下降。
- **只留一个收敛提交，另两态存为 patch**：仓库最干净，但 `cpp-basic-run` 与 `nocarrier` 相差 73 处且独有 `wfcllm/ablation`、`wfcllm/extract`、`wfcllm/common/transform` 三个目录，patch 既大又难干净 apply。

## Consequences

Archival Tag 是脏的 —— 它们忠实记录当时磁盘上的样子，包括手工快照文件。这是刻意的：主线保持干净，归档保持真实。未来开发不应从 Archival Tag 起步。
