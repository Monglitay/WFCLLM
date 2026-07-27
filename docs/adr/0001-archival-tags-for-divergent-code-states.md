---
status: accepted
---

# Divergent Code State 由既有 Archival Tag 固定

历史实验曾来自多个带未提交改动的 Code State，单个 commit 不能完整表达它们。
既有 Archival Tag 固定这些磁盘状态，供需要时只读恢复。

当前开发不从 Archival Tag 起步，也不把其中内容复制回工作树。Reproduction
Core 只维护当前 Gate 实现；历史恢复遵循 ADR 0009。
