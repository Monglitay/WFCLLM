---
status: accepted
---

# 历史内容只保留在 Git 历史与归档 tag

当前工作树只维护 Reproduction Core，不再保留退出该范围的 legacy/archive 代码、配置、脚本、文档、兼容 CLI、旧运行恢复逻辑或历史指标 reader；历史内容统一通过 Git 历史与既有 Archival Tag 恢复。这样牺牲了在当前 checkout 中直接浏览、运行或解析旧实现与旧产物的便利，但避免历史副本和兼容入口持续扩大维护面并混淆 Gate 复现边界。
