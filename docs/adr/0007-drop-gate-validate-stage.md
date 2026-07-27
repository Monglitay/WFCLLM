---
status: accepted
---

# 不设独立 gate-validate 阶段

Gate 的训练期 validation 属于 `gate-train`，检测阈值选择属于 `calibrate`。
当前公开阶段链因此恰好是：

```text
encoder → gate-data → gate-train → generate → calibrate → detect → report → audit
```

`gate-train` 直接发布当前 run 的 candidate Bundle，后续阶段以配置 hash 和
Bundle tree hash 绑定它。不存在 gate-validation publication、skip marker、
validated-external-Bundle 分支或相应 CLI/state row。受控测试后端仍必须带
完整非正式身份，且不能通过生产 preflight。
