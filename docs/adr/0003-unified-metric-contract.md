---
status: accepted
---

# 当前 run 使用统一 Metric Contract

`scripts/wfcllm_metric_contract.py` 只读取当前 run layout，并为每个有效 run
输出一行 `wfcllm-metric-contract/v2`。行内自证 dataset、language、
generation model、Key Identifier、semantic encoder、Gate Bundle、阈值、
全样本 TPR、AUROC 与 posthoc Pass@1。

Metric Contract 无条件输出：指标未得到或样本不足时记录 `null` 和 caveat，
不能因结果未达标而不写。旧报告布局和一次性历史指标脚本不由当前 reader
兼容。
