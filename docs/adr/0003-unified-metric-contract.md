---
status: accepted
---

# 统一 Metric Contract，报告自证数据集与模型

历史上每种语言各有一套一次性指标脚本（C++ 在 `cpp_score_auroc_sweep.json`、Java 在 `positive_filter_auroc_scan_summary.json`、JS 在 `js_goal_result_summary.json`、Python 在 `/tmp` 的散装脚本里），字段名与算法各不相同，且其中数百行只存在于 run 目录或 `/tmp`、不在任何仓库副本里。更糟的是 `reference_report.json` 不记录 dataset / language / model —— 一份报告无法自证它属于哪个数据集、哪个模型，这个绑定只活在目录名和启动脚本里。我们把这些脚本收进主线、去掉硬编码，合并为单一 Metric Contract：输入一个 run 目录、输出一行含数据集、语言、模型、`target_fpr`、TPR、pass、AUROC、Key Identifier 与结果资格的标准化记录，并把 dataset / language / model 写回 `reference_report.json`。

## Consequences

- 三种语言的 AUROC 原本算法不同，统一必须选定一个定义，因此部分历史 AUROC 数字不会完全吻合。
- 原 JS 报告脚本结尾有 `if not summary["tpr"] > 0.22 or not summary["pass_rate"] > 0.56: raise` —— 达标才写文件。这是选择性报告，且与仓库禁止在官方路径上用 pass/正确性代理做门控的规定相冲突；Metric Contract 无条件输出，达标与否只作为字段记录。
- 历史 Reproduction Target 之间口径本就不一致（JS 用 `target_fpr=0.31` / `minimum_reliable_windows=1`，其余用 `0.05` / `2`；Python 那行报的是 Conditional TPR）。Metric Contract 把口径显式记进每一行，使不可横比的行不再被误读为可横比。
