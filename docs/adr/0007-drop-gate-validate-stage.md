---
status: accepted
---

# 去掉 gate-validate 阶段

手稿的方法里没有独立的门控准入阶段。它只有两处阈值选择：门控训练在验证集上选 closure 阈值、suitability 阈值、语义 margin 阈值与最小载体数；检测阈值在留出的未加水印生成上按 5% 目标假阳率校准。前者属于 `gate-train` 内部，后者就是 `calibrate`。手稿中不存在 float 与量化版本的一致性检查、[[Suitable False Positive Rate]] 的准入上限、gate 数据的 feasibility 准入判定，也不存在把 bundle 分为正式合格与仅供诊断两类的发布闸门。因此我们去掉 `gate-validate`，实验链为七阶段：`encoder`、`gate-data`、`gate-train`、`generate`、`calibrate`、`detect`、`report`、`audit`。

## Consequences

- [[Formal Eligible]] 这个概念失去产生者。ADR 0005 里记录的「多语言 Reproduction Target 均为 [[Diagnostic Only]]」是这道被移除闸门的产物 —— 按手稿口径，那个资格问题不成立。[[Metric Contract]] 仍保留资格字段，用于如实转述历史产物里已有的标记，但新产生的 run 不再有该标记。
- 历史 run 状态里的「门控验证已跳过」「未验证的候选 bundle」两个标记随之失去意义；相关的运行时分支需要相应收敛，不能留下一条期待「已验证 bundle」而永远得不到的代码路径。
- 移除准入上限意味着不再有任何机制阻止一个判别力很差的门控进入生成与检测。`attempt14` 的门控 Suitable False Positive Rate 为 0.9886 而原准入上限是 0.05 —— 去掉闸门后，这类情况只能靠 [[Metric Contract]] 事后暴露，不会被事前拦住。这是刻意接受的取舍：手稿没有这道闸门，实现不应擅自增加一道手稿无法解释的门槛。
- 仓库既有规则中「fake backend 绝不可发布正式已验证 bundle」失去载体，须确认移除后没有让受控测试与诊断的隔离要求落空。
