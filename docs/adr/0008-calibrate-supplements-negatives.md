---
status: accepted
---

# calibrate 补足 held-out negatives

`calibrate` 必须收到严格四字段的 held-out 未加水印 `NEGATIVE_INPUT`。若其
条数少于 `calibration.target_negative_count`，阶段使用同一 generation
model 在未被 positive evaluation 使用的提示上生成差额。

补充生成不经过 Gate embedding 或窗口改写，不读取 Deployment Key，且生成即
采用；pass、测试、正确性和质量代理都不得筛选样本。当前 calibration 产物记录
外部/生成条数、模型标识、解码配置和完整 negative corpus SHA-256。缺少输入
文件、可用 held-out 提示或生成模型时明确失败。
