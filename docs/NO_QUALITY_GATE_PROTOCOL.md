# No-quality-gate Protocol

当前方法的生成决策只能读取：

- public prompt 与已生成前缀；
- statement-window 结构；
- Gate close/suitable 输出；
- key-blind rewrite candidate；
- semantic preservation、LSH signature/margin 与 Deployment Key 派生目标。

以下信号不得参与 generation、retry、candidate selection、calibration 或
detection：

- 单元测试或 hidden tests；
- pass/fail、Pass@1、correctness；
- canonical/reference solution；
- 编译或执行是否成功；
- 与答案的相似度、长度差或质量分数；
- 人工选择标签。

`inputs/final_code.jsonl` 在 generation 阶段完成后冻结，其 SHA-256 与行数
由 generation manifest 和 run state 双重绑定，并由所有下游阶段复核。
Pass@1 只能由 `scripts/evaluate.py` 生成 posthoc report，且必须同时携带：

```text
posthoc_only=true
not_used_for_generation=true
not_used_for_retry=true
not_used_for_selection=true
not_used_for_calibration=true
not_used_for_detection=true
```

校准补充的未加水印生成“生成即采用”，不因测试、正确性或质量信号筛选。
`audit` 必须验证严格 detector input 与这些 posthoc 标记；发现质量字段泄漏时
运行失败。
