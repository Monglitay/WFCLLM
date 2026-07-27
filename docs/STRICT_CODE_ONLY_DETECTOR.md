# Strict Code-only Detector Contract

正式 positive detector input 只有：

```text
<run>/inputs/final_code.jsonl
```

每一行必须是 JSON object，并且 key 集合必须精确等于：

```text
id, dataset, prompt, final_code
```

四个值都是字符串，`id` 非空。缺字段、额外字段、重复 JSON key、非有限数、
无法解码的 UTF-8、超限文件/行或 symlink 输入都会失败。

禁止作为 detector input 的内容包括：

- generation audit；
- candidate sidecar 与 raw attempt summary；
- Gate windows、labels、key-bank manifest 或训练指标；
- pass/test/correctness/quality 字段；
- canonical solution、reference answer 或执行结果。

检测器从 `final_code` 重新解析 statement windows，使用当前 run 的 Gate Bundle、
semantic encoder、Deployment Key identifier 和 calibration artifact。校准与检测
必须绑定同一 Bundle、窗口合同、tokenizer、semantic encoder 和 key identifier。
