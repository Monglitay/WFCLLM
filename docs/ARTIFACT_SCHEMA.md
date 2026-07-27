# Current Artifact Schema

只支持 Fresh Reproduction Run 的当前布局：

```text
<experiment-root>/
├── pilot/                       # 独立 pilot encoder + gate-data
├── pilot-private/               # pilot keys（本地私有）
├── private/                     # full keys（本地私有）
├── gate-cache/                  # 本地 cache
├── pilot_state.json
├── run_state.json
└── run/
    ├── encoder/
    ├── gate-data/
    ├── gate-train/
    │   ├── candidate_bundle/
    │   └── candidate_bundle_manifest.json
    ├── inputs/final_code.jsonl
    ├── generation/
    ├── calibration/
    ├── detection/
    ├── reports/
    └── audit/
```

## 关键 artifact

| 路径 | 生产阶段 | 消费阶段 |
|---|---|---|
| `encoder/best_model.pt` | encoder | gate-data、gate-train、generate、detect |
| `gate-data/manifest.json` | gate-data | gate-train、Metric Contract、audit |
| `gate-data/window_groups.jsonl` | gate-data | gate-train、audit |
| `gate-data/feasibility_summary.json` | gate-data | gate-train、audit |
| `gate-train/candidate_bundle_manifest.json` | gate-train | generate、calibrate、detect、audit |
| `gate-train/candidate_bundle/` | gate-train | generate、calibrate、detect |
| `inputs/final_code.jsonl` | generate | detect、posthoc evaluation、audit |
| `generation/audit.jsonl` | generate | report、audit；绝不是 detector input |
| `generation/finalizer.jsonl` | generate（启用 finalizer 时） | audit |
| `generation/progress.json` | generate | audit |
| `calibration/negative_corpus.jsonl` | calibrate | calibrate、audit |
| `calibration/reference_calibration.json` | calibrate | detect、report、Metric Contract |
| `detection/positive_details.jsonl` | detect | report、Metric Contract |
| `reports/reference_report.json` | report | Metric Contract、audit |
| `reports/pass_report_posthoc.json` | posthoc | Metric Contract |
| `reports/metric_contract.jsonl` | audit 后统一汇总 | 当前 Full Reproduction 结果 |
| `audit/*.json` | audit | 独立完整性检查 |

## 身份与绑定

当前 artifact 使用当前 schema version，并以 SHA-256 绑定配置、source catalog、
key identifier、Gate Bundle、tokenizer、semantic encoder、window contract、
negative corpus manifest、generation config 与内容绑定的 generation model
identity。私有 key bytes 不进入公开 artifact。

`generation/manifest.json` 与 run state 同时记录 `final_code_sha256` 和
`final_code_row_count`；calibrate、detect、report、audit 逐阶段复核。
calibration、positive detection details、reference report 也由 run state
记录 SHA-256。`audit/artifact_integrity.json` 汇总这些核心 hash，Metric
Contract 在读取 report 前再次验证，防止 audit 后篡改。

`candidate_bundle_manifest.json` 必须来自同一 run 的 `gate-train`，携带
`candidate_bundle_sha256` 和当前 experiment contract hash。后续阶段重新计算
树 hash；不接受外部路径或旧 candidate layout。

## Test-only artifact

受控测试可注入轻量依赖。只要测试产生 Gate artifact，就必须携带一致的
`diagnostic_test_backend=true`、`formal_eligible=false`、`diagnostic_only=true`
和 `not_official_method=true` 身份。生产 preflight 不接受这些 artifact。

## Metric Contract

`scripts/wfcllm_metric_contract.py` 只读取上述当前布局，输出
`wfcllm-metric-contract/v2`。每个已 audit 的有效当前 run 始终得到一行；
样本不足和缺少 posthoc Pass@1 以 `null` 与 caveat 表达，而不是省略整行。
其他 Pass@k 可生成当前 posthoc 报告，但不进入 Metric Contract。
