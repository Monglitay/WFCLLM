# WFCLLM Mainline Refactor Spec

日期：2026-07-08

状态：Draft for execution

目标分支：`codex/wfcllm-mainline-refactor`

基线分支：`sawr-minimal-generation-embedding`

最终合并目标：`develop`，稳定后再进入 `main`

## 1. Executive Summary

本重构将项目正式方向从旧 WFCLLM watermark pipeline 切换到新版 WFCLLM 主线：

- structure-aware generation-time semantic watermarking
- evidence-only retry
- strict no-quality-gate protocol
- strict final-code-only detector
- 默认主方法候选为原 SAWR 的 `evidence_retry_seed7x3 family`

重构完成后，对外方法和项目名称统一为 **WFCLLM**。`SAWR` 只作为历史实验名和短期兼容 shim 存在，不再作为正式主目录、正式概念或论文主方法名。

旧 WFCLLM 的 token-channel、dual-channel、adaptive gamma、legacy watermark/extract、old ablation/pretrain、旧实验脚本、旧配置和旧文档将归档或移出默认路径。保留并迁移仍有价值的基础设施：datasets、Python parser、encoder、semantic LSH/keying/verifier、KV rollback、structure-aware boundary/state-machine/generator、strict code-only detector、no-quality-gate analysis/integrity audit。

推荐执行方案为 **方案 B：中等方案**。该方案把 SAWR 核心直接迁入新版主目录，同时保留 `wfcllm/sawr` 作为短期 compatibility shim，并系统归档旧代码、配置、文档、脚本和实验结果索引。

## 2. Goals

1. 将 `wfcllm/sawr` 中已经验证的核心机制升格为新版 WFCLLM 正式主干。
2. 将 `evidence_retry_seed7x3 family` 固化为默认 method preset 和论文复现入口。
3. 默认 CLI 和默认 config 指向新版 WFCLLM，而不是旧 `encoder -> watermark -> extract` pipeline。
4. 强制 no-quality-gate：generation、retry、selector、calibration、final candidate selection 中不得使用 pass/test/correctness proxy。
5. 强制 strict final-code-only detector：detector input 只能包含 `id`, `dataset`, `prompt`, `final_code`。
6. 明确 `evidence_only_selector_diagnostic` 只是 diagnostic upper bound，不是正式主方法。
7. 将旧代码、旧配置、旧脚本、旧文档、旧实验结果按策略归档，避免继续污染主线。
8. 建立清晰的 Git/worktree 流程，保护当前 dirty SAWR 工作树，并从正确基线创建重构 worktree。
9. 保留论文复现能力：历史实验不丢失，但不再作为默认路径。

## 3. Non-Goals

1. 本次重构不承诺提升 detector AUROC、TP@<=8FP 或 pass@1。
2. 不把 `evidence_only_selector_diagnostic` 包装成正式 generation-time method。
3. 不继续维护 token-channel、dual-channel、adaptive gamma 作为默认研究路线。
4. 不允许通过 public doctest、hidden tests、AST quality、signature match、tail/length/static correctness 等 proxy 优化主结果。
5. 不把旧 artifact schema 继续暴露为新版 detector input。旧 artifacts 必须通过 sanitizer 变成四字段 final-code JSONL。
6. 不在没有明确确认前删除本地大型实验产物、模型、数据集、论文 PDF 或 review/refine 过程材料。

## 4. New WFCLLM Method Definition

新版 WFCLLM 正式方法定义：

> WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. It embeds semantic LSH evidence during generation through structure-aware boundary detection, rollback-capable evidence-only retry, and final-code-only statistical detection. The protocol strictly forbids any pass/test/correctness proxy during generation, retry, calibration, and final candidate selection.

默认主方法候选为 `evidence_retry_seed7x3 family`。

### 4.1 Current Mainline Candidate

`evidence_retry_seed7x3` 当前实验状态：

- pass@1 = `104/164 = 63.41%`
- AUROC = `0.660080`
- TP@<=8FP = `43/164`
- FP = `8`
- positive insufficient = `20`
- model-negative AUROC = `0.510332`

解释：

- 它 pass 接近目标线。
- 它 detector evidence 仍不够强。
- 它是当前最值得升格为正式 **单配置 generation-time 方法** 的候选。
- 论文和 README 中不能过度声称 detector 已解决，应明确 detector evidence 仍是主要瓶颈。

### 4.2 Diagnostic Upper Bound

`evidence_only_selector_diagnostic` 当前状态：

- pass@1 = `78/164 = 47.56%`
- canonical/reference AUROC = `0.8178353658536586`
- TP@<=8FP = `98/164 = 59.76%`
- FP = `8`
- positive insufficient = `47`
- model-negative AUROC = `0.8165154669839382`
- sanitized detector input strict code-only，fields only: `dataset`, `final_code`, `id`, `prompt`

它只能作为 diagnostic / upper bound：

- 它是多套预注册生成配置上的 evidence-only 后置选择器。
- 即使不使用 pass/test/correctness proxy，它仍然不是单配置 generation-time method。
- 它不能成为默认 method preset，不能进入 `configs/base_config.json`，不能作为论文正式主方法。

### 4.3 Other Single Configs

- `low_perturb_seed6`: pass `111/164 = 67.68%`, AUROC `0.592858`, TP `41/164`, FP `8`
- `ordinal_anchor_seed8`: pass `115/164 = 70.12%`, AUROC `0.589493`, TP `34/164`, FP `8`

解释：

- 两者 pass 更高，但 strict code-only detector evidence 更弱。
- 可以作为 mechanism comparison，不作为默认新版 WFCLLM 主方法。

## 5. No-Quality-Gate Contract

### 5.1 Allowed Signals

正式 generation/retry/select/calibrate/detect 主流程中允许的信号：

- generation-time rule hit/miss
- `accepted_hit_count`
- `closed_without_hit_count`
- `fallback_count`
- `candidate_count`
- retry budget / rollback budget 状态
- final-code-only detector score
- proxy windows
- scoreable contexts
- insufficient-evidence flag
- negative corpus 上的 detector calibration statistic

### 5.2 Forbidden Signals

以下信号禁止进入 generation、retry、selector、calibration、final candidate selection：

- hidden tests
- public doctests
- code execution pass/fail
- `pass`, `passed`, `correctness_result`
- AST parse success as quality gate
- syntax validity
- target function presence
- signature compatibility
- suspicious tail
- truncation heuristic
- duplicate function heuristic
- import safety
- return/yield presence
- prompt contract heuristic
- hardcoded-example heuristic
- static correctness score
- complexity score
- generated-vs-canonical length gap
- reference/canonical correctness proxy
- logits quality proxy
- checkpoint-derived correctness proxy

允许在 final code 冻结后进行 pass@1 统计，但该统计必须标记为 posthoc reporting：

```json
{
  "posthoc_only": true,
  "not_used_for_generation": true,
  "not_used_for_retry": true,
  "not_used_for_selection": true,
  "not_used_for_calibration": true,
  "not_used_for_detection": true
}
```

### 5.3 Evidence Retry Key

正式 evidence retry 只允许使用如下等价排序：

```python
(
    accepted_hit_count,
    -closed_without_hit_count,
    -fallback_count,
    candidate_count,
    -attempt_index,
)
```

禁止将 pass/test/correctness/static quality 作为 retry key。

## 6. Strict Code-Only Detector Contract

### 6.1 Official Detector Input

官方 detector input JSONL 每行必须 exactly 四字段：

```json
{
  "id": "HumanEval/0",
  "dataset": "humaneval",
  "prompt": "...",
  "final_code": "..."
}
```

字段要求：

- `id`: non-empty string
- `dataset`: string，当前正式支持 `humaneval`、`mbpp`
- `prompt`: string
- `final_code`: string

### 6.2 Forbidden Detector Input Fields

detector input 中禁止出现：

- `artifact_type`
- `schema_version`
- `generated_code`
- `watermark_params`
- `blocks`
- `audit`
- `audit_only`
- `detector_input_allowed`
- `generation_*`
- `audit_*`
- `logits`
- `checkpoint`
- `retry_trace`
- `rollback_trace`
- `sampling_trace`
- `candidate_sidecar`
- `p_value`
- `z_score`
- `detector_score`
- `score`
- `is_watermarked`
- `pass`
- `passed`
- `correctness_result`

官方 detector 不再接受 `generated_code` fallback。历史 artifact 必须先通过 sanitizer 变成四字段 `final_code.jsonl`。

### 6.3 Detector Output

detector output 可以包含：

- `id`
- `is_watermarked`
- `score`
- `threshold_5fpr`
- `threshold_at_target_fpr`
- `p_value`
- `fpr_target`
- `scoreable_contexts`
- `proxy_windows`
- `insufficient_evidence`
- `detector_mode`
- `context_summaries`
- `code_chars`
- `direct_statements`

detector output 禁止包含：

- generation checkpoint
- retry trace
- rollback trace
- logits
- audit event id
- candidate sidecar text
- `watermark_params`
- `blocks`

## 7. Proposed Package Layout

最终正式代码结构：

```text
wfcllm/
  __init__.py

  method/
    __init__.py
    config.py
    presets.py
    contracts.py
    artifacts.py
    pipeline.py

  generation/
    __init__.py
    boundary.py
    state_machine.py
    generator.py
    model_context.py
    kv_cache.py
    retry.py
    outputs.py
    pipeline.py

  semantic/
    __init__.py
    rules.py
    lsh.py
    keying.py
    lsh_space.py
    verifier.py
    whitening.py

  detection/
    __init__.py
    config.py
    code_only.py
    proxy_windows.py
    scoring.py
    calibration.py
    pipeline.py
    metrics.py
    reports.py

  audit/
    __init__.py
    forbidden.py
    no_quality_gate.py
    detector_input_integrity.py
    artifact_integrity.py

  diagnostics/
    __init__.py
    evidence_selector.py
    quality_selector.py
    static_selector.py
    mechanism_analysis.py
    sparse_statistics.py

  cli/
  orchestration/
  datasets/
  lang/
  encoder/
  common/

  sawr/
    __init__.py
    boundary.py
    config.py
    generator.py
    pipeline.py
    rules.py
    semantic_lsh.py
    state_machine.py
    selection.py
    detect/
      __init__.py
      calibration.py
      config.py
      metrics.py
      pipeline.py
      proxy_windows.py
      scoring.py
```

`wfcllm/sawr` 只做 compatibility shim，例如：

```python
from wfcllm.generation.boundary import *
```

并发出 `DeprecationWarning`。正式文档、默认 CLI、默认 config、论文复现不再引用 `wfcllm.sawr`。

## 8. Module Migration Map

### 8.1 SAWR Core to New Mainline

| Current path | New path | Policy |
| --- | --- | --- |
| `wfcllm/sawr/config.py` | `wfcllm/method/config.py` | rename `Sawr*` to `WFCLLM*` |
| `wfcllm/sawr/pipeline.py` | `wfcllm/generation/pipeline.py` | final writer must produce four-field detector input plus sidecars |
| `wfcllm/sawr/generator.py` | `wfcllm/generation/generator.py` | split model context if useful |
| `wfcllm/sawr/generator.py` model context | `wfcllm/generation/model_context.py` | isolate LM/KV rollback logic |
| `wfcllm/sawr/boundary.py` | `wfcllm/generation/boundary.py` | direct migration |
| `wfcllm/sawr/state_machine.py` | `wfcllm/generation/state_machine.py` | direct migration |
| `wfcllm/sawr/rules.py` | `wfcllm/semantic/rules.py` | direct migration |
| `wfcllm/sawr/semantic_lsh.py` | `wfcllm/semantic/lsh.py` | direct migration |
| `wfcllm/sawr/detect/config.py` | `wfcllm/detection/config.py` | rename detector mode from SAWR naming to WFCLLM naming |
| `wfcllm/sawr/detect/proxy_windows.py` | `wfcllm/detection/proxy_windows.py` | direct migration |
| `wfcllm/sawr/detect/scoring.py` | `wfcllm/detection/scoring.py` | direct migration |
| `wfcllm/sawr/detect/calibration.py` | `wfcllm/detection/calibration.py` | direct migration |
| `wfcllm/sawr/detect/pipeline.py` | `wfcllm/detection/pipeline.py` | enforce exactly four input fields |
| `wfcllm/sawr/detect/metrics.py` | `wfcllm/detection/metrics.py` | direct migration |
| `wfcllm/sawr/selection.py` | `wfcllm/diagnostics/quality_selector.py` | diagnostic only |

### 8.2 Legacy WFCLLM Utilities to Keep

| Current path | New path | Policy |
| --- | --- | --- |
| `wfcllm/watermark/keying.py` | `wfcllm/semantic/keying.py` | keep |
| `wfcllm/watermark/lsh_space.py` | `wfcllm/semantic/lsh_space.py` | keep |
| `wfcllm/watermark/kv_cache.py` | `wfcllm/generation/kv_cache.py` | keep |

### 8.3 Legacy Code to Archive

Move out of importable production path:

```text
wfcllm/watermark/
wfcllm/watermark/token_channel/
wfcllm/watermark/adaptive_gamma/
wfcllm/extract/
wfcllm/ablation/
wfcllm/pretrain/
wfcllm/evaluation/dual_channel.py
```

Archive destination:

```text
archive/legacy_wfcllm_2026_07/code/
```

## 9. Scripts Cleanup Plan

### 9.1 New Official Scripts

```text
scripts/
  wfcllm_generate.py
  wfcllm_detect.py
  wfcllm_sanitize_final_code.py
```

These scripts are secondary to `run.py`, but useful for reproducibility and direct workflows.

### 9.2 Repro Scripts

```text
scripts/repro/
  run_wfcllm_evidence_retry_seed7x3.sh
  run_wfcllm_detector_scoring.sh
  run_wfcllm_pass_reporting_posthoc.sh
  run_wfcllm_full164_repro.sh
```

### 9.3 Diagnostic Scripts

```text
scripts/diagnostics/
  analyze_no_quality_gate.py
  analyze_evidence_selector_upper_bound.py
  analyze_sparse_statistics.py
  select_evidence_only_candidates.py
  select_quality_candidates_diagnostic.py
  select_static_candidates_diagnostic.py
```

Diagnostic scripts must mark their outputs:

```json
{
  "diagnostic_only": true,
  "not_official_method": true
}
```

### 9.4 Legacy Scripts

Move to:

```text
scripts/legacy/
```

Candidates:

```text
scripts/run_sawr_smoke.py
scripts/run_sawr_detect.py
scripts/run_sawr_evidence_generation.sh
scripts/run_sawr_evidence_detector_scoring.sh
scripts/run_sawr_evidence_pass_eval.sh
scripts/run_sawr_evidence_selector_calibration.sh
scripts/run_sawr_evidence_validation.sh
scripts/run_sawr_e1_e2_diagnostics.py
scripts/run_sawr_negative_seed_probe.py
scripts/run_sawr_ordinal_probe30.sh
scripts/run_sawr_r6_model_negative_control.sh
scripts/run_sawr_r6_score_ranking_control.sh
scripts/analyze_sawr_goal.py
scripts/analyze_sawr_evidence_goal.py
scripts/analyze_sawr_multicandidate_selector.py
scripts/analyze_sawr_no_quality_gate_goal.py
scripts/analyze_sawr_r6_score_ranking_control.py
scripts/analyze_sawr_sparse_stat_conformal.py
scripts/analyze_sawr_sparse_statistics.py
scripts/analyze_sawr_tpr30_optimization.py
scripts/select_sawr_candidates.py
scripts/select_sawr_static_candidates.py
scripts/merge_sawr_final_code.py
scripts/evaluate_dual_channel.py
scripts/run_ablation.py
scripts/build_entropy_profile.py
scripts/calibrate.py
scripts/calibrate_threshold.py
scripts/generate_negative_corpus.py
scripts/merge_watermarked.py
scripts/stage1_lasc_pilot.py
```

Short-term wrappers may remain, but they must print deprecation warnings and point to the new scripts.

## 10. Config Cleanup Plan

Final config layout:

```text
configs/
  base_config.json
  wfcllm/
    evidence_retry_seed7x3.json
    detector_proxy_penalized_alpha04.json
    repro_humaneval_full164.json
    diagnostics/
      evidence_only_selector_diagnostic.json
  legacy/
    README.md
    base_config_legacy_watermark_extract.json
    exp1_ngram_retries.json
    exp2_lower_delta.json
    exp3_higher_gamma.json
    exp3_new_gamma.json
    exp4_higher_lsh_d.json
    exp4_temp_escalation.json
    exp_semantic_only.json
    exp_white.json
    humaneval_10_config.json
    ablation/
      example_dual_channel.yaml
```

### 10.1 New Base Config

`configs/base_config.json` must point to新版 WFCLLM:

```json
{
  "method": {
    "name": "evidence_retry_seed7x3",
    "strict_no_quality_gate": true,
    "strict_code_only_detector": true
  },
  "generation": {},
  "semantic_lsh": {},
  "detector": {},
  "calibration": {},
  "artifacts": {},
  "runtime": {}
}
```

The actual preset values live in:

```text
configs/wfcllm/evidence_retry_seed7x3.json
configs/wfcllm/detector_proxy_penalized_alpha04.json
configs/wfcllm/repro_humaneval_full164.json
```

### 10.2 Forbidden in Default Config

The new `configs/base_config.json` must not contain:

```text
token_channel
dual-channel
adaptive_gamma
watermark
extract
pretrain
token_channel_train
build_entropy_profile
```

## 11. CLI Redesign

### 11.1 New Default Phases

Default `python run.py` should execute新版 WFCLLM main workflow, not old `encoder/watermark/extract`.

New default phase list:

```text
generate
calibrate
detect
report
audit
```

Optional/reporting phases:

```text
posthoc-pass-report
diagnostic-selector
```

Legacy phases must require an explicit legacy flag or legacy entrypoint:

```text
legacy-watermark
legacy-extract
legacy-token-channel-train
legacy-build-entropy-profile
legacy-pretrain
legacy-ablation
```

### 11.2 User-Facing Commands

Recommended official commands:

```bash
python run.py --config configs/base_config.json
python run.py --phase generate
python run.py --phase calibrate --negative-input data/negative/reference_final_code.jsonl
python run.py --phase detect --input data/runs/<run_id>/inputs/final_code.jsonl
python run.py --phase report --positive-details ... --negative-details ...
python run.py --phase audit --run-dir data/runs/<run_id>
```

Diagnostics must be explicit:

```bash
python run.py --phase diagnostic-selector --diagnostic-only
```

There must not be a production option such as `--allow-quality-proxy`.

## 12. Artifact Schema and Run Directory Layout

New official run layout:

```text
data/runs/
  20260708_153000_evidence_retry_seed7x3/
    config/
      resolved_config.json
      method_preset.json
    inputs/
      final_code.jsonl
    generation/
      audit.jsonl
      candidate_sidecar.jsonl
      raw_attempt_summary.jsonl
    calibration/
      reference_calibration.json
    detection/
      positive_details.jsonl
      reference_negative_details.jsonl
      model_negative_details.jsonl
    reports/
      reference_report.json
      model_negative_report.json
      pass_report_posthoc.json
    audit/
      detector_input_integrity.json
      no_quality_gate_integrity.json
      artifact_integrity.json
    logs/
      run.log
```

### 12.1 Official Inputs

`inputs/final_code.jsonl`:

```json
{"id": "...", "dataset": "...", "prompt": "...", "final_code": "..."}
```

Exactly four fields.

### 12.2 Generation Sidecars

`generation/audit.jsonl`:

```json
{
  "audit_only": true,
  "detector_input_allowed": false
}
```

May include event, decision, reason, candidate hash, rule name, hit/miss/fallback counts.

`generation/candidate_sidecar.jsonl`:

```json
{
  "audit_only": true,
  "detector_input_allowed": false,
  "diagnostic_only": true
}
```

May include normalized candidate text and LSH details from generation audit.

`generation/raw_attempt_summary.jsonl`:

- per evidence-retry attempt aggregate
- no pass/correctness fields

### 12.3 Calibration

`calibration/reference_calibration.json` must not contain raw `secret_key`.

It may contain:

- `secret_key_sha256`
- `detector_mode`
- public detector config
- bucket edges
- null distributions
- threshold

### 12.4 Reports

`reports/pass_report_posthoc.json` must be posthoc-only and must never feed generation, retry, selector, calibration, or detection.

## 13. Legacy Code Archive/Delete Policy

### 13.1 Git-Managed Archive

Create:

```text
archive/
  legacy_wfcllm_2026_07/
    README.md
    MANIFEST.json
    code/
    configs/
    scripts/
    docs/
    experiment_results/
    paper_repro/
```

Suggested layout:

```text
archive/legacy_wfcllm_2026_07/
  code/
    watermark/
    extract/
    token_channel/
    adaptive_gamma/
    dual_channel/
    ablation/
    pretrain/
  configs/
  scripts/
  docs/
  experiment_results/
    MANIFEST.md
    sawr_mechanism_no_quality_gate_goal_20260621_summary/
    sawr_evidence_constrained_goal_20260621_summary/
    sawr_full164_tpr30_goal_20260620_summary/
    legacy_watermark_extract_summary/
  paper_repro/
    old_repro_commands.md
    old_metrics_summary.md
    old_artifact_index.md
```

Git-managed archive should contain small files, summaries, configs, docs, and manifests. It must not include large model files, large raw JSONL, checkpoints, logits dumps, or full trace dumps.

### 13.2 Local Data Archive

Large local experiment outputs should move to:

```text
data/archive/
  legacy_wfcllm_2026_07/
    watermarked/
    results/
    diagnostics/
    eval/
    token_channel/
    checkpoints/
    MANIFEST.local.json
  sawr_experiments_2026_06/
    sawr_mechanism_no_quality_gate_goal_20260621/
    sawr_evidence_constrained_goal_20260621/
    sawr_full164_tpr30_goal_20260620/
    sawr_sparse_detector_goal_20260616/
    sawr_e1_e2_sidecar_probe_20260616/
    MANIFEST.local.json
```

### 13.3 Deletion Candidates

Deletion candidates require explicit confirmation:

```text
__pycache__/
*.pyc
data/run_state.json
temporary logs
intermediate logits dumps
intermediate checkpoints
duplicate large JSONL outputs
old smoke subset outputs
```

Do not delete:

```text
data/models/
data/datasets/
papers/
review-stage/
review-input/
refine-logs/
idea-stage/
.aris/
```

unless the user explicitly asks.

## 14. Tests to Add or Migrate

### 14.1 Migrate Tests

| Current path | New path |
| --- | --- |
| `tests/sawr/test_boundary.py` | `tests/generation/test_boundary.py` |
| `tests/sawr/test_state_machine.py` | `tests/generation/test_state_machine.py` |
| `tests/sawr/test_generator.py` | `tests/generation/test_generator.py` |
| `tests/sawr/test_pipeline.py` | `tests/generation/test_pipeline.py` |
| `tests/sawr/test_rules.py` | `tests/semantic/test_rules.py` |
| `tests/sawr/detect/*` | `tests/detection/*` |
| `tests/integration/test_sawr_smoke_cli.py` | `tests/integration/test_wfcllm_generate_cli.py` |
| `tests/integration/test_sawr_detect_cli.py` | `tests/integration/test_wfcllm_detect_cli.py` |

### 14.2 New Tests

```text
tests/method/test_evidence_retry_seed7x3_preset.py
tests/method/test_no_quality_gate_contract.py
tests/method/test_pipeline.py
tests/generation/test_evidence_retry_key.py
tests/detection/test_code_only_contract.py
tests/detection/test_detector_rejects_generated_code.py
tests/audit/test_detector_input_integrity.py
tests/audit/test_no_quality_gate_integrity.py
tests/diagnostics/test_evidence_selector_diagnostic.py
tests/diagnostics/test_quality_selector_diagnostic.py
tests/integration/test_default_cli_uses_new_wfcllm.py
tests/integration/test_legacy_phase_requires_legacy_flag.py
```

### 14.3 Tests to Archive

Archive or remove from default suite:

```text
tests/watermark/token_channel/*
tests/extract/test_token_channel.py
old dual-channel integration tests
old adaptive gamma tests
old pretrain tests
old ablation tests
```

Only retain them under legacy archive if historical validation is needed.

## 15. Documentation Rewrite Plan

### 15.1 New Official Docs

```text
docs/
  WFCLLM_METHOD.md
  NO_QUALITY_GATE_PROTOCOL.md
  STRICT_CODE_ONLY_DETECTOR.md
  ARTIFACT_SCHEMA.md
  REPRO_EVIDENCE_RETRY_SEED7X3.md
  DIAGNOSTIC_UPPER_BOUND.md
  LEGACY_ARCHIVE.md
  design/
    generation_time_structure_aware_watermark.md
    final_code_detector.md
    semantic_lsh_core.md
```

### 15.2 Docs to Archive or Rewrite

Move old docs to:

```text
docs/archive/
  sawr/
  legacy_wfcllm/
  plans/
  progress/
  superpowers/
```

Candidates:

```text
docs/SAWR_GENERATION_TIME_EMBEDDING_SCHEME.md
docs/design/token-channel-batch-inference.md
docs/design/基于LSH的改进方案.md
docs/design/基于语句块的语义特征的生成时代码水印方案.md
docs/design/概要设计文档.md
docs/design/水印检测改造.md
docs/design/详细设计文档.md
docs/history/*
docs/plans/*
docs/progress/*
docs/superpowers/plans/*
docs/superpowers/specs/*
docs/experiment/*
```

`docs/experiment/2026-06-15-sawr-structure-aware-generation-experiment-report.md` should be preserved under:

```text
docs/archive/sawr/experiment_reports/
```

### 15.3 Naming Rules

Official docs should use `WFCLLM` as the method/project name.

`SAWR` may appear only in:

- history sections
- archive docs
- compatibility shim docs
- experiment provenance

It must not appear as the formal current method name in README, CLAUDE, default config, or method docs.

## 16. Step-by-Step Migration Plan

### Phase 0: Git/Worktree Alignment

Current observed repository state:

- current branch: `sawr-minimal-generation-embedding`
- dirty working tree with SAWR code/test changes
- many untracked scripts/docs/data/papers/review artifacts
- many old `.claude/worktrees/agent-*`
- broken Codex ref can break `git log --all`

#### Phase 0.1: Select Correct Baseline

Correct baseline:

```text
sawr-minimal-generation-embedding
```

Reason:

- It contains the SAWR generation/detection commits to be promoted.
- `develop` has no observed extra commits ahead of it at inspection time.
- `main` is stable release target, not the right starting point for this refactor.

#### Phase 0.2: Protect Current Dirty Work

Do not switch branches in the current dirty worktree.

First protect code/test/script/doc changes that should enter Git:

```text
wfcllm/sawr/*
tests/sawr/*
tests/integration/test_sawr_*.py
scripts/run_sawr_*.py
scripts/analyze_sawr_*.py
scripts/select_sawr_*.py
scripts/merge_sawr_final_code.py
docs/SAWR_GENERATION_TIME_EMBEDDING_SCHEME.md
docs/experiment/2026-06-15-sawr-structure-aware-generation-experiment-report.md
```

Suggested protection commit:

```text
chore: preserve sawr working tree before mainline refactor
```

Do not blindly `git stash -u`, because it would mix large local data/papers/review artifacts with source changes.

#### Phase 0.3: Create Refactor Worktree

Create a clean worktree for the actual refactor:

```bash
git worktree add .worktrees/wfcllm-mainline-refactor -b codex/wfcllm-mainline-refactor sawr-minimal-generation-embedding
```

All large-scale file moves happen in:

```text
.worktrees/wfcllm-mainline-refactor
```

#### Phase 0.4: Merge Direction

Correct merge order:

```text
develop -> sawr-minimal-generation-embedding
sawr-minimal-generation-embedding -> codex/wfcllm-mainline-refactor
codex/wfcllm-mainline-refactor -> develop
develop -> main
```

Before merging, verify:

```bash
git log --oneline sawr-minimal-generation-embedding..develop
git log --oneline develop..sawr-minimal-generation-embedding
```

If the first command is empty and the second command lists SAWR commits, the direction is correct.

Do not merge from these branches by default:

```text
worktree-agent-a*
worktree-task1-revision-quality-fix
feature/offline-resources
feature/checkpoint-recovery
develop-LSH
codex-anchor-effectiveness-validation
```

Only cherry-pick from them if a specific needed fix is identified.

#### Phase 0.5: Broken Ref Cleanup

Observed broken ref:

```text
refs/codex/turn-diffs/captures/1781516728192/fd1dc2ad-7dc5-4426-8e4b-31f44e2e412c/base
```

Policy:

- Do not depend on `git log --all` until this is handled.
- Use explicit branch comparisons.
- Delete or repair the broken ref only after explicit confirmation or after backing up refs.

### Phase 1: New Package Layout

Create:

```text
wfcllm/method/
wfcllm/generation/
wfcllm/semantic/
wfcllm/detection/
wfcllm/audit/
wfcllm/diagnostics/
```

### Phase 2: Migrate SAWR Core

Move SAWR implementation to new packages.

Retain `wfcllm/sawr` shims with `DeprecationWarning`.

### Phase 3: Enforce Contracts

Implement:

- no-quality-gate contract checks
- strict code-only input validation
- forbidden field scanning
- artifact integrity audit

### Phase 4: CLI and Config Flip

Update:

- `configs/base_config.json`
- `wfcllm/cli/arguments.py`
- `wfcllm/cli/config_resolver.py`
- `wfcllm/cli/runners.py`
- `wfcllm/orchestration/state.py`
- `wfcllm/orchestration/pipeline.py`

Default must be新版 WFCLLM.

### Phase 5: Legacy Archive

Move old code/config/scripts/docs into archive paths.

### Phase 6: Experiment Results Local Archive

Move or index local experiment results under `data/archive/`.

Generate:

```text
archive/legacy_wfcllm_2026_07/experiment_results/MANIFEST.md
data/archive/legacy_wfcllm_2026_07/MANIFEST.local.json
data/archive/sawr_experiments_2026_06/MANIFEST.local.json
```

### Phase 7: Tests

Migrate tests and add contract tests.

### Phase 8: Docs

Rewrite README, CLAUDE, method docs, repro docs, and archive docs.

### Phase 9: Validation

Run targeted checks:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_default_cli_uses_new_wfcllm.py -v
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Run full suite if feasible:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

### Phase 10: Merge

Merge:

```text
codex/wfcllm-mainline-refactor -> develop
```

After stabilization:

```text
develop -> main
```

## 17. Acceptance Criteria

### 17.1 Code Structure

- `wfcllm/method`, `wfcllm/generation`, `wfcllm/semantic`, `wfcllm/detection`, `wfcllm/audit`, `wfcllm/diagnostics` exist.
- `wfcllm/sawr` only contains compatibility shims.
- Production mainline does not import `wfcllm.sawr`.
- Production mainline does not import archived legacy code.
- Production mainline does not import `experiment.*`.

### 17.2 Config

- `configs/base_config.json` points to `evidence_retry_seed7x3`.
- `configs/base_config.json` contains no `token_channel`, `dual-channel`, `adaptive_gamma`, old `watermark`, old `extract`, `pretrain`, or `token_channel_train`.
- Old configs live under `configs/legacy/`.

### 17.3 CLI

- `python run.py --status` shows new WFCLLM phases.
- `python run.py` runs or prepares the new WFCLLM workflow.
- Old phases require legacy entrypoints or explicit legacy flags.

### 17.4 Detector Contract

- Official detector rejects any input row with fields beyond `id`, `dataset`, `prompt`, `final_code`.
- Official detector rejects `generated_code`.
- Official detector rejects audit/sidecar/logits/checkpoint/retry/pass fields.

### 17.5 No-Quality-Gate Contract

- Generation/retry/final selection does not read pass/test/correctness/static quality fields.
- Posthoc pass report is clearly marked posthoc-only.
- `evidence_only_selector_diagnostic` is labeled diagnostic-only everywhere.

### 17.6 Archive

- `archive/legacy_wfcllm_2026_07/MANIFEST.json` exists.
- `configs/legacy/` contains old configs.
- `scripts/legacy/` contains old scripts or wrappers.
- `docs/archive/` contains old SAWR/legacy docs.
- `data/archive/*/MANIFEST.local.json` exists for local result archives if data movement is performed.

### 17.7 Documentation

- README and CLAUDE describe新版 WFCLLM, not old pipeline.
- `docs/WFCLLM_METHOD.md` defines current method.
- `docs/REPRO_EVIDENCE_RETRY_SEED7X3.md` contains exact reproduction parameters and current known metrics.
- `docs/DIAGNOSTIC_UPPER_BOUND.md` explains why `evidence_only_selector_diagnostic` is not official method.

### 17.8 Git

- Work is performed on `codex/wfcllm-mainline-refactor`.
- Branch is based on `sawr-minimal-generation-embedding`.
- Dirty SAWR work is protected before branch/worktree switching.
- Final merge target is `develop`, then `main`.
- Old agent worktrees are not merged by default.

## 18. Risks and Mitigations

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Dirty current worktree loses SAWR changes | High | Protect with explicit commit before worktree creation |
| Wrong branch chosen as baseline | High | Use `sawr-minimal-generation-embedding` as baseline; verify branch delta |
| Old pipeline remains default | High | Rewrite `configs/base_config.json`, `run.py`, orchestration phases, README |
| Quality proxy leaks through selector | High | Move quality/static selectors to diagnostics; add contract tests |
| Detector accepts legacy generated_code | High | Official detector exact-four-field validation |
| Large data accidentally committed | High | Use local data archive + manifest, not Git |
| Historical reproducibility lost | Medium | Archive code/config/docs/scripts and summary manifests |
| SAWR naming remains in official docs | Medium | Limit SAWR to archive/history/compatibility contexts |
| Broken Codex ref blocks Git commands | Medium | Avoid `git log --all`; repair ref after confirmation |
| model-negative AUROC weakens claim | Medium | Document as limitation; do not overclaim detector robustness |

## 19. Open Questions

1. Should encoder training remain an official optional phase, or should encoder checkpoint be treated as an external prerequisite?
2. Should `python run.py` run `generate -> calibrate -> detect -> report -> audit`, or stop after `generate` if no calibration corpus/artifact is configured?
3. Is `evidence_retry_seed7x3 family` allowed to include future seed/temperature variants, or must the official method fix `seed=7` exactly?
4. Should model-negative AUROC weakness be reported in README, only in repro docs, or only in paper limitation sections?
5. Should `wfcllm/watermark.verifier` be merged into the new semantic verifier API or archived entirely?
6. How long should `wfcllm/sawr` compatibility shims remain?
7. Should archive manifests include sha256 for all local artifacts, or only path/size/role metadata?
8. Should `papers/*.pdf` stay in the repo working tree, move to local archive, or be ignored entirely?
9. Should old `.claude/worktrees/agent-*` be pruned after the refactor branch is created?
10. Should broken Codex refs be cleaned as part of this refactor or left to a separate Git hygiene task?

## Appendix A. Evidence Retry Seed7x3 Preset

Canonical values:

```json
{
  "method": {
    "name": "evidence_retry_seed7x3",
    "strict_no_quality_gate": true,
    "strict_code_only_detector": true
  },
  "generation": {
    "dataset": "humaneval",
    "max_new_tokens": 256,
    "temperature": 0.25,
    "top_p": 0.95,
    "top_k": 0,
    "retry_repetition_penalty": 4.0,
    "torch_dtype": "bf16",
    "device": "cuda",
    "seed": 7,
    "load_in_4bit": true,
    "prompt_mode": "completion",
    "max_group_statements": 2,
    "retry_budget": 2,
    "statement_retry_budget": 4,
    "window_retry_budget": 3,
    "compound_retry_budget": 2,
    "global_rollback_budget": 180,
    "max_total_sampled_tokens": 32768,
    "evidence_retry_attempts": 3,
    "evidence_retry_seed_stride": 101
  },
  "semantic_lsh": {
    "rule_name": "semantic_lsh",
    "lsh_d": 4,
    "lsh_gamma": 0.25,
    "semantic_margin": 0.0,
    "use_ordinal_keying": false
  },
  "detector": {
    "lsh_d": 4,
    "gamma": 0.25,
    "semantic_margin": 0.0,
    "max_group_statements": 2,
    "min_scoreable_contexts": 1,
    "min_proxy_windows": 2,
    "target_fpr": 0.05,
    "use_ordinal_keying": false,
    "evidence_mode": "hit_plus_margin",
    "statistic": "calibrated_context_mean_proxy_penalized",
    "proxy_penalty_alpha": 0.4
  }
}
```

## Appendix B. Local Experiment Archive Manifest Shape

Suggested `MANIFEST.local.json`:

```json
{
  "archive_id": "legacy_wfcllm_2026_07",
  "created_at": "2026-07-08",
  "policy": "local artifacts archived, large files not committed",
  "entries": [
    {
      "name": "sawr_mechanism_no_quality_gate_goal_20260621",
      "original_path": "data/diagnostics/sawr_mechanism_no_quality_gate_goal_20260621",
      "archive_path": "data/archive/sawr_experiments_2026_06/sawr_mechanism_no_quality_gate_goal_20260621",
      "status": "archived_local",
      "role": "historical_diagnostic",
      "official_method": false,
      "diagnostic_only": true,
      "key_results": {
        "evidence_retry_seed7x3_pass": "104/164",
        "evidence_retry_seed7x3_auroc": 0.66008,
        "evidence_only_selector_diagnostic_auroc": 0.8178353658536586
      }
    }
  ]
}
```

## Appendix C. Worktree Policy Summary

Do:

```text
current dirty worktree -> protect commit
sawr-minimal-generation-embedding -> codex/wfcllm-mainline-refactor worktree
codex/wfcllm-mainline-refactor -> develop
develop -> main
```

Do not:

```text
switch current dirty worktree directly
start from main
merge old agent worktrees by default
commit data/models or data/datasets
commit large raw diagnostics/checkpoints/logits
```
