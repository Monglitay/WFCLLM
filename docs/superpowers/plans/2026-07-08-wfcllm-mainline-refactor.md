# WFCLLM 主线重构实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 将 `wfcllm/sawr` 中已验证的 structure-aware generation-time watermarking 升格为新版 WFCLLM 默认主线，并把旧 watermark/extract/token-channel/dual-channel/adaptive-gamma 路线从默认路径归档出去。

**架构：** 新主线按合同分层：`wfcllm.method` 定义 preset 与 no-quality-gate 合同，`wfcllm.generation` 负责结构感知生成和 evidence-only retry，`wfcllm.detection` 只接收四字段 final-code JSONL，`wfcllm.audit` 做完整性审计，`wfcllm.diagnostics` 承载非正式 selector 和机制分析。`wfcllm.sawr` 保留短期兼容 shim，并发出 `DeprecationWarning`；默认 CLI/config/docs 不再把 SAWR 当作正式方法名。

**技术栈：** Python 3.11、dataclasses、argparse、pytest、JSON/JSONL、Conda 环境 `WFCLLM`、离线测试变量 `HF_HUB_OFFLINE=1`。

---

## 执行前约束

- 在执行代码重构前阅读：`CLAUDE.md`、`AGENTS.md`、`docs/WFCLLM_MAINLINE_REFACTOR_SPEC.md`。
- 当前工作树已观察到大量 SAWR 相关 dirty/untracked 文件；执行重构前必须先保护这些工作，不要直接切分支。
- 执行本计划时不删除 `data/models/`、`data/datasets/`、`papers/`、`review-stage/`、`review-input/`、`refine-logs/`、`idea-stage/`、`.aris/`。
- 默认用 `git mv` 保留历史；只有明确的小型新文件用编辑器或 `apply_patch` 创建。
- 所有 pytest 命令都带 `HF_HUB_OFFLINE=1`，除非命令只是 `compileall`。
- 每个任务完成后 commit，commit 格式使用 `<type>: <description>`。

## 文件结构与职责

### 新增或升格的正式主线文件

- 创建：`wfcllm/method/__init__.py`  
  导出正式 method config、preset、contract helper，不导出 SAWR 名称。
- 创建：`wfcllm/method/config.py`  
  定义 `WFCLLMGenerationConfig`、`WFCLLMRuleConfig`、`WFCLLMPipelineConfig`、`WFCLLMMethodConfig`、`WFCLLMResolvedConfig`。从 `wfcllm/sawr/config.py` 迁移并把 `Sawr*` 类型改成 `WFCLLM*`。
- 创建：`wfcllm/method/presets.py`  
  固化 `evidence_retry_seed7x3` preset 和 detector preset；默认 preset 不包含 token-channel、dual-channel、adaptive-gamma、旧 watermark/extract/pretrain 字段。
- 创建：`wfcllm/method/contracts.py`  
  定义 no-quality-gate 禁止字段、允许 evidence retry key、diagnostic-only 标记校验。
- 创建：`wfcllm/method/artifacts.py`  
  定义 run directory 路径解析、四字段 final-code 记录、audit/sidecar/posthoc marker 的 JSON 安全转换。
- 创建：`wfcllm/method/pipeline.py`  
  薄编排层：把生成、校准、检测、报告、审计按新 run layout 连接起来；不持有旧 pipeline 逻辑。
- 创建：`wfcllm/generation/__init__.py`  
  导出 boundary、state machine、generator、pipeline 的正式 WFCLLM 名称。
- 创建：`wfcllm/generation/boundary.py`  
  从 `wfcllm/sawr/boundary.py` 迁移。
- 创建：`wfcllm/generation/state_machine.py`  
  从 `wfcllm/sawr/state_machine.py` 迁移；保持 evidence-only 决策，不读取 quality proxy。
- 创建：`wfcllm/generation/generator.py`  
  从 `wfcllm/sawr/generator.py` 迁移；把 model/KV 逻辑拆到 `model_context.py` 和 `kv_cache.py`。
- 创建：`wfcllm/generation/model_context.py`  
  承载原 `SawrModelContext`、`SawrCheckpoint`、prompt/truncation/model loader helper，并改名为 `WFCLLMModelContext`、`WFCLLMCheckpoint`。
- 创建：`wfcllm/generation/kv_cache.py`  
  从 `wfcllm/watermark/kv_cache.py` 迁移保留。
- 创建：`wfcllm/generation/retry.py`  
  定义 `evidence_retry_key(result, attempt_index)`，返回 `(accepted_hit_count, -closed_without_hit_count, -fallback_count, candidate_count, -attempt_index)`。
- 创建：`wfcllm/generation/outputs.py`  
  定义 final-code、audit、candidate-sidecar、raw-attempt-summary 的 writer；final-code writer 只能输出 `id,dataset,prompt,final_code`。
- 创建：`wfcllm/generation/pipeline.py`  
  从 `wfcllm/sawr/pipeline.py` 迁移；输出新 run directory layout。
- 创建：`wfcllm/semantic/__init__.py`  
  导出 semantic LSH/keying/verifier API。
- 创建：`wfcllm/semantic/rules.py`  
  从 `wfcllm/sawr/rules.py` 迁移。
- 创建：`wfcllm/semantic/lsh.py`  
  从 `wfcllm/sawr/semantic_lsh.py` 迁移。
- 创建：`wfcllm/semantic/keying.py`  
  从 `wfcllm/watermark/keying.py` 迁移。
- 创建：`wfcllm/semantic/lsh_space.py`  
  从 `wfcllm/watermark/lsh_space.py` 迁移。
- 创建：`wfcllm/semantic/verifier.py`  
  从 `wfcllm/watermark/verifier.py` 迁移，保留 semantic verifier API。
- 创建：`wfcllm/semantic/whitening.py`  
  承载 whitening artifact loader；迁移脚本需要的最小 API。
- 创建：`wfcllm/detection/__init__.py`  
  导出正式 detector API。
- 创建：`wfcllm/detection/config.py`  
  从 `wfcllm/sawr/detect/config.py` 迁移，`DETECTOR_MODE` 改为 `wfcllm-structure-aware-proxy-window/v1`。
- 创建：`wfcllm/detection/code_only.py`  
  定义官方 detector input 精确四字段校验，拒绝 `generated_code` fallback 和全部 forbidden fields。
- 创建：`wfcllm/detection/proxy_windows.py`  
  从 `wfcllm/sawr/detect/proxy_windows.py` 迁移。
- 创建：`wfcllm/detection/scoring.py`  
  从 `wfcllm/sawr/detect/scoring.py` 迁移。
- 创建：`wfcllm/detection/calibration.py`  
  从 `wfcllm/sawr/detect/calibration.py` 迁移；校准 artifact 不写 raw `secret_key`。
- 创建：`wfcllm/detection/pipeline.py`  
  从 `wfcllm/sawr/detect/pipeline.py` 迁移，调用 `code_only.validate_final_code_record_exact`。
- 创建：`wfcllm/detection/metrics.py`  
  从 `wfcllm/sawr/detect/metrics.py` 迁移。
- 创建：`wfcllm/detection/reports.py`  
  输出 reference/model-negative detector report；不包含 generation trace。
- 创建：`wfcllm/audit/__init__.py`  
  导出 audit helper。
- 创建：`wfcllm/audit/forbidden.py`  
  集中定义 detector/generation/posthoc 禁止字段集合。
- 创建：`wfcllm/audit/no_quality_gate.py`  
  扫描 config、audit、raw attempt、selector 输入，确保没有 pass/test/correctness/static quality proxy。
- 创建：`wfcllm/audit/detector_input_integrity.py`  
  读取 official `inputs/final_code.jsonl`，验证每行 exactly 四字段。
- 创建：`wfcllm/audit/artifact_integrity.py`  
  检查 audit-only/diagnostic-only marker、secret key 泄漏、detector output 禁止字段。
- 创建：`wfcllm/diagnostics/__init__.py`  
  导出 diagnostic-only selector/analysis。
- 创建：`wfcllm/diagnostics/evidence_selector.py`  
  从当前 evidence-only selector 迁移；所有输出写 `diagnostic_only=true, not_official_method=true`。
- 创建：`wfcllm/diagnostics/quality_selector.py`  
  从 `wfcllm/sawr/selection.py` 迁移；只能 diagnostic-only。
- 创建：`wfcllm/diagnostics/static_selector.py`  
  从 static candidate selector 脚本中抽出纯函数；只能 diagnostic-only。
- 创建：`wfcllm/diagnostics/mechanism_analysis.py`  
  从 no-quality-gate goal 分析脚本抽出复用逻辑。
- 创建：`wfcllm/diagnostics/sparse_statistics.py`  
  从 sparse statistics 脚本抽出复用逻辑。

### 需要改成 shim 或 wrappers 的文件

- 修改：`wfcllm/sawr/__init__.py`  
  兼容导出旧名称；导入时发出 `DeprecationWarning`。
- 修改：`wfcllm/sawr/config.py`  
  从 `wfcllm.method.config` re-export `WFCLLM*` 并提供 `Sawr*` 别名。
- 修改：`wfcllm/sawr/boundary.py`  
  re-export `wfcllm.generation.boundary`。
- 修改：`wfcllm/sawr/state_machine.py`  
  re-export `wfcllm.generation.state_machine`。
- 修改：`wfcllm/sawr/generator.py`  
  re-export `wfcllm.generation.generator` 与 `wfcllm.generation.model_context`。
- 修改：`wfcllm/sawr/pipeline.py`  
  re-export `wfcllm.generation.pipeline`。
- 修改：`wfcllm/sawr/rules.py`  
  re-export `wfcllm.semantic.rules`。
- 修改：`wfcllm/sawr/semantic_lsh.py`  
  re-export `wfcllm.semantic.lsh`。
- 修改：`wfcllm/sawr/detect/*.py`  
  re-export `wfcllm.detection.*`。
- 修改：`scripts/run_sawr_smoke.py`、`scripts/run_sawr_detect.py`  
  保留短期 wrapper，打印 deprecation warning，转发到新 scripts。

### CLI/config/orchestration 文件

- 修改：`configs/base_config.json`  
  指向新版 WFCLLM `evidence_retry_seed7x3`，不包含旧默认字段。
- 创建：`configs/wfcllm/evidence_retry_seed7x3.json`
- 创建：`configs/wfcllm/detector_proxy_penalized_alpha04.json`
- 创建：`configs/wfcllm/repro_humaneval_full164.json`
- 创建：`configs/wfcllm/diagnostics/evidence_only_selector_diagnostic.json`
- 创建：`configs/legacy/README.md`
- 移动：旧 `configs/exp*.json`、`configs/humaneval_10_config.json`、`configs/ablation/example_dual_channel.yaml` 到 `configs/legacy/`。
- 修改：`wfcllm/cli/arguments.py`  
  新 phases 和新 flags；旧 phases 需要 `--legacy`。
- 修改：`wfcllm/cli/config_resolver.py`  
  解析新 config section；保留 legacy resolver 供 legacy phases 使用。
- 修改：`wfcllm/cli/runners.py`  
  新增 `run_generate`、`run_calibrate`、`run_detect`、`run_report`、`run_audit`、`run_posthoc_pass_report`、`run_diagnostic_selector`、legacy runner wrapper。
- 修改：`wfcllm/cli/entry.py`  
  注册新 phases；禁止生产 flag `--allow-quality-proxy`。
- 修改：`wfcllm/orchestration/state.py`  
  主 phases 改成 `generate, calibrate, detect, report, audit`。
- 修改：`wfcllm/orchestration/pipeline.py`  
  extract input bypass 逻辑改成 detect input bypass；legacy phase skip 逻辑必须看到 `args.legacy`。
- 修改：`run.py`  
  仍保持薄 shim，不迁入业务逻辑。

### Scripts/docs/archive

- 创建：`scripts/wfcllm_generate.py`
- 创建：`scripts/wfcllm_detect.py`
- 创建：`scripts/wfcllm_sanitize_final_code.py`
- 创建：`scripts/repro/run_wfcllm_evidence_retry_seed7x3.sh`
- 创建：`scripts/repro/run_wfcllm_detector_scoring.sh`
- 创建：`scripts/repro/run_wfcllm_pass_reporting_posthoc.sh`
- 创建：`scripts/repro/run_wfcllm_full164_repro.sh`
- 创建：`scripts/diagnostics/analyze_no_quality_gate.py`
- 创建：`scripts/diagnostics/analyze_evidence_selector_upper_bound.py`
- 创建：`scripts/diagnostics/analyze_sparse_statistics.py`
- 创建：`scripts/diagnostics/select_evidence_only_candidates.py`
- 创建：`scripts/diagnostics/select_quality_candidates_diagnostic.py`
- 创建：`scripts/diagnostics/select_static_candidates_diagnostic.py`
- 创建：`scripts/legacy/README.md`
- 创建：`archive/legacy_wfcllm_2026_07/README.md`
- 创建：`archive/legacy_wfcllm_2026_07/MANIFEST.json`
- 创建：`archive/legacy_wfcllm_2026_07/experiment_results/MANIFEST.md`
- 创建：`docs/WFCLLM_METHOD.md`
- 创建：`docs/NO_QUALITY_GATE_PROTOCOL.md`
- 创建：`docs/STRICT_CODE_ONLY_DETECTOR.md`
- 创建：`docs/ARTIFACT_SCHEMA.md`
- 创建：`docs/REPRO_EVIDENCE_RETRY_SEED7X3.md`
- 创建：`docs/DIAGNOSTIC_UPPER_BOUND.md`
- 创建：`docs/LEGACY_ARCHIVE.md`
- 修改：`README.md`
- 修改：`CLAUDE.md`

### 测试文件

- 创建：`tests/method/test_evidence_retry_seed7x3_preset.py`
- 创建：`tests/method/test_no_quality_gate_contract.py`
- 创建：`tests/method/test_artifacts.py`
- 创建：`tests/generation/test_boundary.py`
- 创建：`tests/generation/test_state_machine.py`
- 创建：`tests/generation/test_generator.py`
- 创建：`tests/generation/test_pipeline.py`
- 创建：`tests/generation/test_evidence_retry_key.py`
- 创建：`tests/semantic/test_rules.py`
- 创建：`tests/semantic/test_lsh.py`
- 创建：`tests/detection/test_code_only_contract.py`
- 创建：`tests/detection/test_detector_rejects_generated_code.py`
- 创建：`tests/detection/test_pipeline.py`
- 创建：`tests/audit/test_detector_input_integrity.py`
- 创建：`tests/audit/test_no_quality_gate_integrity.py`
- 创建：`tests/audit/test_artifact_integrity.py`
- 创建：`tests/diagnostics/test_evidence_selector_diagnostic.py`
- 创建：`tests/diagnostics/test_quality_selector_diagnostic.py`
- 创建：`tests/integration/test_default_cli_uses_new_wfcllm.py`
- 创建：`tests/integration/test_legacy_phase_requires_legacy_flag.py`
- 创建：`tests/integration/test_wfcllm_generate_cli.py`
- 创建：`tests/integration/test_wfcllm_detect_cli.py`
- 创建：`tests/scripts/test_wfcllm_sanitize_final_code.py`
- 移动：`tests/sawr/*` 到新 package tests 后删除 `tests/sawr/`。
- 移动：旧 watermark/extract/token-channel/dual-channel/adaptive-gamma/pretrain/ablation tests 到 `archive/legacy_wfcllm_2026_07/tests/`，不留在默认 pytest suite。

---

### 任务 0：保护当前 SAWR 工作树并创建重构 worktree

**文件：**
- 修改：无业务代码修改
- 参考：`docs/WFCLLM_MAINLINE_REFACTOR_SPEC.md`

- [ ] **步骤 1：确认当前状态**

运行：

```bash
git status --short
git branch --show-current
git log --oneline sawr-minimal-generation-embedding..develop
git log --oneline develop..sawr-minimal-generation-embedding
```

预期：

```text
sawr-minimal-generation-embedding
```

并且 `git status --short` 显示 SAWR 相关 dirty/untracked 文件。第一个 `git log` 为空或只显示明确已知的 develop 维护提交；第二个 `git log` 显示 SAWR commits。

- [ ] **步骤 2：只暂存应保护的源码、测试、脚本、文档**

运行：

```bash
git add wfcllm/sawr tests/sawr tests/integration/test_sawr_smoke_cli.py tests/integration/test_sawr_detect_cli.py scripts/run_sawr_smoke.py scripts/run_sawr_detect.py scripts/analyze_sawr_evidence_goal.py scripts/analyze_sawr_goal.py scripts/analyze_sawr_multicandidate_selector.py scripts/analyze_sawr_no_quality_gate_goal.py scripts/analyze_sawr_r6_score_ranking_control.py scripts/analyze_sawr_sparse_stat_conformal.py scripts/analyze_sawr_sparse_statistics.py scripts/analyze_sawr_tpr30_optimization.py scripts/select_sawr_candidates.py scripts/select_sawr_static_candidates.py scripts/merge_sawr_final_code.py docs/SAWR_GENERATION_TIME_EMBEDDING_SCHEME.md docs/experiment/2026-06-15-sawr-structure-aware-generation-experiment-report.md
git status --short
```

预期：staged 列表中没有 `data/`、`papers/`、`review-stage/`、`review-input/`、`refine-logs/`、`idea-stage/`、`.aris/` 的大型或过程产物。

- [ ] **步骤 3：保护提交**

运行：

```bash
git commit -m "chore: preserve sawr working tree before mainline refactor"
```

预期：commit 成功。如果没有可提交内容，记录 `git status --short` 输出，并继续步骤 4。

- [ ] **步骤 4：创建专用 worktree**

运行：

```bash
git worktree add .worktrees/wfcllm-mainline-refactor -b codex/wfcllm-mainline-refactor sawr-minimal-generation-embedding
cd .worktrees/wfcllm-mainline-refactor
git branch --show-current
```

预期：

```text
codex/wfcllm-mainline-refactor
```

- [ ] **步骤 5：复制计划文件到 worktree 并提交计划**

如果计划文件不在 worktree 中，运行：

```bash
cp ../../docs/superpowers/plans/2026-07-08-wfcllm-mainline-refactor.md docs/superpowers/plans/2026-07-08-wfcllm-mainline-refactor.md
git add docs/superpowers/plans/2026-07-08-wfcllm-mainline-refactor.md
git commit -m "docs: add wfcllm mainline refactor plan"
```

预期：计划文件被跟踪，后续任务都在 `.worktrees/wfcllm-mainline-refactor` 中执行。

---

### 任务 1：新增 method preset 与 no-quality-gate 合同

**文件：**
- 创建：`wfcllm/method/__init__.py`
- 创建：`wfcllm/method/config.py`
- 创建：`wfcllm/method/presets.py`
- 创建：`wfcllm/method/contracts.py`
- 创建：`tests/method/test_evidence_retry_seed7x3_preset.py`
- 创建：`tests/method/test_no_quality_gate_contract.py`

- [ ] **步骤 1：编写 preset 失败测试**

创建 `tests/method/test_evidence_retry_seed7x3_preset.py`：

```python
from __future__ import annotations

import json

from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset


def test_evidence_retry_seed7x3_preset_matches_spec() -> None:
    preset = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME)

    assert preset.method["name"] == "evidence_retry_seed7x3"
    assert preset.method["strict_no_quality_gate"] is True
    assert preset.method["strict_code_only_detector"] is True
    assert preset.generation["seed"] == 7
    assert preset.generation["evidence_retry_attempts"] == 3
    assert preset.generation["evidence_retry_seed_stride"] == 101
    assert preset.generation["retry_repetition_penalty"] == 4.0
    assert preset.semantic_lsh["rule_name"] == "semantic_lsh"
    assert preset.semantic_lsh["lsh_d"] == 4
    assert preset.semantic_lsh["lsh_gamma"] == 0.25
    assert preset.detector["statistic"] == "calibrated_context_mean_proxy_penalized"
    assert preset.detector["proxy_penalty_alpha"] == 0.4


def test_default_preset_has_no_legacy_sections() -> None:
    payload = load_method_preset(EVIDENCE_RETRY_SEED7X3_NAME).to_dict()
    serialized = json.dumps(payload, sort_keys=True)

    forbidden = [
        "token_channel",
        "dual-channel",
        "adaptive_gamma",
        "watermark",
        "extract",
        "pretrain",
        "token_channel_train",
        "build_entropy_profile",
    ]
    for field in forbidden:
        assert field not in serialized
```

- [ ] **步骤 2：编写 no-quality-gate 失败测试**

创建 `tests/method/test_no_quality_gate_contract.py`：

```python
from __future__ import annotations

import pytest

from wfcllm.method.contracts import (
    evidence_retry_key,
    reject_quality_proxy_fields,
    require_diagnostic_marker,
)


class Result:
    accepted_hit_count = 2
    closed_without_hit_count = 1
    fallback_count = 3
    candidate_count = 8


def test_evidence_retry_key_is_exact_spec_tuple() -> None:
    assert evidence_retry_key(Result(), attempt_index=4) == (2, -1, -3, 8, -4)


@pytest.mark.parametrize(
    "field",
    [
        "pass",
        "passed",
        "correctness_result",
        "syntax_valid",
        "signature_compatible",
        "static_correctness_score",
        "generated_vs_canonical_length_gap",
        "checkpoint_correctness_proxy",
    ],
)
def test_reject_quality_proxy_fields_rejects_nested_forbidden_fields(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        reject_quality_proxy_fields({"sample": {"metadata": {field: True}}})


def test_require_diagnostic_marker_requires_both_marker_fields() -> None:
    require_diagnostic_marker(
        {
            "diagnostic_only": True,
            "not_official_method": True,
            "name": "evidence_only_selector_diagnostic",
        }
    )

    with pytest.raises(ValueError, match="diagnostic_only"):
        require_diagnostic_marker({"not_official_method": True})

    with pytest.raises(ValueError, match="not_official_method"):
        require_diagnostic_marker({"diagnostic_only": True})
```

- [ ] **步骤 3：运行测试验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method/test_evidence_retry_seed7x3_preset.py tests/method/test_no_quality_gate_contract.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'wfcllm.method'`。

- [ ] **步骤 4：创建 method package 最小实现**

创建 `wfcllm/method/config.py`：

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class WFCLLMMethodPreset:
    method: dict[str, Any]
    generation: dict[str, Any] = field(default_factory=dict)
    semantic_lsh: dict[str, Any] = field(default_factory=dict)
    detector: dict[str, Any] = field(default_factory=dict)
    calibration: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    runtime: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method.get("name") != "evidence_retry_seed7x3":
            raise ValueError("method.name must be evidence_retry_seed7x3")
        if self.method.get("strict_no_quality_gate") is not True:
            raise ValueError("method.strict_no_quality_gate must be true")
        if self.method.get("strict_code_only_detector") is not True:
            raise ValueError("method.strict_code_only_detector must be true")
        json.dumps(self.to_dict(), allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
```

创建 `wfcllm/method/presets.py`：

```python
from __future__ import annotations

from copy import deepcopy

from wfcllm.method.config import WFCLLMMethodPreset

EVIDENCE_RETRY_SEED7X3_NAME = "evidence_retry_seed7x3"

_EVIDENCE_RETRY_SEED7X3 = WFCLLMMethodPreset(
    method={
        "name": EVIDENCE_RETRY_SEED7X3_NAME,
        "strict_no_quality_gate": True,
        "strict_code_only_detector": True,
    },
    generation={
        "dataset": "humaneval",
        "max_new_tokens": 256,
        "temperature": 0.25,
        "top_p": 0.95,
        "top_k": 0,
        "retry_repetition_penalty": 4.0,
        "torch_dtype": "bf16",
        "device": "cuda",
        "seed": 7,
        "load_in_4bit": True,
        "prompt_mode": "completion",
        "max_group_statements": 2,
        "retry_budget": 2,
        "statement_retry_budget": 4,
        "window_retry_budget": 3,
        "compound_retry_budget": 2,
        "global_rollback_budget": 180,
        "max_total_sampled_tokens": 32768,
        "evidence_retry_attempts": 3,
        "evidence_retry_seed_stride": 101,
    },
    semantic_lsh={
        "rule_name": "semantic_lsh",
        "lsh_d": 4,
        "lsh_gamma": 0.25,
        "semantic_margin": 0.0,
        "use_ordinal_keying": False,
    },
    detector={
        "lsh_d": 4,
        "gamma": 0.25,
        "semantic_margin": 0.0,
        "max_group_statements": 2,
        "min_scoreable_contexts": 1,
        "min_proxy_windows": 2,
        "target_fpr": 0.05,
        "use_ordinal_keying": False,
        "evidence_mode": "hit_plus_margin",
        "statistic": "calibrated_context_mean_proxy_penalized",
        "proxy_penalty_alpha": 0.4,
    },
    calibration={},
    artifacts={"run_root": "data/runs"},
    runtime={"default_phases": ["generate", "calibrate", "detect", "report", "audit"]},
)


def load_method_preset(name: str) -> WFCLLMMethodPreset:
    if name != EVIDENCE_RETRY_SEED7X3_NAME:
        raise ValueError(f"unknown WFCLLM method preset: {name}")
    payload = deepcopy(_EVIDENCE_RETRY_SEED7X3.to_dict())
    return WFCLLMMethodPreset(**payload)
```

创建 `wfcllm/method/contracts.py`：

```python
from __future__ import annotations

from typing import Any

FORBIDDEN_QUALITY_GATE_FIELDS = {
    "hidden_tests",
    "public_doctests",
    "pass",
    "passed",
    "correctness_result",
    "ast_parse_success",
    "syntax_valid",
    "target_function_presence",
    "signature_compatible",
    "suspicious_tail",
    "truncation_heuristic",
    "duplicate_function_heuristic",
    "import_safety",
    "return_yield_presence",
    "prompt_contract_heuristic",
    "hardcoded_example_heuristic",
    "static_correctness_score",
    "complexity_score",
    "generated_vs_canonical_length_gap",
    "reference_correctness_proxy",
    "canonical_correctness_proxy",
    "logits_quality_proxy",
    "checkpoint_correctness_proxy",
}


def evidence_retry_key(result: Any, attempt_index: int) -> tuple[int, int, int, int, int]:
    return (
        int(result.accepted_hit_count),
        -int(result.closed_without_hit_count),
        -int(result.fallback_count),
        int(result.candidate_count),
        -int(attempt_index),
    )


def reject_quality_proxy_fields(value: Any) -> None:
    forbidden_path = _find_forbidden_quality_proxy(value, path="")
    if forbidden_path is not None:
        raise ValueError(f"quality proxy field is forbidden: {forbidden_path}")


def require_diagnostic_marker(payload: dict[str, Any]) -> None:
    if payload.get("diagnostic_only") is not True:
        raise ValueError("diagnostic_only must be true for diagnostic artifacts")
    if payload.get("not_official_method") is not True:
        raise ValueError("not_official_method must be true for diagnostic artifacts")


def _find_forbidden_quality_proxy(value: Any, *, path: str) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path else key_name
            if key_name in FORBIDDEN_QUALITY_GATE_FIELDS:
                return child_path
            found = _find_forbidden_quality_proxy(nested, path=child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found = _find_forbidden_quality_proxy(item, path=child_path)
            if found is not None:
                return found
    return None
```

创建 `wfcllm/method/__init__.py`：

```python
from __future__ import annotations

from wfcllm.method.config import WFCLLMMethodPreset
from wfcllm.method.contracts import (
    FORBIDDEN_QUALITY_GATE_FIELDS,
    evidence_retry_key,
    reject_quality_proxy_fields,
    require_diagnostic_marker,
)
from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME, load_method_preset

__all__ = [
    "EVIDENCE_RETRY_SEED7X3_NAME",
    "FORBIDDEN_QUALITY_GATE_FIELDS",
    "WFCLLMMethodPreset",
    "evidence_retry_key",
    "load_method_preset",
    "reject_quality_proxy_fields",
    "require_diagnostic_marker",
]
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method/test_evidence_retry_seed7x3_preset.py tests/method/test_no_quality_gate_contract.py -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add wfcllm/method tests/method
git commit -m "feat: add wfcllm method preset contracts"
```

---

### 任务 2：迁移 semantic 与 generation 核心模块，保留 SAWR shim

**文件：**
- 创建/移动：`wfcllm/generation/boundary.py`
- 创建/移动：`wfcllm/generation/state_machine.py`
- 创建/移动：`wfcllm/generation/generator.py`
- 创建：`wfcllm/generation/model_context.py`
- 创建：`wfcllm/generation/retry.py`
- 创建/移动：`wfcllm/semantic/rules.py`
- 创建/移动：`wfcllm/semantic/lsh.py`
- 创建/移动：`wfcllm/semantic/keying.py`
- 创建/移动：`wfcllm/semantic/lsh_space.py`
- 创建/移动：`wfcllm/semantic/verifier.py`
- 修改：`wfcllm/sawr/*.py`
- 测试：`tests/generation/test_boundary.py`
- 测试：`tests/generation/test_state_machine.py`
- 测试：`tests/generation/test_evidence_retry_key.py`
- 测试：`tests/semantic/test_rules.py`
- 测试：`tests/semantic/test_lsh.py`

- [ ] **步骤 1：迁移测试文件路径并改 imports**

运行：

```bash
mkdir -p tests/generation tests/semantic
git mv tests/sawr/test_boundary.py tests/generation/test_boundary.py
git mv tests/sawr/test_state_machine.py tests/generation/test_state_machine.py
git mv tests/sawr/test_rules.py tests/semantic/test_rules.py
```

修改 imports：

```bash
perl -0pi -e 's/wfcllm\.sawr\.boundary/wfcllm.generation.boundary/g; s/wfcllm\.sawr\.state_machine/wfcllm.generation.state_machine/g; s/wfcllm\.sawr\.rules/wfcllm.semantic.rules/g' tests/generation/test_boundary.py tests/generation/test_state_machine.py tests/semantic/test_rules.py
```

创建 `tests/generation/test_evidence_retry_key.py`：

```python
from __future__ import annotations

from wfcllm.generation.retry import evidence_retry_key


class Result:
    accepted_hit_count = 1
    closed_without_hit_count = 2
    fallback_count = 3
    candidate_count = 4


def test_evidence_retry_key_matches_method_contract() -> None:
    assert evidence_retry_key(Result(), attempt_index=5) == (1, -2, -3, 4, -5)
```

创建 `tests/semantic/test_lsh.py`：

```python
from __future__ import annotations


def test_semantic_lsh_module_exports_loader() -> None:
    from wfcllm.semantic.lsh import load_semantic_lsh_rule

    assert callable(load_semantic_lsh_rule)
```

- [ ] **步骤 2：运行迁移测试验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_boundary.py tests/generation/test_state_machine.py tests/generation/test_evidence_retry_key.py tests/semantic/test_rules.py tests/semantic/test_lsh.py -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'wfcllm.generation'` 或 `No module named 'wfcllm.semantic'`。

- [ ] **步骤 3：移动正式模块**

运行：

```bash
mkdir -p wfcllm/generation wfcllm/semantic
git mv wfcllm/sawr/boundary.py wfcllm/generation/boundary.py
git mv wfcllm/sawr/state_machine.py wfcllm/generation/state_machine.py
git mv wfcllm/sawr/generator.py wfcllm/generation/generator.py
git mv wfcllm/sawr/rules.py wfcllm/semantic/rules.py
git mv wfcllm/sawr/semantic_lsh.py wfcllm/semantic/lsh.py
git mv wfcllm/watermark/keying.py wfcllm/semantic/keying.py
git mv wfcllm/watermark/lsh_space.py wfcllm/semantic/lsh_space.py
git mv wfcllm/watermark/verifier.py wfcllm/semantic/verifier.py
git mv wfcllm/watermark/kv_cache.py wfcllm/generation/kv_cache.py
```

替换新模块内部 imports：

```bash
perl -0pi -e 's/wfcllm\.sawr\.boundary/wfcllm.generation.boundary/g; s/wfcllm\.sawr\.state_machine/wfcllm.generation.state_machine/g; s/wfcllm\.sawr\.generator/wfcllm.generation.generator/g; s/wfcllm\.sawr\.config/wfcllm.method.config/g; s/wfcllm\.sawr\.rules/wfcllm.semantic.rules/g; s/wfcllm\.sawr\.semantic_lsh/wfcllm.semantic.lsh/g; s/wfcllm\.watermark\.keying/wfcllm.semantic.keying/g; s/wfcllm\.watermark\.lsh_space/wfcllm.semantic.lsh_space/g; s/wfcllm\.watermark\.verifier/wfcllm.semantic.verifier/g; s/wfcllm\.watermark\.kv_cache/wfcllm.generation.kv_cache/g' wfcllm/generation/*.py wfcllm/semantic/*.py
```

- [ ] **步骤 4：创建 package exports 和 retry helper**

创建 `wfcllm/generation/retry.py`：

```python
from __future__ import annotations

from typing import Any


def evidence_retry_key(result: Any, attempt_index: int) -> tuple[int, int, int, int, int]:
    return (
        int(result.accepted_hit_count),
        -int(result.closed_without_hit_count),
        -int(result.fallback_count),
        int(result.candidate_count),
        -int(attempt_index),
    )
```

创建 `wfcllm/generation/__init__.py`：

```python
from __future__ import annotations

from wfcllm.generation.boundary import BoundaryEvent, BoundaryEventKind, Candidate
from wfcllm.generation.retry import evidence_retry_key
from wfcllm.generation.state_machine import AuditEvent, SawrStateMachine

WFCLLMStateMachine = SawrStateMachine

__all__ = [
    "AuditEvent",
    "BoundaryEvent",
    "BoundaryEventKind",
    "Candidate",
    "SawrStateMachine",
    "WFCLLMStateMachine",
    "evidence_retry_key",
]
```

创建 `wfcllm/semantic/__init__.py`：

```python
from __future__ import annotations

from wfcllm.semantic.rules import EmbeddingRule, HashEmbeddingRule, SemanticLshEmbeddingRule
from wfcllm.semantic.lsh import load_semantic_lsh_rule

__all__ = [
    "EmbeddingRule",
    "HashEmbeddingRule",
    "SemanticLshEmbeddingRule",
    "load_semantic_lsh_rule",
]
```

- [ ] **步骤 5：把 `wfcllm/sawr` 改成 shim**

为每个 shim 模块使用同一模式。`wfcllm/sawr/boundary.py` 内容：

```python
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.boundary is deprecated; import wfcllm.generation.boundary instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.boundary import *  # noqa: F401,F403,E402
```

按同样模式创建这些文件，目标模块逐一对应：

```text
wfcllm/sawr/state_machine.py -> wfcllm.generation.state_machine
wfcllm/sawr/generator.py -> wfcllm.generation.generator
wfcllm/sawr/rules.py -> wfcllm.semantic.rules
wfcllm/sawr/semantic_lsh.py -> wfcllm.semantic.lsh
```

`wfcllm/sawr/config.py` 先写为 method config re-export，任务 3 再完成正式 config 重命名：

```python
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.config is deprecated; import wfcllm.method.config instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.method.config import *  # noqa: F401,F403,E402
```

- [ ] **步骤 6：运行测试验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_boundary.py tests/generation/test_state_machine.py tests/generation/test_evidence_retry_key.py tests/semantic/test_rules.py tests/semantic/test_lsh.py -v
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add wfcllm/generation wfcllm/semantic wfcllm/sawr tests/generation tests/semantic
git commit -m "refactor: promote sawr core to wfcllm generation"
```

---

### 任务 3：实现 official final-code artifacts 与新 generation pipeline

**文件：**
- 创建：`wfcllm/generation/outputs.py`
- 创建/移动：`wfcllm/generation/pipeline.py`
- 修改：`wfcllm/method/artifacts.py`
- 修改：`wfcllm/method/config.py`
- 测试：`tests/generation/test_pipeline.py`
- 测试：`tests/method/test_artifacts.py`

- [ ] **步骤 1：迁移 pipeline 测试并改成四字段预期**

运行：

```bash
git mv tests/sawr/test_pipeline.py tests/generation/test_pipeline.py
perl -0pi -e 's/wfcllm\.sawr\.config/wfcllm.method.config/g; s/wfcllm\.sawr\.generator/wfcllm.generation.generator/g; s/wfcllm\.sawr\.pipeline/wfcllm.generation.pipeline/g; s/wfcllm\.sawr\.state_machine/wfcllm.generation.state_machine/g; s/SawrGenerationConfig/WFCLLMGenerationConfig/g; s/SawrPipelineConfig/WFCLLMPipelineConfig/g; s/SawrRuleConfig/WFCLLMRuleConfig/g; s/SawrPipeline/WFCLLMGenerationPipeline/g' tests/generation/test_pipeline.py
```

在 `tests/generation/test_pipeline.py` 中把 final row 断言改成：

```python
assert set(row) == {"id", "dataset", "prompt", "final_code"}
assert row["id"] == "HumanEval/0"
assert row["dataset"] == "humaneval"
assert row["prompt"] == "def foo():\n"
assert row["final_code"] == "def foo():\n    return 1\n"
```

新增断言，确保 sidecar marker 不进入 final row：

```python
assert "artifact_type" not in row
assert "schema_version" not in row
assert "scientific_claims_enabled" not in row
```

- [ ] **步骤 2：编写 artifact helper 失败测试**

创建 `tests/method/test_artifacts.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.method.artifacts import (
    FinalCodeRecord,
    RunPaths,
    build_run_paths,
    write_jsonl_rows,
)


def test_final_code_record_exact_four_fields() -> None:
    record = FinalCodeRecord(
        id="HumanEval/0",
        dataset="humaneval",
        prompt="def foo():\n",
        final_code="def foo():\n    return 1\n",
    )

    assert record.to_dict() == {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }


def test_final_code_record_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="id"):
        FinalCodeRecord(id="", dataset="humaneval", prompt="", final_code="")


def test_build_run_paths_uses_new_layout(tmp_path: Path) -> None:
    paths = build_run_paths(tmp_path, "20260708_153000_evidence_retry_seed7x3")

    assert isinstance(paths, RunPaths)
    assert paths.final_code_input == tmp_path / "20260708_153000_evidence_retry_seed7x3" / "inputs" / "final_code.jsonl"
    assert paths.audit_log == tmp_path / "20260708_153000_evidence_retry_seed7x3" / "generation" / "audit.jsonl"
    assert paths.detector_integrity == tmp_path / "20260708_153000_evidence_retry_seed7x3" / "audit" / "detector_input_integrity.json"


def test_write_jsonl_rows_rejects_non_json_numbers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON-safe"):
        write_jsonl_rows(tmp_path / "rows.jsonl", [{"score": float("nan")}])


def test_write_jsonl_rows_writes_utf8_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl_rows(path, [{"id": "a", "text": "中文"}])

    assert json.loads(path.read_text(encoding="utf-8").strip()) == {"id": "a", "text": "中文"}
```

- [ ] **步骤 3：运行测试验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method/test_artifacts.py tests/generation/test_pipeline.py -v
```

预期：FAIL，包含 `ModuleNotFoundError: No module named 'wfcllm.method.artifacts'` 或 final row 多出 `artifact_type/schema_version/scientific_claims_enabled`。

- [ ] **步骤 4：实现 artifact helper**

创建 `wfcllm/method/artifacts.py`：

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class FinalCodeRecord:
    id: str
    dataset: str
    prompt: str
    final_code: str

    def __post_init__(self) -> None:
        for field_name in ("id", "dataset", "prompt", "final_code"):
            value = getattr(self, field_name)
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a string")
        if not self.id:
            raise ValueError("id must be a non-empty string")

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "prompt": self.prompt,
            "final_code": self.final_code,
        }


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    resolved_config: Path
    method_preset: Path
    final_code_input: Path
    audit_log: Path
    candidate_sidecar: Path
    raw_attempt_summary: Path
    reference_calibration: Path
    positive_details: Path
    reference_negative_details: Path
    model_negative_details: Path
    reference_report: Path
    model_negative_report: Path
    pass_report_posthoc: Path
    detector_integrity: Path
    no_quality_gate_integrity: Path
    artifact_integrity: Path
    run_log: Path


def build_run_paths(run_root: str | Path, run_id: str) -> RunPaths:
    run_dir = Path(run_root) / run_id
    return RunPaths(
        run_dir=run_dir,
        resolved_config=run_dir / "config" / "resolved_config.json",
        method_preset=run_dir / "config" / "method_preset.json",
        final_code_input=run_dir / "inputs" / "final_code.jsonl",
        audit_log=run_dir / "generation" / "audit.jsonl",
        candidate_sidecar=run_dir / "generation" / "candidate_sidecar.jsonl",
        raw_attempt_summary=run_dir / "generation" / "raw_attempt_summary.jsonl",
        reference_calibration=run_dir / "calibration" / "reference_calibration.json",
        positive_details=run_dir / "detection" / "positive_details.jsonl",
        reference_negative_details=run_dir / "detection" / "reference_negative_details.jsonl",
        model_negative_details=run_dir / "detection" / "model_negative_details.jsonl",
        reference_report=run_dir / "reports" / "reference_report.json",
        model_negative_report=run_dir / "reports" / "model_negative_report.json",
        pass_report_posthoc=run_dir / "reports" / "pass_report_posthoc.json",
        detector_integrity=run_dir / "audit" / "detector_input_integrity.json",
        no_quality_gate_integrity=run_dir / "audit" / "no_quality_gate_integrity.json",
        artifact_integrity=run_dir / "audit" / "artifact_integrity.json",
        run_log=run_dir / "logs" / "run.log",
    )


def write_jsonl_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            try:
                handle.write(json.dumps(row, allow_nan=False, ensure_ascii=False) + "\n")
            except (TypeError, ValueError) as exc:
                raise ValueError("JSONL rows must be JSON-safe") from exc
    return str(output_path)
```

- [ ] **步骤 5：迁移 config dataclasses 并改名**

在 `wfcllm/method/config.py` 中保留任务 1 的 `WFCLLMMethodPreset`，并从原 `wfcllm/sawr/config.py` 迁移这些 dataclass，使用新名称：

```python
WFCLLMGenerationConfig
WFCLLMRuleConfig
WFCLLMPipelineConfig
```

保留旧别名只给 shim 使用：

```python
SawrGenerationConfig = WFCLLMGenerationConfig
SawrRuleConfig = WFCLLMRuleConfig
SawrPipelineConfig = WFCLLMPipelineConfig
```

把 `_ALLOWED_RULE_NAMES` 保持为：

```python
_ALLOWED_RULE_NAMES = ("hash", "semantic_lsh")
```

`WFCLLMPipelineConfig.__post_init__` 中新增对 `candidate_sidecar_output` 和 `run_id` 的类型检查：

```python
if self.candidate_sidecar_output is not None and not isinstance(self.candidate_sidecar_output, str):
    raise ValueError("candidate_sidecar_output must be a string or None")
if self.run_id is not None and (not isinstance(self.run_id, str) or not self.run_id):
    raise ValueError("run_id must be a non-empty string or None")
```

- [ ] **步骤 6：迁移 generation pipeline**

运行：

```bash
git mv wfcllm/sawr/pipeline.py wfcllm/generation/pipeline.py
perl -0pi -e 's/SawrPipelineConfig/WFCLLMPipelineConfig/g; s/SawrPipeline/WFCLLMGenerationPipeline/g; s/SawrGenerateResult/WFCLLMGenerateResult/g; s/SawrGenerator/WFCLLMGenerator/g; s/wfcllm\.sawr\.config/wfcllm.method.config/g; s/wfcllm\.sawr\.generator/wfcllm.generation.generator/g; s/wfcllm\.sawr\.state_machine/wfcllm.generation.state_machine/g' wfcllm/generation/pipeline.py
```

在 `wfcllm/generation/pipeline.py` 中改 `_build_final_row` 为：

```python
from wfcllm.method.artifacts import FinalCodeRecord


def _build_final_row(
    self,
    sample_id: str,
    prompt: str,
    result: WFCLLMGenerateResult,
) -> dict[str, str]:
    return FinalCodeRecord(
        id=sample_id,
        dataset=self._config.dataset,
        prompt=prompt,
        final_code=result.final_code,
    ).to_dict()
```

把 `_evidence_retry_key` 改成调用正式 helper：

```python
from wfcllm.generation.retry import evidence_retry_key


key = evidence_retry_key(result, attempt_index)
```

确保 `FORBIDDEN_FINAL_FIELDS` 至少包含：

```python
FORBIDDEN_FINAL_FIELDS = {
    "artifact_type",
    "schema_version",
    "generated_code",
    "watermark_params",
    "blocks",
    "audit",
    "audit_only",
    "detector_input_allowed",
    "generation_ledger",
    "retry_ledger",
    "p_value",
    "z_score",
    "detector_score",
    "score",
    "is_watermarked",
    "pass",
    "passed",
    "correctness_result",
    "scientific_claims_enabled",
}
```

- [ ] **步骤 7：更新 generator 正式名称但保留兼容别名**

在 `wfcllm/generation/generator.py` 中把公开类新增别名：

```python
WFCLLMCheckpoint = SawrCheckpoint
WFCLLMGenerateResult = SawrGenerateResult
WFCLLMGenerator = SawrGenerator
WFCLLMModelContext = SawrModelContext
load_wfcllm_model = load_sawr_model
```

如果执行者选择彻底改名，必须同时更新所有 tests 中的 import；保留别名可以降低一次性改动风险。

- [ ] **步骤 8：运行测试验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method/test_artifacts.py tests/generation/test_pipeline.py tests/generation/test_evidence_retry_key.py -v
```

预期：PASS。

- [ ] **步骤 9：Commit**

```bash
git add wfcllm/method wfcllm/generation tests/method tests/generation wfcllm/sawr
git commit -m "feat: write strict wfcllm generation artifacts"
```

---

### 任务 4：迁移 detection 并强制 strict final-code-only input

**文件：**
- 创建/移动：`wfcllm/detection/config.py`
- 创建：`wfcllm/detection/code_only.py`
- 创建/移动：`wfcllm/detection/proxy_windows.py`
- 创建/移动：`wfcllm/detection/scoring.py`
- 创建/移动：`wfcllm/detection/calibration.py`
- 创建/移动：`wfcllm/detection/pipeline.py`
- 创建/移动：`wfcllm/detection/metrics.py`
- 创建：`wfcllm/detection/reports.py`
- 修改：`wfcllm/sawr/detect/*.py`
- 测试：`tests/detection/test_code_only_contract.py`
- 测试：`tests/detection/test_detector_rejects_generated_code.py`
- 测试：`tests/detection/test_pipeline.py`

- [ ] **步骤 1：移动 detector 测试并新增 strict contract 测试**

运行：

```bash
mkdir -p tests/detection
git mv tests/sawr/detect/test_config.py tests/detection/test_config.py
git mv tests/sawr/detect/test_calibration.py tests/detection/test_calibration.py
git mv tests/sawr/detect/test_metrics.py tests/detection/test_metrics.py
git mv tests/sawr/detect/test_pipeline.py tests/detection/test_pipeline.py
git mv tests/sawr/detect/test_proxy_windows.py tests/detection/test_proxy_windows.py
git mv tests/sawr/detect/test_scoring.py tests/detection/test_scoring.py
perl -0pi -e 's/wfcllm\.sawr\.detect/wfcllm.detection/g; s/SawrDetectionConfig/WFCLLMDetectionConfig/g; s/SawrDetectionPipeline/WFCLLMDetectionPipeline/g; s/SawrDetectionResult/WFCLLMDetectionResult/g; s/SawrWindowScorer/WFCLLMWindowScorer/g; s/load_sawr_window_scorer/load_wfcllm_window_scorer/g' tests/detection/*.py
```

创建 `tests/detection/test_code_only_contract.py`：

```python
from __future__ import annotations

import pytest

from wfcllm.detection.code_only import (
    OFFICIAL_DETECTOR_INPUT_FIELDS,
    code_from_record,
    validate_final_code_record_exact,
)


def test_validate_final_code_record_exact_accepts_only_four_fields() -> None:
    record = {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }

    validate_final_code_record_exact(record)
    assert set(record) == OFFICIAL_DETECTOR_INPUT_FIELDS
    assert code_from_record(record) == "def foo():\n    return 1\n"


@pytest.mark.parametrize(
    "extra_field",
    [
        "artifact_type",
        "schema_version",
        "generated_code",
        "watermark_params",
        "blocks",
        "audit",
        "audit_only",
        "detector_input_allowed",
        "generation_trace",
        "retry_trace",
        "rollback_trace",
        "sampling_trace",
        "candidate_sidecar",
        "p_value",
        "z_score",
        "detector_score",
        "score",
        "is_watermarked",
        "pass",
        "passed",
        "correctness_result",
    ],
)
def test_validate_final_code_record_exact_rejects_extra_fields(extra_field: str) -> None:
    record = {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
        extra_field: "blocked",
    }

    with pytest.raises(ValueError, match=extra_field):
        validate_final_code_record_exact(record)


def test_code_from_record_rejects_generated_code_fallback() -> None:
    with pytest.raises(ValueError, match="exactly"):
        code_from_record(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "generated_code": "    return 1\n",
            }
        )
```

创建 `tests/detection/test_detector_rejects_generated_code.py`：

```python
from __future__ import annotations

import pytest

from wfcllm.detection.pipeline import validate_final_code_detector_input_record


def test_official_detector_rejects_generated_code_even_with_prompt() -> None:
    with pytest.raises(ValueError, match="generated_code"):
        validate_final_code_detector_input_record(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def target():\n",
                "generated_code": "    return 1\n",
            }
        )
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection/test_code_only_contract.py tests/detection/test_detector_rejects_generated_code.py tests/detection/test_pipeline.py -v
```

预期：FAIL，当前 detector 仍允许 `generated_code` fallback 或 `artifact_type` metadata。

- [ ] **步骤 3：移动 detector implementation**

运行：

```bash
mkdir -p wfcllm/detection
git mv wfcllm/sawr/detect/config.py wfcllm/detection/config.py
git mv wfcllm/sawr/detect/calibration.py wfcllm/detection/calibration.py
git mv wfcllm/sawr/detect/metrics.py wfcllm/detection/metrics.py
git mv wfcllm/sawr/detect/pipeline.py wfcllm/detection/pipeline.py
git mv wfcllm/sawr/detect/proxy_windows.py wfcllm/detection/proxy_windows.py
git mv wfcllm/sawr/detect/scoring.py wfcllm/detection/scoring.py
perl -0pi -e 's/wfcllm\.sawr\.detect/wfcllm.detection/g; s/wfcllm\.sawr\.rules/wfcllm.semantic.rules/g; s/wfcllm\.sawr\.semantic_lsh/wfcllm.semantic.lsh/g; s/SawrDetectionConfig/WFCLLMDetectionConfig/g; s/SawrDetectionPipeline/WFCLLMDetectionPipeline/g; s/SawrDetectionResult/WFCLLMDetectionResult/g; s/SawrWindowScorer/WFCLLMWindowScorer/g; s/load_sawr_window_scorer/load_wfcllm_window_scorer/g; s/sawr-structure-aware-proxy-window\/v1/wfcllm-structure-aware-proxy-window\/v1/g' wfcllm/detection/*.py
```

- [ ] **步骤 4：实现 exact four-field validator**

创建 `wfcllm/detection/code_only.py`：

```python
from __future__ import annotations

from typing import Any

OFFICIAL_DETECTOR_INPUT_FIELDS = {"id", "dataset", "prompt", "final_code"}
SUPPORTED_OFFICIAL_DATASETS = {"humaneval", "mbpp"}

FORBIDDEN_DETECTOR_INPUT_FIELDS = {
    "artifact_type",
    "schema_version",
    "generated_code",
    "watermark_params",
    "blocks",
    "audit",
    "audit_only",
    "detector_input_allowed",
    "logits",
    "checkpoint",
    "retry_trace",
    "rollback_trace",
    "sampling_trace",
    "candidate_sidecar",
    "p_value",
    "z_score",
    "detector_score",
    "score",
    "is_watermarked",
    "pass",
    "passed",
    "correctness_result",
}
FORBIDDEN_DETECTOR_INPUT_PREFIXES = ("generation_", "audit_")


def validate_final_code_record_exact(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("detector input record must be a mapping")
    actual_fields = set(record)
    if actual_fields != OFFICIAL_DETECTOR_INPUT_FIELDS:
        extra = sorted(actual_fields - OFFICIAL_DETECTOR_INPUT_FIELDS)
        missing = sorted(OFFICIAL_DETECTOR_INPUT_FIELDS - actual_fields)
        if extra:
            raise ValueError(f"detector input must have exactly four fields; forbidden extra fields: {extra}")
        raise ValueError(f"detector input must have exactly four fields; missing fields: {missing}")
    sample_id = record["id"]
    dataset = record["dataset"]
    prompt = record["prompt"]
    final_code = record["final_code"]
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("id must be a non-empty string")
    if not isinstance(dataset, str):
        raise ValueError("dataset must be a string")
    if dataset not in SUPPORTED_OFFICIAL_DATASETS:
        raise ValueError(f"dataset must be one of {sorted(SUPPORTED_OFFICIAL_DATASETS)}")
    if not isinstance(prompt, str):
        raise ValueError("prompt must be a string")
    if not isinstance(final_code, str):
        raise ValueError("final_code must be a string")


def reject_forbidden_detector_output_fields(payload: dict[str, Any]) -> None:
    forbidden = FORBIDDEN_DETECTOR_INPUT_FIELDS & set(payload)
    if forbidden:
        raise ValueError(f"forbidden detector output fields: {sorted(forbidden)}")


def code_from_record(record: dict[str, Any]) -> str:
    validate_final_code_record_exact(record)
    return record["final_code"]
```

- [ ] **步骤 5：wire validator into detection pipeline**

在 `wfcllm/detection/pipeline.py` 中替换 `code_from_record` 和 `validate_final_code_detector_input_record` 实现：

```python
from wfcllm.detection.code_only import (
    code_from_record,
    validate_final_code_record_exact,
)


def validate_final_code_detector_input_record(record: dict[str, Any]) -> None:
    validate_final_code_record_exact(record)
```

删除 `_join_prompt_and_generated_code` 的 production 路径；如果保留函数，只能放在 `archive/legacy_wfcllm_2026_07` 或 test fixture，不得被 official detector 调用。

- [ ] **步骤 6：创建 detection package exports 和 sawr detect shims**

创建 `wfcllm/detection/__init__.py`，导出当前 tests 使用的正式名称：

```python
from __future__ import annotations

from wfcllm.detection.code_only import code_from_record, validate_final_code_record_exact
from wfcllm.detection.config import DETECTOR_MODE, BucketEdges, WFCLLMDetectionConfig
from wfcllm.detection.pipeline import (
    FORBIDDEN_DETECTOR_OUTPUT_FIELDS,
    ContextDetectionSummary,
    WFCLLMDetectionPipeline,
    WFCLLMDetectionResult,
    load_jsonl_records,
    validate_final_code_detector_input_record,
)

__all__ = [
    "BucketEdges",
    "ContextDetectionSummary",
    "DETECTOR_MODE",
    "FORBIDDEN_DETECTOR_OUTPUT_FIELDS",
    "WFCLLMDetectionConfig",
    "WFCLLMDetectionPipeline",
    "WFCLLMDetectionResult",
    "code_from_record",
    "load_jsonl_records",
    "validate_final_code_detector_input_record",
    "validate_final_code_record_exact",
]
```

为 `wfcllm/sawr/detect/*.py` 创建 shim，模式同任务 2。`wfcllm/sawr/detect/pipeline.py` 示例：

```python
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.sawr.detect.pipeline is deprecated; import wfcllm.detection.pipeline instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.detection.pipeline import *  # noqa: F401,F403,E402
```

- [ ] **步骤 7：运行检测测试验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
```

预期：PASS。若迁移后的旧测试仍断言 `generated_code` 被接受，删除该断言并用 `tests/detection/test_detector_rejects_generated_code.py` 的行为替代。

- [ ] **步骤 8：Commit**

```bash
git add wfcllm/detection wfcllm/sawr/detect tests/detection
git commit -m "feat: enforce strict final-code detector"
```

---

### 任务 5：实现 audit integrity checks

**文件：**
- 创建：`wfcllm/audit/__init__.py`
- 创建：`wfcllm/audit/forbidden.py`
- 创建：`wfcllm/audit/no_quality_gate.py`
- 创建：`wfcllm/audit/detector_input_integrity.py`
- 创建：`wfcllm/audit/artifact_integrity.py`
- 创建：`tests/audit/test_detector_input_integrity.py`
- 创建：`tests/audit/test_no_quality_gate_integrity.py`
- 创建：`tests/audit/test_artifact_integrity.py`

- [ ] **步骤 1：编写 detector input integrity 失败测试**

创建 `tests/audit/test_detector_input_integrity.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.audit.detector_input_integrity import audit_detector_input_file


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_audit_detector_input_file_accepts_four_field_rows(tmp_path: Path) -> None:
    path = tmp_path / "final_code.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "final_code": "def foo():\n    return 1\n",
            }
        ],
    )

    report = audit_detector_input_file(path)

    assert report["ok"] is True
    assert report["records_checked"] == 1
    assert report["errors"] == []


def test_audit_detector_input_file_reports_generated_code(tmp_path: Path) -> None:
    path = tmp_path / "final_code.jsonl"
    _write_jsonl(
        path,
        [
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "generated_code": "    return 1\n",
            }
        ],
    )

    report = audit_detector_input_file(path)

    assert report["ok"] is False
    assert report["records_checked"] == 1
    assert "generated_code" in report["errors"][0]["error"]
```

- [ ] **步骤 2：编写 no-quality-gate 和 artifact integrity 失败测试**

创建 `tests/audit/test_no_quality_gate_integrity.py`：

```python
from __future__ import annotations

import pytest

from wfcllm.audit.no_quality_gate import audit_no_quality_gate_payload


def test_audit_no_quality_gate_payload_accepts_evidence_counts() -> None:
    report = audit_no_quality_gate_payload(
        {
            "accepted_hit_count": 2,
            "closed_without_hit_count": 1,
            "fallback_count": 0,
            "candidate_count": 5,
        }
    )

    assert report == {"ok": True, "forbidden_path": None}


def test_audit_no_quality_gate_payload_rejects_nested_pass_field() -> None:
    report = audit_no_quality_gate_payload({"attempt": {"pass": True}})

    assert report["ok"] is False
    assert report["forbidden_path"] == "attempt.pass"
```

创建 `tests/audit/test_artifact_integrity.py`：

```python
from __future__ import annotations

import pytest

from wfcllm.audit.artifact_integrity import (
    assert_audit_only_marker,
    assert_posthoc_pass_report_marker,
    reject_secret_key_leak,
)


def test_assert_audit_only_marker_requires_detector_input_false() -> None:
    assert_audit_only_marker({"audit_only": True, "detector_input_allowed": False})

    with pytest.raises(ValueError, match="detector_input_allowed"):
        assert_audit_only_marker({"audit_only": True, "detector_input_allowed": True})


def test_assert_posthoc_pass_report_marker_requires_all_flags() -> None:
    assert_posthoc_pass_report_marker(
        {
            "posthoc_only": True,
            "not_used_for_generation": True,
            "not_used_for_retry": True,
            "not_used_for_selection": True,
            "not_used_for_calibration": True,
            "not_used_for_detection": True,
        }
    )

    with pytest.raises(ValueError, match="not_used_for_detection"):
        assert_posthoc_pass_report_marker({"posthoc_only": True})


def test_reject_secret_key_leak_finds_nested_secret_key() -> None:
    with pytest.raises(ValueError, match="secret_key"):
        reject_secret_key_leak({"calibration": {"secret_key": "1010"}})
```

- [ ] **步骤 3：运行测试验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'wfcllm.audit'`。

- [ ] **步骤 4：实现 audit modules**

创建 `wfcllm/audit/forbidden.py`：

```python
from __future__ import annotations

FORBIDDEN_SECRET_FIELDS = {"secret_key", "raw_secret_key"}
POSTHOC_PASS_REQUIRED_FLAGS = {
    "posthoc_only",
    "not_used_for_generation",
    "not_used_for_retry",
    "not_used_for_selection",
    "not_used_for_calibration",
    "not_used_for_detection",
}
```

创建 `wfcllm/audit/no_quality_gate.py`：

```python
from __future__ import annotations

from typing import Any

from wfcllm.method.contracts import reject_quality_proxy_fields


def audit_no_quality_gate_payload(payload: Any) -> dict[str, object]:
    try:
        reject_quality_proxy_fields(payload)
    except ValueError as exc:
        message = str(exc)
        marker = "quality proxy field is forbidden: "
        return {
            "ok": False,
            "forbidden_path": message.split(marker, 1)[1] if marker in message else message,
        }
    return {"ok": True, "forbidden_path": None}
```

创建 `wfcllm/audit/detector_input_integrity.py`：

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wfcllm.detection.code_only import validate_final_code_record_exact


def audit_detector_input_file(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    errors: list[dict[str, object]] = []
    records_checked = 0
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        records_checked += 1
        try:
            record = json.loads(line)
            validate_final_code_record_exact(record)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append({"line": line_number, "error": str(exc)})
    return {"ok": not errors, "records_checked": records_checked, "errors": errors}
```

创建 `wfcllm/audit/artifact_integrity.py`：

```python
from __future__ import annotations

from typing import Any

from wfcllm.audit.forbidden import FORBIDDEN_SECRET_FIELDS, POSTHOC_PASS_REQUIRED_FLAGS


def assert_audit_only_marker(payload: dict[str, Any]) -> None:
    if payload.get("audit_only") is not True:
        raise ValueError("audit_only must be true")
    if payload.get("detector_input_allowed") is not False:
        raise ValueError("detector_input_allowed must be false")


def assert_posthoc_pass_report_marker(payload: dict[str, Any]) -> None:
    for field in sorted(POSTHOC_PASS_REQUIRED_FLAGS):
        if payload.get(field) is not True:
            raise ValueError(f"{field} must be true")


def reject_secret_key_leak(payload: Any) -> None:
    path = _find_secret_path(payload, path="")
    if path is not None:
        raise ValueError(f"secret material is forbidden in public artifacts: {path}")


def _find_secret_path(value: Any, *, path: str) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path else key_name
            if key_name in FORBIDDEN_SECRET_FIELDS:
                return child_path
            found = _find_secret_path(nested, path=child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found = _find_secret_path(item, path=child_path)
            if found is not None:
                return found
    return None
```

创建 `wfcllm/audit/__init__.py`：

```python
from __future__ import annotations

from wfcllm.audit.artifact_integrity import (
    assert_audit_only_marker,
    assert_posthoc_pass_report_marker,
    reject_secret_key_leak,
)
from wfcllm.audit.detector_input_integrity import audit_detector_input_file
from wfcllm.audit.no_quality_gate import audit_no_quality_gate_payload

__all__ = [
    "assert_audit_only_marker",
    "assert_posthoc_pass_report_marker",
    "audit_detector_input_file",
    "audit_no_quality_gate_payload",
    "reject_secret_key_leak",
]
```

- [ ] **步骤 5：运行测试验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
```

预期：PASS。

- [ ] **步骤 6：Commit**

```bash
git add wfcllm/audit tests/audit
git commit -m "feat: add wfcllm integrity audits"
```

---

### 任务 6：切换 CLI、phase state 和 base config 到新版 WFCLLM

**文件：**
- 修改：`configs/base_config.json`
- 创建：`configs/wfcllm/evidence_retry_seed7x3.json`
- 创建：`configs/wfcllm/detector_proxy_penalized_alpha04.json`
- 创建：`configs/wfcllm/repro_humaneval_full164.json`
- 创建：`configs/wfcllm/diagnostics/evidence_only_selector_diagnostic.json`
- 修改：`wfcllm/orchestration/state.py`
- 修改：`wfcllm/orchestration/pipeline.py`
- 修改：`wfcllm/cli/arguments.py`
- 修改：`wfcllm/cli/config_resolver.py`
- 修改：`wfcllm/cli/runners.py`
- 修改：`wfcllm/cli/entry.py`
- 测试：`tests/integration/test_default_cli_uses_new_wfcllm.py`
- 测试：`tests/integration/test_legacy_phase_requires_legacy_flag.py`

- [ ] **步骤 1：编写 CLI phase 失败测试**

创建 `tests/integration/test_default_cli_uses_new_wfcllm.py`：

```python
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.entry import _populate_phase_registry
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.state import PHASES


def test_default_phase_list_is_new_wfcllm_mainline() -> None:
    assert PHASES == ["generate", "calibrate", "detect", "report", "audit"]


def test_parser_accepts_new_official_phases() -> None:
    parser = build_parser()

    for phase in ["generate", "calibrate", "detect", "report", "audit", "posthoc-pass-report", "diagnostic-selector"]:
        args = parser.parse_args(["--phase", phase])
        assert args.phase == phase


def test_parser_does_not_expose_allow_quality_proxy() -> None:
    parser = build_parser()

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--allow-quality-proxy" not in option_strings


def test_phase_registry_registers_new_mainline_runners(monkeypatch) -> None:
    registry = PhaseRegistry()
    _populate_phase_registry(registry)

    for phase in ["generate", "calibrate", "detect", "report", "audit"]:
        assert callable(registry.get(phase))
```

创建 `tests/integration/test_legacy_phase_requires_legacy_flag.py`：

```python
from __future__ import annotations

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.entry import validate_legacy_phase_request


def test_legacy_phase_requires_legacy_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--phase", "legacy-watermark"])

    assert validate_legacy_phase_request(args) == "[错误] legacy phase requires --legacy"


def test_legacy_phase_allows_explicit_legacy_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--phase", "legacy-watermark", "--legacy"])

    assert validate_legacy_phase_request(args) is None
```

- [ ] **步骤 2：运行 CLI tests 验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_default_cli_uses_new_wfcllm.py tests/integration/test_legacy_phase_requires_legacy_flag.py -v
```

预期：FAIL，当前 phases 仍是 `encoder, watermark, extract`，parser 不认识 `generate/calibrate/detect/report/audit`。

- [ ] **步骤 3：更新 phase source of truth**

修改 `wfcllm/orchestration/state.py`：

```python
PHASES = ["generate", "calibrate", "detect", "report", "audit"]
OPTIONAL_PHASES = ["encoder", "posthoc-pass-report", "diagnostic-selector"]
LEGACY_PHASES = [
    "legacy-watermark",
    "legacy-extract",
    "legacy-token-channel-train",
    "legacy-build-entropy-profile",
    "legacy-pretrain",
    "legacy-ablation",
]
ALL_PHASES = PHASES + OPTIONAL_PHASES + LEGACY_PHASES
```

Keep `DEFAULT_STATE_FILE = Path("data/run_state.json")` unchanged.

- [ ] **步骤 4：更新 argparse 新 flags**

在 `wfcllm/cli/arguments.py` 中：

```python
parser.add_argument("--phase", choices=ALL_PHASES, help="运行指定阶段（不指定则运行新版 WFCLLM 主流程）")
parser.add_argument("--legacy", action="store_true", help="允许显式运行归档的 legacy phase")
parser.add_argument("--run-id", default=None, help="WFCLLM run id；不传则按时间和 method 名生成")
parser.add_argument("--run-dir", default=None, help="WFCLLM run directory；优先于 config artifacts.run_root")
parser.add_argument("--input", default=None, help="detect 阶段 official final_code.jsonl 输入")
parser.add_argument("--negative-input", default=None, help="calibrate 阶段 reference negative final_code.jsonl")
parser.add_argument("--calibration", default=None, help="detect 阶段 calibration artifact 路径")
parser.add_argument("--positive-details", default=None, help="report 阶段 positive details JSONL")
parser.add_argument("--negative-details", default=None, help="report 阶段 negative details JSONL")
parser.add_argument("--diagnostic-only", action="store_true", help="允许运行 diagnostic-selector phase")
```

保留旧 flags 供 legacy runner 使用，但 help 文案标注 legacy，例如：

```python
help="legacy watermark 阶段使用：代码生成 LLM 路径"
```

- [ ] **步骤 5：新增 runners 并注册**

在 `wfcllm/cli/runners.py` 中新增最小 runner 形状，先让 CLI tests 通过，再在后续任务接上真实 pipeline：

```python
def run_generate(args: argparse.Namespace, state: RunStateManager) -> int:
    from wfcllm.method.presets import EVIDENCE_RETRY_SEED7X3_NAME

    state.mark_done("generate", method=EVIDENCE_RETRY_SEED7X3_NAME)
    print("=== WFCLLM generate ===")
    return 0


def run_calibrate(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("calibrate")
    print("=== WFCLLM calibrate ===")
    return 0


def run_detect(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("detect")
    print("=== WFCLLM detect ===")
    return 0


def run_report(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("report")
    print("=== WFCLLM report ===")
    return 0


def run_audit(args: argparse.Namespace, state: RunStateManager) -> int:
    state.mark_done("audit")
    print("=== WFCLLM audit ===")
    return 0
```

在 `wfcllm/cli/entry.py` 中注册：

```python
def _populate_phase_registry(reg: PhaseRegistry) -> None:
    reg.register("generate", run_generate)
    reg.register("calibrate", run_calibrate)
    reg.register("detect", run_detect)
    reg.register("report", run_report)
    reg.register("audit", run_audit)
    reg.register("posthoc-pass-report", run_posthoc_pass_report)
    reg.register("diagnostic-selector", run_diagnostic_selector)
    reg.register("encoder", run_encoder)
    reg.register("legacy-watermark", run_watermark)
    reg.register("legacy-extract", run_extract)
    reg.register("legacy-token-channel-train", run_token_channel_train)
    reg.register("legacy-build-entropy-profile", run_build_entropy_profile)
    reg.register("legacy-pretrain", run_pretrain)
```

新增 legacy validation：

```python
def validate_legacy_phase_request(args: argparse.Namespace) -> str | None:
    phase = getattr(args, "phase", None)
    if isinstance(phase, str) and phase.startswith("legacy-") and not getattr(args, "legacy", False):
        return "[错误] legacy phase requires --legacy"
    return None
```

在 `main()` 解析参数后调用，错误时 return 1。

- [ ] **步骤 6：更新 base config**

把 `configs/base_config.json` 改成：

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
  "artifacts": {
    "run_root": "data/runs"
  },
  "runtime": {
    "default_phases": ["generate", "calibrate", "detect", "report", "audit"]
  }
}
```

创建 `configs/wfcllm/evidence_retry_seed7x3.json`，内容与 `wfcllm.method.presets` 中 canonical values 一致。创建 detector/repro/diagnostics config，diagnostic config 必须包含：

```json
{
  "diagnostic_only": true,
  "not_official_method": true,
  "name": "evidence_only_selector_diagnostic"
}
```

- [ ] **步骤 7：更新 orchestrator detect skip**

在 `wfcllm/orchestration/pipeline.py` 中把 `_has_explicit_extract_input` 改成 `_has_explicit_detect_input`：

```python
if phase == "detect" and self._has_explicit_detect_input(args):
    return False
```

helper：

```python
@staticmethod
def _has_explicit_detect_input(args: argparse.Namespace) -> bool:
    cli_input = getattr(args, "input", None)
    config_cache = getattr(args, "_config_cache", None) or {}
    config_input = (config_cache.get("detector") or {}).get("input")
    return bool(cli_input or config_input)
```

- [ ] **步骤 8：运行 CLI/config tests 验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_default_cli_uses_new_wfcllm.py tests/integration/test_legacy_phase_requires_legacy_flag.py -v
conda run -n WFCLLM python run.py --status
```

预期：pytest PASS；`run.py --status` 输出 `generate, calibrate, detect, report, audit`。

- [ ] **步骤 9：Commit**

```bash
git add configs wfcllm/cli wfcllm/orchestration tests/integration
git commit -m "feat: switch cli default to wfcllm mainline"
```

---

### 任务 7：新增 official scripts、sanitizer 和 SAWR wrappers

**文件：**
- 创建：`scripts/wfcllm_generate.py`
- 创建：`scripts/wfcllm_detect.py`
- 创建：`scripts/wfcllm_sanitize_final_code.py`
- 修改：`scripts/run_sawr_smoke.py`
- 修改：`scripts/run_sawr_detect.py`
- 创建：`tests/integration/test_wfcllm_generate_cli.py`
- 创建：`tests/integration/test_wfcllm_detect_cli.py`
- 创建：`tests/scripts/test_wfcllm_sanitize_final_code.py`

- [ ] **步骤 1：编写 sanitizer 失败测试**

创建 `tests/scripts/test_wfcllm_sanitize_final_code.py`：

```python
from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_sanitize_final_code as sanitize_cli


def test_sanitize_final_code_writes_exact_four_fields(tmp_path: Path) -> None:
    source = tmp_path / "legacy.jsonl"
    output = tmp_path / "final_code.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def foo():\n",
                "generated_code": "def foo():\n    return 1\n",
                "watermark_params": {"secret_key": "blocked"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = sanitize_cli.main(["--input", str(source), "--output", str(output)])

    assert rc == 0
    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row == {
        "id": "HumanEval/0",
        "dataset": "humaneval",
        "prompt": "def foo():\n",
        "final_code": "def foo():\n    return 1\n",
    }
```

- [ ] **步骤 2：编写 script CLI 失败测试**

创建 `tests/integration/test_wfcllm_generate_cli.py`：

```python
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import scripts.wfcllm_generate as generate_cli


def test_wfcllm_generate_cli_builds_pipeline_config(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    pipeline = MagicMock()
    pipeline.run.return_value = str(tmp_path / "run" / "inputs" / "final_code.jsonl")

    with (
        patch("scripts.wfcllm_generate.WFCLLMGenerator"),
        patch("scripts.wfcllm_generate.WFCLLMGenerationPipeline", return_value=pipeline) as pipeline_cls,
    ):
        rc = generate_cli.main(
            [
                "--dataset",
                "humaneval",
                "--dataset-path",
                "data/datasets",
                "--model-path",
                str(model_path),
                "--output-dir",
                str(tmp_path / "run"),
                "--evidence-retry-attempts",
                "3",
                "--evidence-retry-seed-stride",
                "101",
                "--rule-name",
                "hash",
            ]
        )

    assert rc == 0
    assert pipeline_cls.call_args.kwargs["config"].evidence_retry_attempts == 3
    assert pipeline_cls.call_args.kwargs["config"].evidence_retry_seed_stride == 101
```

创建 `tests/integration/test_wfcllm_detect_cli.py` by copying `tests/integration/test_sawr_detect_cli.py` and replacing imports/names:

```python
import scripts.wfcllm_detect as detect_cli
```

and patch paths:

```python
"scripts.wfcllm_detect.load_wfcllm_window_scorer"
"scripts.wfcllm_detect.WFCLLMDetectionPipeline"
```

- [ ] **步骤 3：运行 script tests 验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/scripts/test_wfcllm_sanitize_final_code.py tests/integration/test_wfcllm_generate_cli.py tests/integration/test_wfcllm_detect_cli.py -v
```

预期：FAIL，scripts 不存在。

- [ ] **步骤 4：创建 sanitizer script**

创建 `scripts/wfcllm_sanitize_final_code.py`：

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.code_only import validate_final_code_record_exact  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sanitize legacy rows into WFCLLM final_code JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _sanitize_row(row: dict[str, Any]) -> dict[str, str]:
    final_code = row.get("final_code", row.get("generated_code"))
    sanitized = {
        "id": str(row["id"]),
        "dataset": str(row.get("dataset", "humaneval")),
        "prompt": str(row.get("prompt", "")),
        "final_code": final_code,
    }
    validate_final_code_record_exact(sanitized)
    return sanitized


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open(encoding="utf-8") as source:
            with output_path.open("w", encoding="utf-8") as target:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"input row must be object at line {line_number}")
                    target.write(json.dumps(_sanitize_row(row), allow_nan=False, ensure_ascii=False) + "\n")
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] WFCLLM final-code input saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **步骤 5：创建 official generate/detect scripts**

Create `scripts/wfcllm_generate.py` by adapting `scripts/run_sawr_smoke.py` with these import replacements:

```python
from wfcllm.method.config import WFCLLMGenerationConfig, WFCLLMPipelineConfig, WFCLLMRuleConfig
from wfcllm.generation.generator import WFCLLMGenerator
from wfcllm.generation.pipeline import WFCLLMGenerationPipeline
from wfcllm.semantic.rules import HashEmbeddingRule
from wfcllm.semantic.lsh import load_semantic_lsh_rule
```

Final success message:

```python
print(f"[完成] WFCLLM final-code rows saved to {output_path}", file=sys.stderr)
```

Create `scripts/wfcllm_detect.py` by adapting `scripts/run_sawr_detect.py` with:

```python
from wfcllm.detection.calibration import load_calibration_artifact, write_calibration_artifact
from wfcllm.detection.config import WFCLLMDetectionConfig
from wfcllm.detection.metrics import build_detection_report, split_records_by_task, write_detection_report
from wfcllm.detection.pipeline import WFCLLMDetectionPipeline, load_jsonl_records
from wfcllm.detection.scoring import load_wfcllm_window_scorer
```

Parser description:

```python
description="Run WFCLLM strict final-code-only detection."
```

- [ ] **步骤 6：把 SAWR scripts 改成 deprecation wrappers**

`scripts/run_sawr_smoke.py` wrapper content：

```python
#!/usr/bin/env python
from __future__ import annotations

import sys
import warnings

from scripts.wfcllm_generate import main as wfcllm_generate_main


def main(argv: list[str] | None = None) -> int:
    warnings.warn(
        "scripts/run_sawr_smoke.py is deprecated; use scripts/wfcllm_generate.py",
        DeprecationWarning,
        stacklevel=2,
    )
    print("[deprecated] use scripts/wfcllm_generate.py", file=sys.stderr)
    return wfcllm_generate_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
```

`scripts/run_sawr_detect.py` wrapper content mirrors this and calls `scripts.wfcllm_detect.main`.

- [ ] **步骤 7：运行 tests 验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/scripts/test_wfcllm_sanitize_final_code.py tests/integration/test_wfcllm_generate_cli.py tests/integration/test_wfcllm_detect_cli.py -v
```

预期：PASS。

- [ ] **步骤 8：Commit**

```bash
git add scripts tests/integration/test_wfcllm_generate_cli.py tests/integration/test_wfcllm_detect_cli.py tests/scripts
git commit -m "feat: add wfcllm official scripts"
```

---

### 任务 8：迁移 diagnostic selectors 并强制 diagnostic-only 标记

**文件：**
- 创建：`wfcllm/diagnostics/__init__.py`
- 创建：`wfcllm/diagnostics/evidence_selector.py`
- 创建：`wfcllm/diagnostics/quality_selector.py`
- 创建：`wfcllm/diagnostics/static_selector.py`
- 创建：`wfcllm/diagnostics/mechanism_analysis.py`
- 创建：`wfcllm/diagnostics/sparse_statistics.py`
- 创建：`tests/diagnostics/test_evidence_selector_diagnostic.py`
- 创建：`tests/diagnostics/test_quality_selector_diagnostic.py`
- 修改：`scripts/diagnostics/select_evidence_only_candidates.py`
- 修改：`scripts/diagnostics/select_quality_candidates_diagnostic.py`
- 修改：`scripts/diagnostics/select_static_candidates_diagnostic.py`

- [ ] **步骤 1：编写 diagnostic marker 失败测试**

创建 `tests/diagnostics/test_evidence_selector_diagnostic.py`：

```python
from __future__ import annotations

from wfcllm.diagnostics.evidence_selector import mark_evidence_selector_diagnostic


def test_mark_evidence_selector_diagnostic_sets_required_fields() -> None:
    payload = mark_evidence_selector_diagnostic({"name": "evidence_only_selector_diagnostic"})

    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["name"] == "evidence_only_selector_diagnostic"
```

创建 `tests/diagnostics/test_quality_selector_diagnostic.py`：

```python
from __future__ import annotations

from wfcllm.diagnostics.quality_selector import mark_quality_selector_diagnostic


def test_mark_quality_selector_diagnostic_sets_required_fields() -> None:
    payload = mark_quality_selector_diagnostic({"name": "quality_selector"})

    assert payload["diagnostic_only"] is True
    assert payload["not_official_method"] is True
    assert payload["uses_quality_proxy"] is True
```

- [ ] **步骤 2：运行测试验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
```

预期：FAIL，报错包含 `ModuleNotFoundError: No module named 'wfcllm.diagnostics'`。

- [ ] **步骤 3：实现 diagnostic marker helpers**

创建 `wfcllm/diagnostics/evidence_selector.py`：

```python
from __future__ import annotations

from typing import Any

from wfcllm.method.contracts import require_diagnostic_marker


def mark_evidence_selector_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    marked = {
        **payload,
        "diagnostic_only": True,
        "not_official_method": True,
    }
    require_diagnostic_marker(marked)
    return marked
```

创建 `wfcllm/diagnostics/quality_selector.py`：

```python
from __future__ import annotations

from typing import Any

from wfcllm.method.contracts import require_diagnostic_marker


def mark_quality_selector_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    marked = {
        **payload,
        "diagnostic_only": True,
        "not_official_method": True,
        "uses_quality_proxy": True,
    }
    require_diagnostic_marker(marked)
    return marked
```

创建 `wfcllm/diagnostics/__init__.py`：

```python
from __future__ import annotations

from wfcllm.diagnostics.evidence_selector import mark_evidence_selector_diagnostic
from wfcllm.diagnostics.quality_selector import mark_quality_selector_diagnostic

__all__ = [
    "mark_evidence_selector_diagnostic",
    "mark_quality_selector_diagnostic",
]
```

- [ ] **步骤 4：迁移 selector 逻辑**

运行：

```bash
git mv wfcllm/sawr/selection.py wfcllm/diagnostics/static_selector.py
perl -0pi -e 's/wfcllm\.sawr/wfcllm.diagnostics/g' wfcllm/diagnostics/static_selector.py
```

从 `scripts/select_sawr_candidates.py` 提取 evidence-only 纯函数到 `wfcllm/diagnostics/evidence_selector.py`，所有对外返回 payload 必须经过 `mark_evidence_selector_diagnostic()`。从 `scripts/select_sawr_static_candidates.py` 提取 static selector 纯函数到 `wfcllm/diagnostics/static_selector.py`，返回 payload 必须包含：

```python
{
    "diagnostic_only": True,
    "not_official_method": True,
    "uses_quality_proxy": True,
}
```

- [ ] **步骤 5：创建 diagnostics scripts**

移动 scripts：

```bash
mkdir -p scripts/diagnostics
git mv scripts/analyze_sawr_no_quality_gate_goal.py scripts/diagnostics/analyze_no_quality_gate.py
git mv scripts/analyze_sawr_sparse_statistics.py scripts/diagnostics/analyze_sparse_statistics.py
git mv scripts/select_sawr_candidates.py scripts/diagnostics/select_evidence_only_candidates.py
git mv scripts/select_sawr_static_candidates.py scripts/diagnostics/select_static_candidates_diagnostic.py
```

在每个 diagnostics script 输出 JSON 的位置加 marker：

```python
payload["diagnostic_only"] = True
payload["not_official_method"] = True
```

quality/static selector 额外加：

```python
payload["uses_quality_proxy"] = True
```

- [ ] **步骤 6：运行 diagnostics tests 验证通过**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
```

预期：PASS。

- [ ] **步骤 7：Commit**

```bash
git add wfcllm/diagnostics scripts/diagnostics tests/diagnostics wfcllm/sawr
git commit -m "refactor: move selectors to diagnostics"
```

---

### 任务 9：归档 legacy code/config/scripts/tests/docs

**文件：**
- 创建：`archive/legacy_wfcllm_2026_07/README.md`
- 创建：`archive/legacy_wfcllm_2026_07/MANIFEST.json`
- 创建：`archive/legacy_wfcllm_2026_07/experiment_results/MANIFEST.md`
- 移动：`wfcllm/watermark/` 剩余内容到 `archive/legacy_wfcllm_2026_07/code/watermark/`
- 移动：`wfcllm/extract/` 到 `archive/legacy_wfcllm_2026_07/code/extract/`
- 移动：`wfcllm/ablation/` 到 `archive/legacy_wfcllm_2026_07/code/ablation/`
- 移动：`wfcllm/pretrain/` 到 `archive/legacy_wfcllm_2026_07/code/pretrain/`
- 移动：`wfcllm/evaluation/dual_channel.py` 到 `archive/legacy_wfcllm_2026_07/code/dual_channel/dual_channel.py`
- 移动：旧 configs 到 `configs/legacy/`
- 移动：旧 scripts 到 `scripts/legacy/`
- 移动：旧 tests 到 `archive/legacy_wfcllm_2026_07/tests/`
- 创建：`tests/test_legacy_archive_manifest.py`

- [ ] **步骤 1：编写 archive manifest 失败测试**

创建 `tests/test_legacy_archive_manifest.py`：

```python
from __future__ import annotations

import json
from pathlib import Path


def test_legacy_archive_manifest_exists_and_names_archived_roots() -> None:
    manifest_path = Path("archive/legacy_wfcllm_2026_07/MANIFEST.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["archive_id"] == "legacy_wfcllm_2026_07"
    archived_paths = {entry["path"] for entry in manifest["entries"]}
    assert "archive/legacy_wfcllm_2026_07/code/watermark" in archived_paths
    assert "archive/legacy_wfcllm_2026_07/code/extract" in archived_paths
    assert "configs/legacy" in archived_paths
    assert "scripts/legacy" in archived_paths
```

- [ ] **步骤 2：运行 archive test 验证失败**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_legacy_archive_manifest.py -v
```

预期：FAIL，manifest 文件不存在。

- [ ] **步骤 3：移动 legacy code**

运行：

```bash
mkdir -p archive/legacy_wfcllm_2026_07/code archive/legacy_wfcllm_2026_07/tests
git mv wfcllm/watermark archive/legacy_wfcllm_2026_07/code/watermark
git mv wfcllm/extract archive/legacy_wfcllm_2026_07/code/extract
git mv wfcllm/ablation archive/legacy_wfcllm_2026_07/code/ablation
git mv wfcllm/pretrain archive/legacy_wfcllm_2026_07/code/pretrain
mkdir -p archive/legacy_wfcllm_2026_07/code/dual_channel
git mv wfcllm/evaluation/dual_channel.py archive/legacy_wfcllm_2026_07/code/dual_channel/dual_channel.py
```

保留 `wfcllm/evaluation/__init__.py`、`wfcllm/evaluation/benchmark.py`、`wfcllm/evaluation/code_execution.py`、`wfcllm/evaluation/detection_report.py` 和 `wfcllm/evaluation/anchor_validation/`，因为 posthoc pass reporting 和 historical metrics still use these utilities.

- [ ] **步骤 4：移动 legacy tests**

运行：

```bash
git mv tests/watermark archive/legacy_wfcllm_2026_07/tests/watermark
git mv tests/extract archive/legacy_wfcllm_2026_07/tests/extract
git mv tests/ablation archive/legacy_wfcllm_2026_07/tests/ablation
git mv tests/integration/test_watermark_phase.py archive/legacy_wfcllm_2026_07/tests/test_watermark_phase.py
git mv tests/integration/test_extract_phase.py archive/legacy_wfcllm_2026_07/tests/test_extract_phase.py
git mv tests/integration/test_token_channel_train.py archive/legacy_wfcllm_2026_07/tests/test_token_channel_train.py
git mv tests/integration/test_pretrain_phase.py archive/legacy_wfcllm_2026_07/tests/test_pretrain_phase.py
git mv tests/integration/test_build_entropy_profile_runner.py archive/legacy_wfcllm_2026_07/tests/test_build_entropy_profile_runner.py
git mv tests/integration/test_generate_negative.py archive/legacy_wfcllm_2026_07/tests/test_generate_negative.py
```

- [ ] **步骤 5：移动 legacy configs/scripts/docs**

运行：

```bash
mkdir -p configs/legacy/ablation scripts/legacy docs/archive/sawr docs/archive/legacy_wfcllm docs/archive/plans docs/archive/progress
git mv configs/exp1_ngram_retries.json configs/legacy/exp1_ngram_retries.json
git mv configs/exp2_lower_delta.json configs/legacy/exp2_lower_delta.json
git mv configs/exp3_higher_gamma.json configs/legacy/exp3_higher_gamma.json
git mv configs/exp3_new_gamma.json configs/legacy/exp3_new_gamma.json
git mv configs/exp4_higher_lsh_d.json configs/legacy/exp4_higher_lsh_d.json
git mv configs/exp4_temp_escalation.json configs/legacy/exp4_temp_escalation.json
git mv configs/exp_semantic_only.json configs/legacy/exp_semantic_only.json
git mv configs/exp_white.json configs/legacy/exp_white.json
git mv configs/humaneval_10_config.json configs/legacy/humaneval_10_config.json
git mv configs/ablation/example_dual_channel.yaml configs/legacy/ablation/example_dual_channel.yaml
git mv scripts/evaluate_dual_channel.py scripts/legacy/evaluate_dual_channel.py
git mv scripts/run_ablation.py scripts/legacy/run_ablation.py
git mv scripts/build_entropy_profile.py scripts/legacy/build_entropy_profile.py
git mv scripts/calibrate.py scripts/legacy/calibrate.py
git mv scripts/calibrate_threshold.py scripts/legacy/calibrate_threshold.py
git mv scripts/generate_negative_corpus.py scripts/legacy/generate_negative_corpus.py
git mv scripts/merge_watermarked.py scripts/legacy/merge_watermarked.py
git mv docs/SAWR_GENERATION_TIME_EMBEDDING_SCHEME.md docs/archive/sawr/SAWR_GENERATION_TIME_EMBEDDING_SCHEME.md
git mv docs/experiment/2026-06-15-sawr-structure-aware-generation-experiment-report.md docs/archive/sawr/2026-06-15-sawr-structure-aware-generation-experiment-report.md
```

- [ ] **步骤 6：创建 manifests**

创建 `archive/legacy_wfcllm_2026_07/README.md`：

```markdown
# Legacy WFCLLM Archive 2026-07

This archive preserves the pre-mainline WFCLLM watermark/extract/token-channel/adaptive-gamma/dual-channel code path for historical reproducibility. New production code must not import from this directory.
```

创建 `archive/legacy_wfcllm_2026_07/MANIFEST.json`：

```json
{
  "archive_id": "legacy_wfcllm_2026_07",
  "created_at": "2026-07-08",
  "policy": "small source/config/script/doc/test artifacts are tracked; large local data and model artifacts are not tracked",
  "entries": [
    {"path": "archive/legacy_wfcllm_2026_07/code/watermark", "role": "legacy watermark pipeline"},
    {"path": "archive/legacy_wfcllm_2026_07/code/extract", "role": "legacy extract pipeline"},
    {"path": "archive/legacy_wfcllm_2026_07/code/ablation", "role": "legacy ablation sweeps"},
    {"path": "archive/legacy_wfcllm_2026_07/code/pretrain", "role": "legacy pretrain phase"},
    {"path": "archive/legacy_wfcllm_2026_07/code/dual_channel", "role": "legacy dual-channel evaluation"},
    {"path": "archive/legacy_wfcllm_2026_07/tests", "role": "archived tests removed from default pytest suite"},
    {"path": "configs/legacy", "role": "legacy configs"},
    {"path": "scripts/legacy", "role": "legacy scripts"},
    {"path": "docs/archive", "role": "archived docs"}
  ]
}
```

创建 `archive/legacy_wfcllm_2026_07/experiment_results/MANIFEST.md`：

```markdown
# Legacy Experiment Results Manifest

This manifest indexes historical result families without tracking large raw JSONL, checkpoint, logits, model, or dataset artifacts in Git.

| Name | Role | Git-tracked status |
| --- | --- | --- |
| sawr_mechanism_no_quality_gate_goal_20260621 | Historical SAWR mechanism diagnostic | summary/index only |
| sawr_evidence_constrained_goal_20260621 | Historical evidence-constrained diagnostic | summary/index only |
| sawr_full164_tpr30_goal_20260620 | Historical full-164 detector target run | summary/index only |
| legacy_watermark_extract_summary | Legacy WFCLLM watermark/extract baseline | summary/index only |
```

- [ ] **步骤 7：运行 archive/default import checks**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_legacy_archive_manifest.py -v
rg "from wfcllm\\.(watermark|extract|ablation|pretrain)|import wfcllm\\.(watermark|extract|ablation|pretrain)" wfcllm tests scripts --glob '!archive/**'
```

预期：pytest PASS；`rg` 没有 production mainline import legacy archived packages。允许 `archive/` 中出现 legacy imports。

- [ ] **步骤 8：Commit**

```bash
git add archive configs scripts docs tests wfcllm
git commit -m "chore: archive legacy wfcllm pipeline"
```

---

### 任务 10：重写 README、CLAUDE 和正式方法文档

**文件：**
- 修改：`README.md`
- 修改：`CLAUDE.md`
- 创建：`docs/WFCLLM_METHOD.md`
- 创建：`docs/NO_QUALITY_GATE_PROTOCOL.md`
- 创建：`docs/STRICT_CODE_ONLY_DETECTOR.md`
- 创建：`docs/ARTIFACT_SCHEMA.md`
- 创建：`docs/REPRO_EVIDENCE_RETRY_SEED7X3.md`
- 创建：`docs/DIAGNOSTIC_UPPER_BOUND.md`
- 创建：`docs/LEGACY_ARCHIVE.md`
- 测试/检查：`rg` 文档命名扫描

- [ ] **步骤 1：写文档检查脚本命令并先验证当前失败**

运行：

```bash
rg "阶段一|阶段二|阶段三|token-channel|dual-channel|adaptive gamma|SAWR" README.md CLAUDE.md docs/WFCLLM_METHOD.md docs/NO_QUALITY_GATE_PROTOCOL.md docs/STRICT_CODE_ONLY_DETECTOR.md docs/ARTIFACT_SCHEMA.md docs/REPRO_EVIDENCE_RETRY_SEED7X3.md docs/DIAGNOSTIC_UPPER_BOUND.md docs/LEGACY_ARCHIVE.md
```

预期：当前 README/CLAUDE 中出现旧阶段表述，且新 docs 不存在。

- [ ] **步骤 2：创建 `docs/WFCLLM_METHOD.md`**

内容必须包含：

```markdown
# WFCLLM Method

WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. It embeds semantic LSH evidence during generation through structure-aware boundary detection, rollback-capable evidence-only retry, and final-code-only statistical detection.

The default official method preset is `evidence_retry_seed7x3`.

The protocol forbids pass/test/correctness proxies during generation, retry, calibration, detection, and final candidate selection.
```

并列出当前 known metrics：

```markdown
| Metric | Value |
| --- | --- |
| pass@1 | 104/164 = 63.41% |
| AUROC | 0.660080 |
| TP@<=8FP | 43/164 |
| FP | 8 |
| positive insufficient | 20 |
| model-negative AUROC | 0.510332 |
```

文末必须写：

```markdown
Detector evidence remains the main limitation. The current preset is the best official single-configuration generation-time candidate, not a solved detector result.
```

- [ ] **步骤 3：创建 protocol/schema/repro docs**

`docs/NO_QUALITY_GATE_PROTOCOL.md` 必须列出 allowed signals 和 forbidden signals；posthoc pass report JSON marker 必须为：

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

`docs/STRICT_CODE_ONLY_DETECTOR.md` 必须包含 official input exactly:

```json
{"id": "HumanEval/0", "dataset": "humaneval", "prompt": "...", "final_code": "..."}
```

`docs/ARTIFACT_SCHEMA.md` 必须列出 `data/runs/<run_id>/` layout。

`docs/REPRO_EVIDENCE_RETRY_SEED7X3.md` 必须写入 Appendix A preset values，并给出命令：

```bash
python run.py --config configs/wfcllm/repro_humaneval_full164.json
python run.py --phase generate --config configs/wfcllm/evidence_retry_seed7x3.json
python run.py --phase calibrate --negative-input data/negative/reference_final_code.jsonl
python run.py --phase detect --input data/runs/<run_id>/inputs/final_code.jsonl
```

`docs/DIAGNOSTIC_UPPER_BOUND.md` 必须说明 `evidence_only_selector_diagnostic` 不是正式 method，包含：

```markdown
`diagnostic_only=true` and `not_official_method=true` are required for all selector outputs.
```

`docs/LEGACY_ARCHIVE.md` 必须说明 archive 路径与不追踪大型文件策略。

- [ ] **步骤 4：重写 README 和 CLAUDE**

`README.md` 顶部改成：

```markdown
# WFCLLM: Structure-Aware Generation-Time Semantic Watermarking for Code LLMs

WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. The default method preset is `evidence_retry_seed7x3`.
```

Quick start 使用新命令：

```bash
python run.py --status
python run.py --config configs/base_config.json
python run.py --phase generate
python run.py --phase detect --input data/runs/<run_id>/inputs/final_code.jsonl
python run.py --phase audit --run-dir data/runs/<run_id>
```

`CLAUDE.md` 的架构图改成：

```text
wfcllm/
  method/
  generation/
  semantic/
  detection/
  audit/
  diagnostics/
  cli/
  orchestration/
  datasets/
  lang/
  encoder/
  common/
```

- [ ] **步骤 5：运行文档扫描**

运行：

```bash
rg "阶段一|阶段二|阶段三|token-channel|dual-channel|adaptive gamma" README.md CLAUDE.md docs/WFCLLM_METHOD.md docs/NO_QUALITY_GATE_PROTOCOL.md docs/STRICT_CODE_ONLY_DETECTOR.md docs/ARTIFACT_SCHEMA.md docs/REPRO_EVIDENCE_RETRY_SEED7X3.md docs/DIAGNOSTIC_UPPER_BOUND.md docs/LEGACY_ARCHIVE.md
rg "SAWR" README.md CLAUDE.md docs/WFCLLM_METHOD.md docs/REPRO_EVIDENCE_RETRY_SEED7X3.md
```

预期：第一个命令无输出。第二个命令无输出；`SAWR` 只允许出现在 `docs/DIAGNOSTIC_UPPER_BOUND.md`、`docs/LEGACY_ARCHIVE.md`、`docs/archive/**`、`archive/**`。

- [ ] **步骤 6：Commit**

```bash
git add README.md CLAUDE.md docs
git commit -m "docs: document wfcllm mainline method"
```

---

### 任务 11：全局验证与合并准备

**文件：**
- 修改：只允许修复前面任务暴露的 import/test/doc 断点
- 检查：`wfcllm/`、`tests/`、`scripts/`、`configs/`、`docs/`

- [ ] **步骤 1：运行 targeted suites**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/semantic -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_default_cli_uses_new_wfcllm.py tests/integration/test_legacy_phase_requires_legacy_flag.py tests/integration/test_wfcllm_generate_cli.py tests/integration/test_wfcllm_detect_cli.py -v
```

预期：全部 PASS。

- [ ] **步骤 2：运行 import and syntax smoke**

运行：

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

预期：退出码 0。

- [ ] **步骤 3：运行 full default suite**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

预期：PASS。若因本机缺少 GPU/model 导致少量 integration 测试不可运行，把失败命令、错误首行、受影响测试文件记录在 final message 中；不要把缺少资源描述成代码成功。

- [ ] **步骤 4：扫描 forbidden imports 和 official docs naming**

运行：

```bash
rg "from experiment|import experiment" wfcllm tests scripts
rg "from wfcllm\\.sawr|import wfcllm\\.sawr" wfcllm tests scripts --glob '!archive/**' --glob '!wfcllm/sawr/**'
rg "from wfcllm\\.(watermark|extract|ablation|pretrain)|import wfcllm\\.(watermark|extract|ablation|pretrain)" wfcllm tests scripts --glob '!archive/**'
rg "token_channel|dual-channel|adaptive_gamma|build_entropy_profile|pretrain" configs/base_config.json
```

预期：所有命令无输出，除了 `wfcllm/sawr/**` shim 自身和 `archive/**`。

- [ ] **步骤 5：检查 git diff**

运行：

```bash
git status --short
git diff --stat develop...HEAD
git log --oneline --decorate -n 20
```

预期：只有本计划相关代码、测试、配置、脚本、文档、archive 变更。没有 `data/models/`、`data/datasets/`、大型 JSONL、checkpoint、logits dump。

- [ ] **步骤 6：最终 commit**

如果步骤 1-5 暴露了修复并产生未提交修改：

```bash
git add wfcllm tests scripts configs docs archive README.md CLAUDE.md run.py
git commit -m "fix: stabilize wfcllm mainline refactor"
```

如果没有未提交修改，运行：

```bash
git status --short
```

预期：工作树干净。

- [ ] **步骤 7：合并准备说明**

在 PR/merge description 中写明：

```markdown
Validation:
- HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
- HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
- HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
- HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
- HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
- HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
- conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools

Protocol:
- Default method preset: evidence_retry_seed7x3.
- Detector input: exact id,dataset,prompt,final_code only.
- Diagnostic selectors: diagnostic_only=true, not_official_method=true.
- Legacy code/config/scripts/docs archived under archive/configs/legacy/scripts/legacy/docs/archive.
```

---

## 规格覆盖自检

- Goals 1-3：任务 1、2、3、6 把 SAWR 核心升格为 `method/generation/semantic/detection` 并切换默认 CLI/config。
- Goals 4-6：任务 1、4、5、8 强制 no-quality-gate、strict final-code-only detector、diagnostic-only selector marker。
- Goals 7-9：任务 0、9、10、11 覆盖 Git/worktree 保护、legacy archive、docs rewrite、repro preservation。
- Non-goals：计划没有引入 detector 指标提升目标，没有把 diagnostic selector 设为正式 method，没有把旧 token-channel/dual-channel/adaptive-gamma 留在默认 config。
- Artifact schema：任务 3、4、5 覆盖 final-code input、audit/sidecar marker、detector output 禁止字段、posthoc pass report marker。
- CLI redesign：任务 6 覆盖新 phases、legacy flag、官方命令参数。
- Tests：任务 1-11 覆盖 method/generation/semantic/detection/audit/diagnostics/integration/script/docs/archive。
- Git hygiene：任务 0 和任务 11 覆盖 dirty worktree protection、worktree branch、large data exclusion、merge target。

## 完整性与一致性自检

- 本计划的每个任务都有明确文件、测试、命令、预期结果和 commit 边界。
- `WFCLLMGenerationConfig`、`WFCLLMRuleConfig`、`WFCLLMPipelineConfig`、`WFCLLMDetectionConfig`、`WFCLLMGenerationPipeline`、`WFCLLMDetectionPipeline` 的名称在测试和实现步骤中保持一致。
- `evidence_retry_key` 在 `wfcllm.method.contracts` 与 `wfcllm.generation.retry` 中签名一致：`(result, attempt_index) -> tuple[int, int, int, int, int]`。
- Official detector input 字段在所有任务中保持一致：`id`, `dataset`, `prompt`, `final_code`。
- Diagnostic marker 在所有任务中保持一致：`diagnostic_only=true`, `not_official_method=true`。
