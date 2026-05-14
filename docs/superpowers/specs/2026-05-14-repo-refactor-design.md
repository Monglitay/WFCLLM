# 仓库重构 & 实验可扩展性设计

**日期**：2026-05-14
**状态**：设计待审
**范围**：仅整理 + 抽接口；不实现新语言/新数据集的具体逻辑

## 1. 背景与动机

仓库已实现核心 3 阶段（编码器预训练 → 生成时嵌入 → 提取检测）+ 双通道水印 + 自适应 γ，处于"功能完整、内部结构松散"的状态。下一阶段计划：

- **多语言扩展**：Python（HumanEval+MBPP，已有）→ C++/Java/JS（HumanEvalPack）
- **多数据集扩展**：HumanEvalPack 加入，原 HumanEval/MBPP 保持
- **消融实验**：扫双通道模式 + 检测阈值、margin/LSH 参数、上游 LLM/编码器设置三组

当前仓库存在两类问题阻碍这些扩展：

**类型 A：游离件分散** — 同一职责的代码物理上散在多处（"calibrate" 一词同时指 Phase 2 熵剖面校准和 Phase 3 阈值校准；负样本生成被拆在 `scripts/` 和 `wfcllm/extract/` 两处；评估代码挂在 `wfcllm/common/`、`wfcllm/extract/` 和 `scripts/` 三处）。

**类型 B：语言/数据集假设硬编码** — `tree_sitter_python` 直接在 `common/ast_parser.py` 顶层 import；`SUPPORTED_DATASETS = ("humaneval", "mbpp")` 在多处字符串匹配分发；变换规则全部 Python AST 专属。要加新语言/新数据集得改 5+ 个文件。

## 2. 目标与非目标

### 目标

1. 把 7 组游离件整合到清晰归属的子包/顶级包；
2. 抽出 `LanguageAdapter` 和 `DatasetAdapter` 两层注册式接口，今天只实现 Python/HumanEval/MBPP，未来加新语言/数据集 = 加新 Adapter；
3. 提供最小可用的消融实验框架（YAML 描述笛卡尔积扫描 + 指标 JSONL 输出）；
4. 拆 `run.py`（56KB）和 `tests/test_run.py`（64KB），单职责的 CLI / Orchestration / 测试分组；
5. 保持已训 encoder / token-channel 权重的 `state_dict` 兼容；
6. 预飞步骤（熵剖面、负语料、阈值校准）在主流程缺产物时**自动触发**。

### 非目标

- 不实现 Java/C++/JS 的真实 LanguageAdapter（仅留接口和空目录骨架）；
- 不实现 HumanEvalPack 的真实 DatasetAdapter（仅留接口）；
- 不动 `experiment/` 目录（仍受 CLAUDE.md 约束）；
- 不改算法逻辑（编码器对比损失、LSH 验证、双通道融合等）；
- 不做分布式调度 / 超参优化 / 自动超参建议。

## 3. 目标包结构

```
WFCLLM/
├── run.py                                  # 3 行壳：from wfcllm.cli.entry import main
│
├── wfcllm/
│   ├── __init__.py
│   ├── common/                             # 通用基础设施（瘦身）
│   │   ├── ast_parser.py                   # 改：调度到 LanguageAdapter.extract_blocks
│   │   ├── block_contract.py               # 不动
│   │   ├── dataset_loader.py               # 改：调度到 DatasetAdapter
│   │   ├── registry.py                     # 新：通用 Registry[T] 容器
│   │   └── transform/                      # 瘦身：只保留 base.py + engine.py
│   │       ├── base.py
│   │       └── engine.py
│   │
│   ├── lang/                               # 新：语言适配器
│   │   ├── adapter.py                      # LanguageAdapter ABC
│   │   ├── registry.py
│   │   └── python/                         # PythonAdapter
│   │       ├── adapter.py
│   │       ├── statement_types.py          # ← common/ast_parser.py SIMPLE/COMPOUND_STATEMENT_TYPES
│   │       └── transform/                  # ← common/transform/{positive,negative}/
│   │           ├── positive/
│   │           └── negative/
│   │
│   ├── datasets/                           # 新：数据集适配器
│   │   ├── adapter.py                      # DatasetAdapter ABC
│   │   ├── registry.py
│   │   ├── humaneval.py                    # 实现
│   │   ├── mbpp.py                         # 实现
│   │   └── humanevalpack.py                # 接口骨架（不实现具体加载）
│   │
│   ├── encoder/                            # Phase 1 子阶段 a（不动）
│   │
│   ├── pretrain/                           # 新：Phase 1 编排
│   │   ├── pipeline.py                     # PretrainPipeline：encoder → lexical
│   │   ├── config.py
│   │   └── __main__.py
│   │
│   ├── watermark/                          # Phase 2
│   │   ├── orchestrator.py                 # ← generator.py 主流程部分（约 600 行）
│   │   ├── semantic_channel.py             # ← generator.py 语义验证调用（约 400 行）
│   │   ├── interceptor.py                  # 不动
│   │   ├── verifier.py                     # 不动
│   │   ├── keying.py                       # 不动
│   │   ├── lsh_space.py                    # 不动
│   │   ├── context.py                      # 不动
│   │   ├── cascade.py                      # 不动
│   │   ├── retry_loop.py                   # 不动
│   │   ├── kv_cache.py                     # 不动
│   │   ├── diagnostics.py                  # 不动
│   │   ├── pipeline.py                     # 不动
│   │   ├── config.py                       # 不动
│   │   ├── adaptive_gamma/                 # 新子包
│   │   │   ├── entropy.py                  # ← watermark/entropy.py
│   │   │   ├── profile.py                  # ← watermark/entropy_profile.py
│   │   │   ├── schedule.py                 # ← watermark/gamma_schedule.py
│   │   │   └── calibrate.py                # ← scripts/calibrate.py 核心
│   │   └── token_channel/                  # 内部三分层
│   │       ├── core/                       # 训/推共享
│   │       │   ├── config.py
│   │       │   ├── model.py
│   │       │   ├── features.py
│   │       │   └── protocol.py
│   │       ├── training/                   # Phase 1 用
│   │       │   ├── teacher.py
│   │       │   ├── corpus.py
│   │       │   ├── corpus_streaming.py
│   │       │   ├── trainer.py
│   │       │   └── workflow.py
│   │       └── runtime/                    # Phase 2 用
│   │           └── injector.py
│   │
│   ├── extract/                            # Phase 3
│   │   ├── detector.py                     # 不动（内部走 LanguageAdapter）
│   │   ├── scorer.py                       # 不动
│   │   ├── hypothesis.py                   # 不动
│   │   ├── dp_selector.py                  # 不动
│   │   ├── alignment.py                    # 不动
│   │   ├── pipeline.py                     # 不动
│   │   ├── config.py                       # 不动
│   │   ├── token_channel.py                # 不动（replay 检测）
│   │   └── calibration/                    # 新子包
│   │       ├── negative_corpus.py          # ← extract/negative_corpus.py
│   │       ├── threshold.py                # ← extract/calibrator.py
│   │       └── runner.py                   # ← scripts/generate_negative_corpus.py 核心
│   │
│   ├── evaluation/                         # 新顶级：跨阶段评估
│   │   ├── code_execution.py               # ← common/offline_code_eval.py
│   │   ├── detection_report.py             # ← extract/offline_analysis.py
│   │   └── dual_channel.py                 # ← scripts/evaluate_dual_channel.py
│   │
│   ├── ablation/                           # 新顶级：消融实验框架
│   │   ├── sweep.py                        # SweepSpec：YAML 解析 + 笛卡尔积
│   │   ├── runner.py                       # AblationRunner：逐组合 + 断点恢复
│   │   └── metrics.py                      # 指标聚合 + JSONL 输出
│   │
│   ├── cli/                                # 新
│   │   ├── arguments.py                    # argparse 定义（每 phase 一组）
│   │   ├── config_resolver.py              # CLI override + JSON merge + 实例化
│   │   └── entry.py                        # main()
│   │
│   └── orchestration/                      # 新
│       ├── state.py                        # RunStateManager
│       ├── phase_registry.py
│       ├── pipeline.py                     # PhaseOrchestrator
│       └── prereq.py                       # 预飞自动触发逻辑
│
├── scripts/                                # 全部退化为 argparse 壳
│   ├── build_entropy_profile.py            # ← scripts/calibrate.py
│   ├── calibrate_threshold.py              # 新名（消歧）
│   ├── generate_negative_corpus.py         # 保留旧名
│   ├── evaluate.py                         # 统一评估入口（子命令）
│   └── run_ablation.py                     # 新
│
├── tools/                                  # 不动（download/verify/debug 工具）
│
├── configs/
│   ├── base_config.json                    # 改 schema（增 language / dataset）
│   └── ablation/                           # 新
│       └── example_dual_channel.yaml
│
├── tests/
│   ├── conftest.py
│   ├── unit/                               # 重组：原 common/ encoder/ watermark/ extract/
│   ├── integration/                        # 拆自 test_run.py
│   │   ├── test_pretrain_phase.py
│   │   ├── test_watermark_phase.py
│   │   ├── test_extract_phase.py
│   │   ├── test_calibration_phases.py
│   │   ├── test_evaluation.py
│   │   ├── test_ablation.py
│   │   ├── test_cli_resolution.py
│   │   └── test_orchestration.py
│   └── lang/                               # 新：LanguageAdapter 测试
│       └── test_python_adapter.py
│
└── docs/
    ├── design/                             # 活的设计文档
    ├── history/                            # 已完成的重构/修复记录
    ├── progress/                           # 汇报材料
    ├── plans/                              # 实现计划（不动）
    ├── experiment/                         # 实验报告（不动）
    ├── superpowers/                        # skill 资料（不动）
    └── references/                         # 论文 PDF + 规则 CSV
```

## 4. 七组游离件的整合

### 4.1 Phase 1：语义编码器 + 词汇通道判别器联合训练

**现状**：词汇通道训练码（`teacher.py`、`train.py`、`train_workflow.py`、`train_corpus*.py` ≈ 2400 行）混在 `wfcllm/watermark/token_channel/` 根目录，和运行时推理 `runtime.py` 并列。

**整合**：

1. 把 `wfcllm/watermark/token_channel/` 按职责切三层：
   - `core/`：训/推共享（`config.py`、`model.py`、`features.py`、`protocol.py`）
   - `training/`：仅 Phase 1 用（`teacher.py`、`corpus.py`、`corpus_streaming.py`、`trainer.py`、`workflow.py`）
   - `runtime/`：仅 Phase 2 用（`injector.py` ← 原 `runtime.py`）

2. 新增 `wfcllm/pretrain/`：

   ```python
   # wfcllm/pretrain/config.py
   @dataclass
   class PretrainConfig:
       stages: list[str] = field(default_factory=lambda: ["encoder", "lexical"])
       encoder: EncoderConfig = ...
       lexical: TokenChannelTrainConfig = ...

   # wfcllm/pretrain/pipeline.py
   class PretrainPipeline:
       def run(self) -> None:
           if "encoder" in self.cfg.stages:
               train_semantic_encoder(self.cfg.encoder)
           if "lexical" in self.cfg.stages:
               self._require_encoder_checkpoint()    # fail-fast
               train_lexical_channel(self.cfg.lexical)
   ```

3. CLI：

   ```
   python run.py --phase pretrain                            # 默认两个 stage 都跑
   python run.py --phase pretrain --stages encoder           # 只训编码器
   python run.py --phase pretrain --stages lexical           # 只训词汇通道（需 encoder ckpt）
   ```

   `--stages` 内部按固定顺序排（即使写 `lexical encoder` 也按 encoder → lexical 跑）。

4. `run_state.json`：

   ```jsonc
   {
     "pretrain": {
       "encoder": {"done": true,  "checkpoint": "data/models/encoder/"},
       "lexical": {"done": false, "checkpoint": null}
     }
   }
   ```

**权重兼容**：`core/model.py` 里的类名和 `state_dict` key 与现状一致 → 已训词汇通道权重直接加载。

### 4.2 Phase 2：自适应 γ 机器收口

**现状**：`watermark/entropy.py`、`entropy_profile.py`、`gamma_schedule.py` 散在 watermark 根目录；`scripts/calibrate.py` 是它们的离线驱动但物理脱离。

**整合**：

```
wfcllm/watermark/adaptive_gamma/
├── entropy.py        ← watermark/entropy.py
├── profile.py        ← watermark/entropy_profile.py
├── schedule.py       ← watermark/gamma_schedule.py
└── calibrate.py      ← scripts/calibrate.py 核心
```

`scripts/build_entropy_profile.py` 改名（消歧），只剩 argparse 壳。

`AdaptiveGammaConfig` 字段不变，仅调 import 路径。

### 4.3 Phase 3：calibration 子包

**现状**：`extract/calibrator.py`（FPR 阈值）、`extract/negative_corpus.py`（负样本生成）、`scripts/generate_negative_corpus.py`（CLI 壳）。

**整合**：

```
wfcllm/extract/calibration/
├── negative_corpus.py    ← extract/negative_corpus.py
├── threshold.py          ← extract/calibrator.py（更明确的名字）
└── runner.py             ← scripts/generate_negative_corpus.py 核心
```

CLI 命名：

- `scripts/generate_negative_corpus.py`：保留旧名（CLI 习惯）
- `scripts/calibrate_threshold.py`：新名

### 4.4 跨阶段评估包

**现状**：`extract/offline_analysis.py`（687 行）、`common/offline_code_eval.py`（360 行）、`scripts/evaluate_dual_channel.py` 散在三处。

**整合**：

```
wfcllm/evaluation/
├── code_execution.py     ← common/offline_code_eval.py
├── detection_report.py   ← extract/offline_analysis.py
└── dual_channel.py       ← scripts/evaluate_dual_channel.py
```

CLI 统一入口：

```
python scripts/evaluate.py exec       data/watermarked/*.jsonl --metric pass_at_1
python scripts/evaluate.py detection  data/results/*.jsonl --plot roc
python scripts/evaluate.py dual       data/watermarked/*.jsonl data/results/*.jsonl
```

### 4.5 `run.py` 拆 CLI 和 Orchestration

**现状**：56KB 单文件混了 argparse 定义、JSON config 合并、`run_state.json` 读写、phase 调度。

**整合**：

```
wfcllm/cli/
├── arguments.py          # 每 phase 一组 argparse group
├── config_resolver.py    # CLI override + JSON merge + dataclass 实例化
└── entry.py              # main()

wfcllm/orchestration/
├── state.py              # RunStateManager（封装 run_state.json）
├── phase_registry.py     # phase name → callable
├── pipeline.py           # PhaseOrchestrator
└── prereq.py             # 预飞自动触发（见 §6）
```

`run.py` 退化为：

```python
from wfcllm.cli.entry import main
if __name__ == "__main__":
    main()
```

### 4.6 测试重组

**现状**：`tests/test_run.py` 64KB 一文件覆盖所有 phase。

**整合**：

```
tests/
├── unit/                          # 重组现有 common/ encoder/ watermark/ extract/
│   ├── common/
│   ├── encoder/
│   ├── watermark/
│   ├── extract/
│   └── evaluation/                # 新
├── integration/                   # 拆自 test_run.py
│   ├── test_pretrain_phase.py
│   ├── test_watermark_phase.py
│   ├── test_extract_phase.py
│   ├── test_calibration_phases.py
│   ├── test_evaluation.py
│   ├── test_ablation.py
│   ├── test_cli_resolution.py
│   └── test_orchestration.py
├── lang/                          # 新：LanguageAdapter 契约测试
│   └── test_python_adapter.py
└── conftest.py
```

### 4.7 文档重组

**现状**：`docs/` 根目录 ~15 个 markdown 混杂。

**整合**：

```
docs/
├── design/         # 活的设计文档（核心方案 .md）
├── history/        # 已完成重构/fix 记录
├── progress/       # 汇报 md
├── plans/          # 实现计划（已存在，不动）
├── experiment/     # 实验报告（已存在，不动）
├── superpowers/    # skill 资料（不动）
└── references/     # 论文 PDF + 规则 CSV
```

## 5. 扩展接口

### 5.1 LanguageAdapter

**接口**：

```python
# wfcllm/lang/adapter.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass(frozen=True)
class StatementTypes:
    simple: frozenset[str]
    compound: frozenset[str]

class LanguageAdapter(ABC):
    name: str                                # "python" / "java" / "cpp" / "js"

    @abstractmethod
    def tree_sitter_language(self): ...      # tree_sitter.Language 对象

    @abstractmethod
    def statement_types(self) -> StatementTypes: ...

    @abstractmethod
    def positive_rules(self) -> list[Rule]: ...

    @abstractmethod
    def negative_rules(self) -> list[Rule]: ...

    @abstractmethod
    def extract_blocks(self, source: str) -> list[StatementBlock]:
        """语言相关的 AST 遍历，产出统一的 StatementBlock。"""

    # 可选钩子（多数语言共用默认实现）
    def normalize_whitespace(self, source: str) -> str: ...
    def parent_node_type(self, block: StatementBlock) -> str | None: ...
```

**注册**：

```python
# wfcllm/lang/registry.py
_REGISTRY: dict[str, type[LanguageAdapter]] = {}

def register(name: str):
    def deco(cls): _REGISTRY[name] = cls; return cls
    return deco

def get(name: str) -> LanguageAdapter:
    if name not in _REGISTRY:
        raise ValueError(
            f"未注册的语言：{name}。已注册：{sorted(_REGISTRY)}"
        )
    return _REGISTRY[name]()

# wfcllm/lang/python/adapter.py
@register("python")
class PythonAdapter(LanguageAdapter):
    name = "python"
    ...
```

**契约测试**（`tests/lang/test_python_adapter.py`）：所有 LanguageAdapter 实现必须通过同一组测试 —— 同一段 Python 代码经过适配器解析、变换、再解析后，行为与原有 `common/ast_parser.py` 一致。新增 Java/C++/JS Adapter 时复用这套契约。

**配置接入**：所有 phase 的 dataclass 增加 `language: str = "python"`，运行时统一 `lang.get(cfg.language)` 拿 adapter。

**今天只实现 PythonAdapter**；`wfcllm/lang/java/`、`cpp/`、`js/` 是空目录（含 `__init__.py`）占位。

### 5.2 DatasetAdapter

**接口**：

```python
# wfcllm/datasets/adapter.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

@dataclass(frozen=True)
class CodeSample:
    task_id: str
    prompt: str
    canonical_solution: str
    language: str                            # "python" / "java" / ...
    metadata: dict                            # tests、entry_point、test_setup_code 等

class DatasetAdapter(ABC):
    name: str                                # "humaneval" / "mbpp" / "humanevalpack"
    supported_languages: tuple[str, ...]

    @abstractmethod
    def iter_samples(self, language: str | None = None) -> Iterator[CodeSample]: ...

    @abstractmethod
    def get_sample(self, task_id: str) -> CodeSample: ...

    def task_ids(self, language: str | None = None) -> list[str]:
        return [s.task_id for s in self.iter_samples(language)]
```

**实现**：

```python
@register_dataset("humaneval")
class HumanEvalAdapter(DatasetAdapter):
    name = "humaneval"
    supported_languages = ("python",)
    ...

@register_dataset("mbpp")
class MBPPAdapter(DatasetAdapter):
    name = "mbpp"
    supported_languages = ("python",)
    ...

@register_dataset("humanevalpack")
class HumanEvalPackAdapter(DatasetAdapter):
    name = "humanevalpack"
    supported_languages = ("python", "java", "cpp", "js")
    # 今天只实现 __init__ + 错误信息：
    def iter_samples(self, language=None):
        raise NotImplementedError(
            "HumanEvalPack Adapter 留作接口；多语言阶段实现"
        )
```

**联合校验**：`config_resolver` 在解析时检查 `(dataset, language)` 组合 —— `dataset` 必须 `supports` 当前 `language`。

**配置接入**：所有 phase config 增加 `dataset: str = "humaneval"`。

### 5.3 消融实验框架

**最小可用形态**：YAML 描述笛卡尔积扫描 + JSONL 指标记录。**不做**分布式调度、HPO、自动超参建议。

**Sweep 规格**（YAML）：

```yaml
# configs/ablation/example_dual_channel.yaml
name: dual_channel_vs_threshold
base_config: configs/base_config.json
phases: [watermark, extract]                 # 每组合跑哪些 phase
axes:
  watermark.token_channel.mode: [semantic-only, lexical-only, dual-channel]
  extract.fpr_threshold:        [2.0, 2.5, 3.0, 3.5]
output_dir: data/ablation/dual_channel_vs_threshold
metrics:
  - z_score_mean
  - hit_rate
  - false_positive_rate
  - pass_at_1
```

**核心类**：

```python
# wfcllm/ablation/sweep.py
@dataclass
class SweepSpec:
    name: str
    base_config: Path
    phases: list[str]
    axes: dict[str, list]                    # 点号路径 -> 值列表
    output_dir: Path
    metrics: list[str]

    def expand(self) -> Iterator[ResolvedConfig]:
        """笛卡尔积展开为每组合的实际 config。"""

# wfcllm/ablation/runner.py
class AblationRunner:
    def run(self, spec: SweepSpec):
        for i, cfg in enumerate(spec.expand()):
            run_id = f"{i:04d}_{cfg.short_hash()}"
            if self._already_done(run_id): continue   # 断点恢复
            results = run_phases(cfg, spec.phases)
            metrics = collect_metrics(results, spec.metrics)
            self._append_jsonl(run_id, cfg, metrics)
```

**输出**：`data/ablation/<name>/results.jsonl`，每行一组合 `{run_id, axes_values, metrics}`。

**CLI**：

```
python scripts/run_ablation.py configs/ablation/example_dual_channel.yaml
python scripts/run_ablation.py configs/ablation/example_dual_channel.yaml --resume
```

**消融轴覆盖（已确认要扫的三组）**：

| 维度 | 涉及 config 路径 |
|---|---|
| 双通道 + 检测阈值 | `watermark.token_channel.mode`、`watermark.token_channel.switch_threshold`、`extract.fpr_threshold`、`extract.adaptive_detection.mode` |
| Margin + LSH | `watermark.margin_base`、`watermark.margin_alpha`、`watermark.adaptive_gamma.enabled`、`watermark.adaptive_gamma.strategy`、`watermark.lsh_d`、`watermark.lsh_gamma` |
| LLM + 编码器 | `watermark.lm_model_path`、`encoder.model_name`、`encoder.use_lora`、`encoder.embed_dim` |

**关键设计点**：所有这些 knob 已经在现有 dataclass config 里，本次重构**不引入新参数**，仅确认它们都能通过 `axes` 的点号路径触达。`AblationRunner` 不需要懂参数语义，只做"沿点号路径覆盖 + 跑 phase + 收指标"。

## 6. CLI 与预飞自动触发

### 6.1 主流程 phase

```
--phase pretrain                  # Phase 1：encoder + lexical（默认两个都训）
  --stages encoder lexical        # 可选子集
--phase watermark                 # Phase 2
--phase extract                   # Phase 3
```

### 6.2 预飞 phase（独立可跑）

```
--phase build-entropy-profile     # Phase 2 用：扫熵分布
--phase generate-negative         # Phase 3 用：生成无水印负语料
--phase calibrate-threshold       # Phase 3 用：从负语料算 z-score 阈值
```

### 6.3 预飞自动触发（默认行为）

**主流程 phase 启动时自动检查依赖产物，缺什么补什么**，不需要 flag。

**watermark 检查顺序**：

1. `adaptive_gamma.enabled` 是否 true？
   - 否 → 跳过；
   - 是 → 检查 `adaptive_gamma.profile_path` 文件是否存在：
     - 存在 → 跳过；
     - 不存在 → 自动触发 `build-entropy-profile`（记录到 `run_state.json`），完成后继续。

**extract 检查顺序**：

1. `fpr_threshold` 是否标记为"需校准"？（config 字段 `extract.calibration.required: bool`，默认 false）
   - false → 用 config 里的固定值；
   - true → 检查 `extract.calibration.threshold_path` 是否存在：
     - 存在 → 加载该阈值覆盖 config；
     - 不存在 → 自动触发 `generate-negative`（如果负语料也缺）→ `calibrate-threshold` → 加载阈值。

**用户可见的日志**：

```
$ python run.py --phase watermark --config configs/base_config.json
[orchestrator] watermark requires entropy profile (adaptive_gamma.enabled=true)
[orchestrator] profile not found at data/calibration/codet5_humaneval.json
[orchestrator] auto-triggering build-entropy-profile (~2h on this hardware)
[build-entropy-profile] ...
[orchestrator] entropy profile ready, resuming watermark
[watermark] ...
```

### 6.4 旧 phase 名处置

旧 phase 名 `encoder` 和 `token-channel-train` **直接断掉**，错误信息提示新写法：

```
$ python run.py --phase token-channel-train
[error] phase 'token-channel-train' 已废弃，请使用：
        python run.py --phase pretrain --stages lexical

$ python run.py --phase encoder
[error] phase 'encoder' 已废弃，请使用：
        python run.py --phase pretrain --stages encoder
```

`watermark`、`extract`、`generate-negative` 名字保留不变。新增 `build-entropy-profile`、`calibrate-threshold` 两个 phase 名。

### 6.5 全量参数表（保留）

现有 CLI 上的所有 override flag（`--secret-key`、`--lm-model-path`、`--dataset`、`--gamma-strategy`、`--token-channel-enabled` 等）全部保留，行为不变。

## 7. 配置 schema 变化

**Top-level 字段新增**：

```jsonc
{
  "language": "python",         // 新；缺省 python，旧 config 无需改
  "dataset": "humaneval",       // 新；缺省 humaneval

  "pretrain": { ... },          // 新；包装 encoder + lexical
  "encoder":  { ... },          // 保留（pretrain.encoder 的别名读路径）
  "watermark": { ... },         // 不动
  "extract":  { ... },          // 不动
  "token_channel_train": { ... }  // 保留（pretrain.lexical 的别名读路径）
}
```

`config_resolver` 在加载时把旧顶层 `encoder`/`token_channel_train` 合并到 `pretrain.encoder`/`pretrain.lexical`，旧 config 文件零改动可跑。当新旧字段同时存在时：

- 若值一致 → 静默使用；
- 若值冲突 → fail-fast 报错，要求用户清理 config 后重试（不静默选一方，避免误用）。

## 8. 迁移与兼容性

### 8.1 已训权重兼容

- 编码器 `state_dict` 不动 → 旧 ckpt 直接加载；
- 词汇通道 `state_dict` 不动 → 旧 ckpt 直接加载（`core/model.py` 类名 + key 与原 `token_channel/model.py` 一致）。

### 8.2 旧 import 路径过渡

迁移期内（≤1 个月）在旧路径放 shim：

```python
# wfcllm/watermark/entropy.py（旧路径，迁移后保留 shim）
import warnings
warnings.warn(
    "wfcllm.watermark.entropy 已迁移至 wfcllm.watermark.adaptive_gamma.entropy",
    DeprecationWarning, stacklevel=2,
)
from wfcllm.watermark.adaptive_gamma.entropy import *
```

类似 shim 覆盖：
- `wfcllm.watermark.entropy` → `wfcllm.watermark.adaptive_gamma.entropy`
- `wfcllm.watermark.entropy_profile` → `wfcllm.watermark.adaptive_gamma.profile`
- `wfcllm.watermark.gamma_schedule` → `wfcllm.watermark.adaptive_gamma.schedule`
- `wfcllm.extract.calibrator` → `wfcllm.extract.calibration.threshold`
- `wfcllm.extract.negative_corpus` → `wfcllm.extract.calibration.negative_corpus`
- `wfcllm.common.offline_code_eval` → `wfcllm.evaluation.code_execution`

迁移期满后整批删除（一次性 commit）。

### 8.3 `run_state.json` 迁移

启动时检测旧 schema（顶层有 `encoder`/`token_channel_train` 字段）：

1. 备份原文件到 `data/run_state.json.bak.<timestamp>`；
2. 重写为新 schema（`pretrain.encoder`/`pretrain.lexical`），完成时间和 checkpoint 字段保留；
3. 日志打印一次性迁移消息。

### 8.4 旧 CLI 用户引导

`README.md` 增加"CLI 迁移对照表"小节，列出每个旧 phase 名对应的新写法。

## 9. 测试策略

### 9.1 单元测试

- `tests/lang/test_python_adapter.py` —— PythonAdapter 实现一组**契约测试**（StatementTypes 完备、extract_blocks 与旧行为一致、变换规则可枚举且数量与旧一致）。新增 Adapter 时复用同一契约。
- `tests/unit/evaluation/` —— 评估包搬迁后的单元测试，覆盖 pass@k、ROC、混淆矩阵计算。
- 现有 `tests/unit/common/`、`encoder/`、`watermark/`、`extract/` 适配新 import 路径。

### 9.2 集成测试

- `test_pretrain_phase.py` —— 验 `--stages encoder`、`--stages lexical`、`--stages encoder lexical` 三种模式；验 `lexical` 在缺 encoder ckpt 时 fail-fast。
- `test_watermark_phase.py` —— 验 watermark 缺 entropy profile 时自动触发 `build-entropy-profile`。
- `test_extract_phase.py` —— 验 extract 在 `calibration.required=true` 且阈值缺失时自动触发负语料生成 + 阈值校准。
- `test_calibration_phases.py` —— 三个预飞 phase 的独立运行。
- `test_ablation.py` —— `SweepSpec.expand()` 笛卡尔积正确性 + `AblationRunner` 断点恢复。
- `test_cli_resolution.py` —— 旧 config schema 自动迁移到新；CLI override 优先级；`(dataset, language)` 校验。
- `test_orchestration.py` —— `RunStateManager` 旧 schema 迁移；`PhaseOrchestrator` 调度。

### 9.3 离线约束

所有测试加 `HF_HUB_OFFLINE=1`；使用 `data/models/codet5-base/`、`data/datasets/`。

## 10. 实施顺序建议

写实现计划时分阶段（防止一次破坏太多）：

1. **基础设施**：`common/registry.py`、`cli/`、`orchestration/`（包括 `RunStateManager`、`prereq.py`）拆出来，`run.py` 退化为壳；现有功能行为不变。
2. **`evaluation/` 包**：把三处评估代码搬迁，加 shim。
3. **`adaptive_gamma/` 子包**：搬熵剖面三件 + scripts 改名，加 shim。
4. **`extract/calibration/` 子包**：搬负语料 + 阈值校准 + scripts 改名，加 shim。
5. **`token_channel/` 三分层 + `pretrain/`**：先内部重组 + 加 shim；后接 `pretrain/` 编排层。
6. **`lang/` + `datasets/` 适配器**：先 ABC + Registry + PythonAdapter / HumanEval+MBPP Adapter；改 `common/ast_parser.py` 和 `common/dataset_loader.py` 走 Adapter；加 `language`/`dataset` 配置字段。
7. **`ablation/` 框架**：SweepSpec + Runner + 单元测试 + 一个 example YAML。
8. **`generator.py` 拆 `orchestrator.py` + `semantic_channel.py`**：最有风险，放最后。
9. **测试重组 + 文档归档**：拆 `test_run.py`、归类 `docs/`。
10. **shim 清理**：上述全部稳定后，一次性 commit 删除所有 deprecation shim。

每阶段独立可发布，单测/集成测全绿才进下一阶段。

## 11. 范围之外（明确不做）

- 不实现 Java/C++/JS LanguageAdapter（只留接口骨架）；
- 不实现 HumanEvalPack DatasetAdapter（只留接口骨架）；
- 不引入新算法参数；
- 不重写 `generator.py` 内部算法（仅按职责切两片）；
- 不做分布式 sweep 调度；
- 不做超参建议 / Bayesian HPO；
- 不动 `experiment/`、`tools/`、已训模型权重；
- 不引入新外部依赖。

## 12. 风险与决策点

| 风险 | 缓解 |
|---|---|
| `token_channel/` 三分层时 `model.py` / `features.py` 在训/推共享，切的时候可能 import 循环 | 把共享代码全放 `core/`；`training/` 和 `runtime/` 只单向依赖 `core/` |
| `generator.py` 拆两片可能破坏 KV-cache snapshot 和 cascade 时序 | 放实施顺序最后；保留完整集成测试（特别是 cascade fallback 路径） |
| 自动触发预飞会让用户误以为某次 watermark 跑得慢 | 日志显式打印"auto-triggering ..."和预估耗时；`run_state.json` 记录每个预飞何时被自动触发 |
| 新 `(language, dataset)` 校验可能让旧 config 报错 | `config_resolver` 在缺字段时默认 `python`/`humaneval`，确保旧 config 100% 可加载 |
| 测试拆分时 `test_run.py` 内部依赖丢失 | 每拆一组测试，先在新位置跑通，再删旧文件中对应部分；保留 64KB 原文件直到全部迁移完毕 |
