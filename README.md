# WFCLLM Gate-only Full Reproduction Core

本仓库只维护 WFCLLM 的 Gated Lineage。当前唯一方法是
`gated_semantic_window_v1`，唯一公开实验形态是从头训练语义 encoder 与
Gate 的 Full Reproduction Profile。退出当前范围的实现、配置、运行产物和
文档只从 Git 历史与既有 Archival Tag 恢复。

## Reproduction Coverage

公开覆盖固定为五个组合：

| Wrapper | 语言 | 数据集 | 默认样本数 | 窗口合同 | 改写策略 |
|---|---|---|---:|---|---|
| `run_python_humaneval_full.sh` | Python | HumanEval | 164 | `python-statement-window/v1` | `python_ast_equivalent` |
| `run_python_mbpp_full.sh` | Python | MBPP | 974 | `python-statement-window/v1` | `python_ast_equivalent` |
| `run_cpp_humanevalpack_full.sh` | C++ | HumanEvalPack | 164 | `cpp-statement-window/v1` | `model_semantic_window` |
| `run_java_humanevalpack_full.sh` | Java | HumanEvalPack | 164 | `java-statement-window/v1` | `model_semantic_window` |
| `run_js_humanevalpack_full.sh` | JavaScript | HumanEvalPack | 164 | `js-statement-window/v1` | `model_semantic_window` |

五个配置位于 `configs/wfcllm/experiments/`。生成模型和 rewrite 模型由运行时
路径注入，不按模型复制配置。

## Fresh Reproduction Run

完整阶段顺序固定为：

```text
encoder → gate-data → gate-train → generate → calibrate → detect → report → audit
```

运行目录和状态文件必须不存在。当前执行路径不恢复旧状态、不接受外部历史
Gate Bundle，也不读取旧 schema 或旧报告布局。pilot 与 full Gate source
catalog 必须是两个不同文件。

在已经配置好的 `WFCLLM` conda 环境中准备本地资源：

```bash
export GENERATION_MODEL_PATH=/absolute/path/to/generation-model
# C++、Java、JavaScript 必须设置；Python AST 改写不需要。
export REWRITE_MODEL_PATH=/absolute/path/to/rewrite-model
export SEMANTIC_ENCODER_MODEL_PATH=/absolute/path/to/semantic-encoder-base
export GATE_BASE_MODEL_PATH=/absolute/path/to/gate-base-model
export DATASET_PATH=/absolute/path/to/local-datasets
export PILOT_SOURCE_CATALOG=/absolute/path/to/pilot-sources.jsonl
export FULL_SOURCE_CATALOG=/absolute/path/to/full-sources.jsonl
export NEGATIVE_INPUT=/absolute/path/to/held-out-negative-final-code.jsonl
export EXPERIMENT_ROOT=/absolute/path/to/a-new-run-root
```

实际运行示例：

```bash
bash scripts/experiments/run_python_humaneval_full.sh
```

缺少模型、数据集、source catalog、held-out negative input 或私有密钥时，
preflight/runner 会非零退出。Public Execution Surface 不会用 canonical
solution、简化检测器或测试后端替代真实资源。

## 统一 CLI

`run.py` 是 `wfcllm.cli.entry:main` 的薄入口。可单阶段运行当前链中的一个
阶段，但前置阶段必须已在同一 Fresh Reproduction Run 的状态文件中完成：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python run.py \
  --phase encoder \
  --config configs/wfcllm/experiments/python_humaneval_full.json \
  --run-dir /absolute/path/to/new-root/run \
  --state-file /absolute/path/to/new-root/run_state.json \
  --gate-source-catalog /absolute/path/to/full-sources.jsonl \
  --semantic-encoder-model-path /absolute/path/to/semantic-encoder-base
```

查看当前八阶段状态：

```bash
conda run -n WFCLLM python run.py \
  --status \
  --state-file /tmp/wfcllm-status.json
```

## Detector Input

唯一正式正样本检测输入是 `run/inputs/final_code.jsonl`，每行必须且只能包含：

```json
{"id":"HumanEval/0","dataset":"humaneval","prompt":"...","final_code":"..."}
```

generation audit、candidate sidecar、Gate 数据、pass/test 结果都不是检测输入。

## Calibration 与生成后评估

`NEGATIVE_INPUT` 是严格四字段的 held-out 未加水印生成。若条数低于当前
`target_negative_count`，`calibrate` 使用同一 generation model 在与正样本
不重叠的提示上补足；生成即采用，不使用 pass/test/正确性信号筛选。

当前 posthoc correctness 工具只支持 Pass@1，且只在生成完成后计算；
报告与 Metric Contract 只消费 Pass@1：

```bash
conda run -n WFCLLM python scripts/evaluate.py \
  /absolute/path/to/posthoc-correctness.jsonl \
  --output /absolute/path/to/run/reports/pass_report_posthoc.json
```

`--k` 与 Pass@k 已退出当前公开 surface。任何正确性结果都不得反馈到
generation、retry、candidate selection、calibration 或 detection。

为一个或多个当前 run 生成 Metric Contract JSONL：

```bash
conda run -n WFCLLM python scripts/wfcllm_metric_contract.py \
  --run-dir /absolute/path/to/run \
  --output /absolute/path/to/metric-contract.jsonl
```

Metric Contract 只读取已通过 audit 且核心 artifact SHA-256 未变化的当前
run，并无条件输出一行；未得到的 Pass@1 等指标记为 `null` 并附 caveat，
不因指标未达标而抑制输出。

## 离线验证

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -n WFCLLM python -m pytest --collect-only -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -n WFCLLM python -m pytest tests/ -q
bash -n scripts/experiments/run_gated_experiment.sh \
  scripts/experiments/run_*_full.sh
```

这些命令验证代码、合同和受控测试，不等于完成真实 GPU 科学复现。

## 文档

- `docs/WFCLLM_METHOD.md`：当前 Gate 方法。
- `docs/REPRODUCTION_ENVIRONMENT.md`：本地资源与运行环境。
- `docs/ARTIFACT_SCHEMA.md`：当前运行目录和 schema。
- `docs/STRICT_CODE_ONLY_DETECTOR.md`：严格四字段检测合同。
- `docs/NO_QUALITY_GATE_PROTOCOL.md`：正确性信号的因果隔离。
- `CONTEXT.md`：领域词汇。
- `docs/adr/`：仍然有效的架构决定。
