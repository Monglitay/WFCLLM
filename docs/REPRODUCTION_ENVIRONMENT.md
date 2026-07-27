# Reproduction Environment

## 软件

- 使用 conda 环境 `WFCLLM`，不要在 `base` 中运行项目。
- 依赖以 `requirements.txt` 为准。
- Python、pytest 和 Hugging Face 运行默认离线：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## Local Reproduction Resources

以下资源不进入 Git，必须由操作者提供本地绝对路径：

| 环境变量 | 要求 |
|---|---|
| `GENERATION_MODEL_PATH` | 本地 Hugging Face code generation model 目录 |
| `REWRITE_MODEL_PATH` | C++、Java、JavaScript 必需的本地 causal rewrite model 目录；Python 不需要 |
| `SEMANTIC_ENCODER_MODEL_PATH` | semantic encoder base 目录 |
| `GATE_BASE_MODEL_PATH` | Gate base model 目录 |
| `DATASET_PATH` | HumanEval、MBPP 或 HumanEvalPack 本地数据根目录 |
| `PILOT_SOURCE_CATALOG` | pilot Gate source JSONL |
| `FULL_SOURCE_CATALOG` | full Gate source JSONL，路径与内容都必须不同于 pilot |
| `NEGATIVE_INPUT` | held-out 未加水印、严格四字段 JSONL |
| `EXPERIMENT_ROOT` | 不存在的新运行根目录 |

source catalog 不能包含 HumanEval evaluation items。prepare 工具会为 pilot
和 full 各生成独立的 Training Key Bank、Holdout Key Bank、Deployment Key
与 source manifest。

## 运行

选择五个 wrapper 之一：

```bash
bash scripts/experiments/run_python_humaneval_full.sh
bash scripts/experiments/run_python_mbpp_full.sh
bash scripts/experiments/run_cpp_humanevalpack_full.sh
bash scripts/experiments/run_java_humanevalpack_full.sh
bash scripts/experiments/run_js_humanevalpack_full.sh
```

runner 在创建 `EXPERIMENT_ROOT` 前验证配置、语言、数据集、profile、parser、
rewrite 策略、模型 config/权重/tokenizer、本地数据集精确任务数、严格负样本
schema，以及 pilot/full catalog 的完整来源组成。任一资源缺失、结构不可加载
或 pilot/full catalog 路径/内容相同都会非零退出。若 root 已存在也会失败；
请使用新 root，不要覆盖或续跑。

## 本地目录

仓库只跟踪以下占位文件：

```text
data/models/.gitkeep
data/datasets/.gitkeep
data/checkpoints/.gitkeep
data/results/.gitkeep
```

models、datasets、keys、Gate cache、source catalogs、checkpoints、run trees、
结果与日志均受 `.gitignore` 保护。清理 tracked 文件时不得删除这些本地资源。

## 验证与科学结果

离线 pytest、compileall、preflight 和 shell syntax 只证明实现与合同可执行。
只有五个 Full Reproduction Profile 在真实模型、真实数据、私有 key 和 GPU 上
完整跑完，才构成科学复现。报告中必须明确区分这两者。
