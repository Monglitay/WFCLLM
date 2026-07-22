# WFCLLM

这个仓库是代码大模型水印实验项目。当前主方法是“无 carrier 的门控语义窗口方法”：模型生成代码时，先按语句窗口切分代码，再由门控模型判断哪些窗口适合改写，最后用语义 LSH 规则判断水印信号。这里的重点是：水印不是靠插入固定 carrier 文本，而是靠门控模型选择窗口后做语义等价或近似等价的代码改写。

当前已经跑过的一轮核心实验是 Python HumanEval 全量 164 题。服务器保存的结果是：

```text
Pass@1: 113 / 164 = 68.9024%
n >= 3 条件 TPR: 80 / 130 = 61.5385%
```

`Pass@1` 可以简单理解为：164 道题里，有多少题生成的单个答案能通过真实测试。`n >= 3 条件 TPR` 可以理解为：只看检测器有至少 3 个可靠判断窗口的样本时，水印被检测出来的比例。这个 TPR 是实验诊断指标，不等于正式统计校准结论。

## 目录速览

常用目录如下：

```text
configs/wfcllm/experiments/        实验配置
scripts/experiments/               一键运行脚本
wfcllm/                            项目代码
tests/                             测试
docs/                              方法和协议文档
data/                              本地数据和运行产物，通常不要提交
```

这次 HumanEval 实验主要用到：

```text
configs/wfcllm/experiments/python_humaneval_full.json
scripts/experiments/run_python_humaneval_full.sh
scripts/experiments/run_gated_experiment.sh
README.md
```

## 服务器准备

建议在 AutoDL 这类服务器上把仓库、模型、数据、实验输出都放到数据盘：

```text
/root/autodl-tmp
```

不要把大模型、数据集、实验输出放到系统盘 `/`，系统盘通常很小。

如果是新服务器，先进入数据盘：

```bash
cd /root/autodl-tmp
```

克隆仓库：

```bash
git clone https://github.com/Monglitay/WFCLLM.git WFCLLM
cd /root/autodl-tmp/WFCLLM
```

如果服务器上已经有仓库，直接进入即可：

```bash
cd /root/autodl-tmp/WFCLLM
```

确认当前代码是最新的 `main`：

```bash
git fetch origin main
git switch -C main FETCH_HEAD
git log -1 --oneline
```

最后一行应该显示一个最新提交。当前可复现实验状态对应的提交是：

```text
6e94ab2 feat: integrate gated semantic experiment state
```

如果服务器后续又有更新，以远程 `main` 最新提交为准。

## 安装 Python 环境

推荐使用 conda，并把环境也放到数据盘：

```bash
/root/miniconda3/bin/conda create -p /root/autodl-tmp/conda/envs/WFCLLM python=3.11 -y
```

安装依赖：

```bash
/root/miniconda3/bin/conda run -p /root/autodl-tmp/conda/envs/WFCLLM pip install -r requirements.txt
```

如果只想检查环境是否能导入项目，可以运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  /root/miniconda3/bin/conda run -p /root/autodl-tmp/conda/envs/WFCLLM \
  python run.py --status
```

这里的三个 `OFFLINE` 变量表示不要临时联网下载 HuggingFace 模型或数据。正式跑实验时，也建议一直带着它们，避免实验中途因为下载失败或版本变化出问题。

## 需要提前放好的文件

完整跑 HumanEval 实验前，服务器本地必须已经有这些东西：

```text
1. 代码生成模型
2. 语义编码模型
3. gate 基础模型 codet5-small
4. HumanEval 数据集
5. gate 训练用 source catalog
6. gate full run 用 source catalog
7. negative final_code JSONL
```

这些文件不会由脚本自动下载。路径由环境变量告诉脚本。

建议按下面这种方式组织：

```text
/root/autodl-tmp/models/
  <local-code-generation-model>/
  <local-semantic-encoder>/
  codet5-small/

/root/autodl-tmp/data/
  datasets/
  gate_sources/
    python_humaneval_pilot.jsonl
    python_humaneval_full.jsonl
  negative/
    humaneval_negative_final_code.jsonl
```

如果你的实际路径不一样，没有关系，把后面环境变量改成你的真实路径即可。

## 环境变量怎么填

进入仓库根目录：

```bash
cd /root/autodl-tmp/WFCLLM
```

先设置 conda 路径：

```bash
export CONDA_EXE=/root/miniconda3/bin/conda
export CONDA_ENV_PREFIX=/root/autodl-tmp/conda/envs/WFCLLM
```

设置模型路径：

```bash
export GENERATION_MODEL_PATH=/root/autodl-tmp/models/<local-code-generation-model>
export REWRITE_MODEL_PATH="${GENERATION_MODEL_PATH}"
export SEMANTIC_ENCODER_MODEL_PATH=/root/autodl-tmp/models/<local-semantic-encoder>
export GATE_BASE_MODEL_PATH=/root/autodl-tmp/models/codet5-small
```

这几个变量的意思是：

- `GENERATION_MODEL_PATH`：用来生成 HumanEval 答案的代码模型。
- `REWRITE_MODEL_PATH`：用来生成改写候选的模型。本次实验可以先设成和 `GENERATION_MODEL_PATH` 一样。
- `SEMANTIC_ENCODER_MODEL_PATH`：用来把代码窗口变成语义向量的模型。
- `GATE_BASE_MODEL_PATH`：训练门控模型用的基础模型，通常是本地的 `codet5-small`。

设置数据路径：

```bash
export DATASET_PATH=/root/autodl-tmp/data/datasets
export PILOT_SOURCE_CATALOG=/root/autodl-tmp/data/gate_sources/python_humaneval_pilot.jsonl
export FULL_SOURCE_CATALOG=/root/autodl-tmp/data/gate_sources/python_humaneval_full.jsonl
export NEGATIVE_INPUT=/root/autodl-tmp/data/negative/humaneval_negative_final_code.jsonl
```

这几个变量的意思是：

- `DATASET_PATH`：HumanEval 数据所在目录。
- `PILOT_SOURCE_CATALOG`：先做小规模可行性检查用的数据清单。
- `FULL_SOURCE_CATALOG`：正式 full run 用的数据清单。
- `NEGATIVE_INPUT`：没有水印的代码样本，用来估计误报情况。

注意：`PILOT_SOURCE_CATALOG` 和 `FULL_SOURCE_CATALOG` 必须是两个不同文件。脚本会检查这一点。这样做是为了避免先用一批数据调方法，正式实验又用同一批数据，导致结果不干净。

设置输出目录和实验规模：

```bash
export EXPERIMENT_ROOT=/root/autodl-tmp/wfcllm-runs/python-humaneval-full
export SAMPLE_LIMIT=164
export CALIBRATION_LIMIT=32
export GATE_EPOCHS=3
```

这几个变量的意思是：

- `EXPERIMENT_ROOT`：本次实验所有输出放在哪里。
- `SAMPLE_LIMIT=164`：跑完整 HumanEval 164 题。
- `CALIBRATION_LIMIT=32`：快速校准时取多少 negative 样本。full run 会使用脚本里的 full 流程。
- `GATE_EPOCHS=3`：门控模型训练轮数。想更快可以调小，想更充分可以调大。

## 一键运行 HumanEval 全量实验

所有变量设置好以后，运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  scripts/experiments/run_python_humaneval_full.sh
```

这个脚本会按顺序做这些事：

```text
1. 检查配置和运行环境
2. 准备 gate 训练/验证需要的私有 key 和 manifest
3. 用 pilot 数据做一次可行性检查
4. 准备 full run 数据
5. 训练门控模型
6. 验证门控模型
7. 生成 HumanEval 164 题答案
8. 用 negative 样本校准检测阈值
9. 对生成结果做水印检测
10. 生成报告
11. 做审计检查
```

脚本结束时会打印：

```text
completed: <run_dir>
```

这里的 `<run_dir>` 就是正式结果目录。

## 结果在哪里看

如果你按上面的例子设置：

```bash
export EXPERIMENT_ROOT=/root/autodl-tmp/wfcllm-runs/python-humaneval-full
```

正式结果会在：

```text
/root/autodl-tmp/wfcllm-runs/python-humaneval-full/run/
```

常看文件：

```text
run/inputs/final_code.jsonl
```

这是最终生成的代码。检测器正式只看这个文件里的最终代码，不看生成过程日志。

```text
run/calibration/reference_calibration.json
```

这是 negative 样本上得到的校准结果。

```text
run/detection/
```

这里是检测结果和每个窗口的细节。

```text
run/reports/
```

这里是汇总报告。

```text
run/audit/
```

这里是审计结果，用来确认实验有没有违反“不能用测试结果来挑答案”等规则。

## 本次已经跑出的结果文件

这次服务器上保存过一份汇总结果：

```text
/tmp/gamma25_gate_pass113/humaneval_summary.json
/tmp/gamma25_gate_pass113/humaneval_results.jsonl
/tmp/gamma25_gate_pass113/n_ge_3_tpr_summary.json
```

其中：

```text
humaneval_summary.json
```

看 HumanEval Pass@1 汇总。

```text
humaneval_results.jsonl
```

看 164 道题每一题是否通过。

```text
n_ge_3_tpr_summary.json
```

看 `n >= 3` 条件下的 TPR 汇总。

已保存数字：

```text
Pass@1 = 113 / 164 = 68.9024%
n >= 3 条件 TPR = 80 / 130 = 61.5385%
未纳入 n >= 3 分母的 positive = 34 / 164
heldout negative false positive = 4 / 128
```

这里的 `n >= 3` 是指检测时可靠窗口数量至少为 3。窗口太少的样本不放进这个条件 TPR 的分母。

## 如何单独验证代码环境

如果只是想确认当前代码能跑，不想立刻跑完整实验，可以先跑这些轻量测试：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -p /root/autodl-tmp/conda/envs/WFCLLM pytest \
  tests/detection/test_gated_pipeline.py \
  tests/generation/test_gated_generator.py \
  tests/generation/test_window_rewriter.py \
  tests/cli/test_experiment_preflight.py \
  tests/cli/test_gated_generation_dataset_selection.py \
  tests/integration/test_experiment_matrix.py \
  tests/lang/test_cpp_adapter.py \
  tests/lang/test_java_adapter.py \
  tests/windowing/test_multilanguage.py \
  tests/generation/test_multilanguage_rewriter.py -q
```

正常情况下会看到类似：

```text
68 passed
```

如果报 `No module named tree_sitter_cpp` 或 `No module named tree_sitter_java`，说明依赖没装全，重新运行：

```bash
/root/miniconda3/bin/conda run -p /root/autodl-tmp/conda/envs/WFCLLM pip install -r requirements.txt
```

## 常见问题

### 1. 脚本说找不到模型

检查这些变量是否是真实存在的目录：

```bash
echo "$GENERATION_MODEL_PATH"
echo "$REWRITE_MODEL_PATH"
echo "$SEMANTIC_ENCODER_MODEL_PATH"
echo "$GATE_BASE_MODEL_PATH"
```

再逐个检查：

```bash
test -d "$GENERATION_MODEL_PATH" && echo OK
test -d "$SEMANTIC_ENCODER_MODEL_PATH" && echo OK
test -d "$GATE_BASE_MODEL_PATH" && echo OK
```

### 2. 脚本说 pilot 和 full catalog 不能一样

这是正常保护。请准备两个不同文件：

```bash
echo "$PILOT_SOURCE_CATALOG"
echo "$FULL_SOURCE_CATALOG"
```

它们不能指向同一个路径。

### 3. 跑到一半找不到 HumanEval 数据

检查：

```bash
echo "$DATASET_PATH"
find "$DATASET_PATH" -maxdepth 3 -type f | head
```

`DATASET_PATH` 必须指向本地数据目录。

### 4. 想重新跑一次干净实验

换一个新的输出目录即可：

```bash
export EXPERIMENT_ROOT=/root/autodl-tmp/wfcllm-runs/python-humaneval-full-rerun-001
```

然后重新运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  scripts/experiments/run_python_humaneval_full.sh
```

不要直接覆盖旧目录，这样方便对比不同运行。

### 5. 服务器没有 GPU

完整实验需要 GPU。可以先跑轻量测试确认代码没坏，但 full generation/training 没有 GPU 会非常慢或直接失败。

## 不要提交什么

不要提交这些内容：

```text
data/
模型文件
数据集原文
私有 key
checkpoint
大 JSONL
实验日志
实验输出目录
```

仓库应该只提交代码、配置、小脚本、测试和必要文档。

## 协议说明

当前实验遵守两个基本规则：

```text
1. 生成和选择答案时，不能看 HumanEval 测试是否通过。
2. 检测器正式输入只使用最终代码 final_code.jsonl。
```

这意味着 Pass@1 只能在生成结束后统计，不能反过来影响生成过程。

相关文档：

- [方法说明](docs/WFCLLM_METHOD.md)
- [Artifact 结构](docs/ARTIFACT_SCHEMA.md)
- [No Quality Gate 规则](docs/NO_QUALITY_GATE_PROTOCOL.md)
- [Strict Code-Only Detector](docs/STRICT_CODE_ONLY_DETECTOR.md)
- [Diagnostic Upper Bound](docs/DIAGNOSTIC_UPPER_BOUND.md)
