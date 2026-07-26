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

## 影响嵌入、提取和检测效果的主要参数

这一节专门解释“哪些参数会影响实验结果”。这里说的“嵌入”就是生成代码时把水印信号放进去；“提取”就是只看最终代码，把水印信号读出来；“检测”就是判断这份代码是否带水印。

最重要的结论先放前面：

```text
想提高 Pass@1：不要让改写太激进，优先保证语义不变。
想提高嵌入率：让更多窗口进入改写，或者增加每个窗口的尝试次数。
想提高 TPR：让更多可靠窗口命中水印，或者降低检测阈值。
想降低 FPR：提高检测阈值，或者要求更多可靠窗口。
想跑得更快：减少样本数、训练轮数、改写次数，或者减少模型大小。
```

这些目标通常互相拉扯。比如增加改写次数可能提高嵌入率和 TPR，但也会更慢，并且有可能伤害 Pass@1。降低检测阈值可能提高 TPR，但也可能提高 FPR。

### 1. 实验规模参数

这些参数一般通过环境变量设置。

```bash
export SAMPLE_LIMIT=164
export CALIBRATION_LIMIT=32
export GATE_EPOCHS=3
```

| 参数 | 当前建议 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `SAMPLE_LIMIT` | `164` | HumanEval 覆盖范围、运行时间 | `164` 表示跑完整 HumanEval。调小可以快速试跑，但不能当 full-164 结果汇报。 |
| `CALIBRATION_LIMIT` | `32` | 校准速度、negative 覆盖 | fast run 里会限制 negative 数量。negative 越少越快，但 FPR 估计更不稳。full run 应尽量用完整 negative 输入。 |
| `GATE_EPOCHS` | `3` 或配置默认 | gate 训练质量、运行时间 | 轮数更多，gate 可能更稳，但更慢，也可能过拟合。轮数太少，gate 可能选不准窗口。 |

如果只是检查流程，先用小 `SAMPLE_LIMIT`。如果要复现实验结果，必须用：

```bash
export SAMPLE_LIMIT=164
```

### 2. 生成模型参数

这些在配置里：

```json
"generation": {
  "max_new_tokens": 512,
  "temperature": 0.0,
  "top_p": 0.95,
  "top_k": 0,
  "seed": 7,
  "load_in_4bit": true,
  "max_total_sampled_tokens": 32768
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `generation.max_new_tokens` | `512` | Pass@1、运行时间 | 每题最多生成多少 token。太小会截断答案，Pass@1 下降；太大会更慢。 |
| `generation.temperature` | `0.0` | 稳定性、Pass@1 | `0.0` 更稳定，便于复现。调高会更随机，有时提高个别题质量，但结果波动更大。 |
| `generation.top_p` | `0.95` | 生成多样性 | 和 temperature 一起控制随机性。temperature 为 `0.0` 时影响通常较小。 |
| `generation.top_k` | `0` | 生成候选范围 | `0` 通常表示不额外限制 top-k。 |
| `generation.seed` | `7` | 复现性 | 同样模型、同样环境、同样 seed 更容易复现实验。 |
| `generation.load_in_4bit` | `true` | 显存、速度、数值差异 | 省显存，适合单卡。可能带来轻微数值差异。 |
| `generation.max_total_sampled_tokens` | `32768` | 总预算 | 防止生成阶段无限变长。太小会提前停，太大更耗时。 |

一般不建议为了追 Pass@1 临时改这些参数后直接和原结果比较。要比较，就保存新配置和新输出目录。

### 3. 窗口切分参数

“窗口”就是一次准备判断和改写的代码片段。当前方法不是随便改整段代码，而是先按语句切成窗口。

配置位置：

```json
"method": {
  "windowing": {
    "max_units": 3,
    "max_preceding_units": 3,
    "excluded_statement_types": [...],
    "compound_header_singleton": true
  }
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `windowing.max_units` | `3` | 嵌入机会、Pass@1、检测窗口数 | 一个窗口最多包含几个语句。值越大，窗口更长，可能更容易产生稳定信号，但改写风险也更高。当前限制在 1 到 3。 |
| `windowing.max_preceding_units` | `3` | gate 判断质量 | 给 gate 看多少前文。前文更多，判断可能更准，但输入更长。 |
| `windowing.excluded_statement_types` | 见配置 | Pass@1、安全性 | 排除 `import`、`raise`、`assert`、`pass`、函数/类头等不适合改写的语句。排除越多，Pass 更稳，但嵌入机会更少。 |
| `windowing.compound_header_singleton` | `true` | 结构安全 | `if/for/while/def/class` 这类头部单独成窗，避免和 body 混改导致语法或语义问题。 |

这次你提到的“一个语句生成后，门控模型决定等几个语句组成窗口再嵌入”，主要就是这部分和 gate 部分一起控制的。

### 4. gate 参数

gate 可以理解为“窗口选择器”。它决定什么时候关闭窗口，以及这个窗口适不适合改写。

配置位置：

```json
"method": {
  "gate": {
    "uncertain_boundary_policy": "close_and_skip",
    "max_input_tokens": 256
  }
}
```

代码里还有 gate 阈值对象：

```text
close_low
close_high
suitable_accept
max_units
max_input_tokens
```

| 参数 | 当前值/含义 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `gate.uncertain_boundary_policy` | `close_and_skip` | Pass@1、嵌入率 | gate 不确定时关闭窗口并跳过。更保守，Pass 更稳，但嵌入率可能下降。 |
| `gate.max_input_tokens` | `256` | gate 输入长度、速度 | gate 每次最多看多少 token。太小会缺上下文，太大更慢。 |
| `close_low` | 代码阈值 | 窗口长度 | close 分数低于它时倾向继续等下一个语句。 |
| `close_high` | 代码阈值 | 窗口长度 | close 分数高于它时倾向关闭窗口。 |
| `suitable_accept` | 代码阈值 | 嵌入率、Pass@1 | suitable 分数高于它才认为窗口适合改写。调低会增加嵌入机会，也可能增加坏改写；调高更保守。 |

通俗地说：

```text
gate 越宽松：嵌入机会更多，TPR 可能更高，但 Pass@1 风险更大。
gate 越严格：Pass@1 更稳，但很多样本可能窗口太少，TPR 分母变小或嵌入率下降。
```

### 5. 改写参数

改写就是对一个窗口生成候选版本，然后看能不能带来水印命中。

配置位置：

```json
"method": {
  "rewrite": {
    "max_attempts": 3,
    "experiment_budgets": [1, 3],
    "key_blind": true,
    "temperature": 0.2,
    "top_p": 0.95,
    "max_new_tokens": 16,
    "generation_attempts": 3,
    "candidate_selection": "fixed-key-blind-abc-trajectory/v1"
  }
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `rewrite.max_attempts` | `3` | 嵌入率、速度、Pass@1 | 每个窗口最多尝试几个改写候选。更多尝试通常提高命中机会，但更慢，也增加误改风险。 |
| `rewrite.experiment_budgets` | `[1, 3]` | 对比不同预算 | 用来比较只试 1 次和最多试 3 次的效果。 |
| `rewrite.key_blind` | `true` | 方法合规性 | 改写器不知道密钥，不能直接按答案作弊式生成水印。 |
| `rewrite.temperature` | `0.2` | 改写多样性、Pass@1 | 越高改写越发散，可能增加命中，也更容易破坏语义。 |
| `rewrite.top_p` | `0.95` | 改写候选范围 | 控制候选采样范围。太低保守，太高发散。 |
| `rewrite.max_new_tokens` | `16` | 改写长度、速度 | 每个窗口改写最多生成多少 token。太小可能生成不完整，太大更慢且更容易跑偏。 |
| `rewrite.generation_attempts` | `3` | 后端生成次数 | 后端内部最多生成几次候选。更多更慢。 |
| `rewrite.candidate_selection` | 固定轨迹 | 复现性 | 固定 A/B/C 轨迹便于复现和审计。 |

当前代码还会做结构检查：候选必须还是 1 到 3 个语句，不能跑到别的父结构里。结构不合格就拒绝，不会写进最终答案。

### 6. 语义保持参数

这部分控制“改写后是否还像原来的代码”。如果改写语义不稳，Pass@1 可能掉。

配置位置：

```json
"method": {
  "semantic": {
    "preservation": {
      "rule": "codet5-cosine-to-original/v1",
      "threshold": 0.9
    }
  }
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `semantic.preservation.threshold` | `0.9` | Pass@1、嵌入率 | 改写和原窗口的语义相似度要达到这个阈值。调高更安全但更少改写通过；调低嵌入更多但 Pass 风险更大。 |
| `semantic.preservation.rule` | `codet5-cosine-to-original/v1` | 判断方式 | 用语义编码模型比较原窗口和候选窗口。 |

另外，当前 Python 改写里有一些 AST/literal 等价证明。如果候选能被证明是 AST 等价或字面量等价，就可以绕开纯 cosine 的不稳定性。这对 Pass@1 很重要，因为它比“看起来相似”更可靠。

### 7. 语义 LSH 参数

语义 LSH 是水印信号的核心。它把窗口语义向量映射成一个签名，再根据密钥判断这个窗口是否“命中”。

配置位置：

```json
"semantic_lsh": {
  "rule_name": "semantic_lsh",
  "lsh_d": 12,
  "lsh_gamma": 0.45,
  "semantic_margin": 0.0,
  "use_ordinal_keying": false
}
```

同样的信息也在：

```json
"method": {
  "semantic": {
    "lsh": {
      "d": 12,
      "gamma": 0.45,
      "margin": 0.0
    }
  }
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `semantic_lsh.rule_name` | `semantic_lsh` | 是否是当前方法 | 必须是 `semantic_lsh`。旧 carrier 方法会出现 `keyed_text_region`，那不是当前主方法。 |
| `semantic_lsh.lsh_d` / `semantic.lsh.d` | `12` | 命中难度、FPR、TPR | 维度越大，随机命中更难，FPR 更低，但嵌入也更难；维度越小，容易命中，但误报风险更高。 |
| `semantic_lsh.lsh_gamma` / `semantic.lsh.gamma` | `0.45` | 命中宽松程度 | 越大通常越宽松，更多窗口可能命中，TPR 可能升，FPR 也可能升；越小更严格。 |
| `semantic_lsh.semantic_margin` / `semantic.lsh.margin` | `0.0` | 稳定性 | 要求窗口离边界至少有多少距离。提高 margin 会减少不稳定命中，但也减少可用证据。 |
| `semantic_lsh.use_ordinal_keying` | `false` | key 派生方式 | 是否把窗口序号也放入 key。当前不使用 ordinal keying，减少生成和提取时序号变化带来的不一致。 |

简单理解：

```text
lsh_d 控制“签名空间有多细”。
lsh_gamma 控制“命中判定有多宽”。
semantic_margin 控制“离边界太近的证据要不要放弃”。
```

这次保存的诊断里，用了 `hit_rate >= 0.60` 和 `reliable_window_count >= 3` 计算条件 TPR。这里的 `hit_rate` 就是命中窗口数除以可靠窗口数。

### 8. 检测参数

检测阶段只看最终代码，不看生成过程。

配置位置：

```json
"detector": {
  "target_fpr": 0.05,
  "minimum_reliable_windows": 2,
  "statistic": "reliable_window_hit_rate",
  "abstain_policy": "exclude_from_denominator"
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `detector.target_fpr` | `0.05` | FPR/TPR 平衡 | 目标误报率。越低越严格，FPR 降低但 TPR 可能下降。 |
| `detector.minimum_reliable_windows` | `2` | 覆盖率、可靠性 | 至少需要多少可靠窗口才给检测结论。调高会更稳，但更多样本变成证据不足。 |
| `detector.statistic` | `reliable_window_hit_rate` | 检测统计量 | 使用命中率作为检测分数。 |
| `detector.abstain_policy` | `exclude_from_denominator` | 分母怎么算 | 不稳定窗口不算入命中率分母。 |

检测时每个窗口会有三种状态：

```text
hit      命中水印
miss     没命中
abstain  不稳定，放弃判断
```

计算：

```text
reliable_window_count = hit_count + miss_count
hit_rate = hit_count / reliable_window_count
```

如果 `reliable_window_count` 小于 `minimum_reliable_windows`，检测结果就是证据不足。

这也是为什么我们单独报告过 `n >= 3`：当可靠窗口至少 3 个时，判断更有意义。

### 9. 校准参数

校准阶段用 negative 样本估计“没有水印时会有多少误报”。

配置位置：

```json
"calibration": {
  "method": "pooled_negative_binomial_right_tail",
  "group_by": "pooled_binomial_tail",
  "target_fpr": 0.05
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `calibration.method` | `pooled_negative_binomial_right_tail` | 阈值估计方式 | 用 pooled 的方式估计右尾概率。 |
| `calibration.group_by` | `pooled_binomial_tail` | negative 如何分组 | 把窗口证据汇总到同一个背景分布里。 |
| `calibration.target_fpr` | `0.05` | 检测阈值 | 和 `detector.target_fpr` 对齐。越小越保守。 |
| `NEGATIVE_INPUT` | 环境变量 | FPR 是否可信 | negative 样本越干净、越接近真实无水印代码，FPR 估计越可信。 |

不要用 positive 样本来挑正式阈值。可以做诊断扫阈值，但报告时要说明是诊断，不是正式校准。

### 10. gate-data 参数

gate-data 是训练 gate 前的数据准备阶段。它决定 gate 学到什么样的窗口选择规则。

配置位置：

```json
"gate_data": {
  "window_lengths": [1, 2, 3],
  "rewrite_count": 3,
  "rewrite_budgets": [1, 3],
  "training_key_count": 32,
  "holdout_key_count": 8,
  "label_thresholds": {
    "reliable_success_rate_r3_min": 0.6,
    "structurally_valid_rewrite_rate_r3_min": 0.6666666666666666,
    "unstable_candidate_rate_r3_max": 0.1
  }
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `gate_data.window_lengths` | `[1, 2, 3]` | gate 学哪些窗口长度 | 包含 1、2、3 语句窗口。少了某个长度，gate 对该长度判断会弱。 |
| `gate_data.rewrite_count` | `3` | 训练标签质量、成本 | 每个窗口准备多少候选改写。更多候选让标签更充分，但数据准备更慢。 |
| `gate_data.rewrite_budgets` | `[1, 3]` | 比较不同尝试次数 | 能看到只试一次和试三次的差别。 |
| `gate_data.training_key_count` | `32` | gate 训练稳定性 | 训练用 key 数量。更多 key 更稳，但更慢。 |
| `gate_data.holdout_key_count` | `8` | 泛化检查 | 留出 key 用来检查 gate 是否只记住训练 key。 |
| `reliable_success_rate_r3_min` | `0.6` | 什么算 positive 窗口 | 三次尝试下成功率至少 0.6 才更可能被标成适合。 |
| `structurally_valid_rewrite_rate_r3_min` | `2/3` | 结构安全 | 三次候选里结构有效比例太低，就不适合改。 |
| `unstable_candidate_rate_r3_max` | `0.1` | 稳定性 | 不稳定候选比例太高，gate 应该少选这种窗口。 |

这里的标签会直接影响 gate 的选择风格。标签太宽松，gate 会选很多危险窗口；标签太严格，gate 会漏掉很多可嵌入窗口。

### 11. gate-train 参数

gate-train 是训练门控模型。

配置位置：

```json
"gate_train": {
  "learning_rate": 0.00002,
  "max_epochs": 4,
  "early_stopping_patience": 1,
  "loss_weights": {
    "suitable_false_positive": 4.0
  }
}
```

| 参数 | 当前值 | 影响什么 | 怎么影响 |
| --- | --- | --- | --- |
| `gate_train.learning_rate` | `0.00002` | 训练稳定性 | 太大可能训练不稳，太小可能学得慢。 |
| `gate_train.max_epochs` | `4` | 训练充分度、时间 | 上限轮数。环境变量 `GATE_EPOCHS` 可覆盖运行轮数。 |
| `gate_train.early_stopping_patience` | `1` | 过拟合和时间 | 验证集不提升时提前停止。 |
| `loss_weights.suitable_false_positive` | `4.0` | Pass@1、安全性、嵌入率 | 对“把不适合窗口误判为适合”的惩罚更重。值越大，gate 越保守。 |

想让 gate 更激进，可以降低误判惩罚或 suitable 阈值；想保 Pass，更应保持保守。

### 12. gated 阶段序列

gated 预设不再有独立的 gate-validate 阶段（ADR 0007）。默认阶段序列由配置解析得到：

```text
encoder -> gate-data -> gate-train -> generate -> calibrate -> detect -> report
```

full 档在此之后追加 `audit` 阶段；fast 档跳过 audit。

要点：

- `encoder`：按数据集从本次 run 的 gate 源 catalog 训练语义编码器（ADR 0006），是 gate-data 的前置条件；也可用 `--semantic-encoder-checkpoint-path` 显式提供 checkpoint。
- gate-train 产出的候选 bundle 是唯一解析入口：generate、calibrate、detect 都通过同一个 publication manifest 解析并做哈希绑定校验，不存在"先验证再放行"的独立阶段。
- 使用外部 bundle 时（`method.gate.bundle_path` + `bundle_sha256`），序列从 `generate` 开始，只跑四个主阶段（generate/calibrate/detect/report），且必须显式提供语义编码器 checkpoint。

### 13. 这次结果里几个关键诊断口径

这次保存的诊断结果里有几个口径需要分清：

```text
full HumanEval Pass@1:
113 / 164 = 68.9024%

n >= 3 条件 TPR:
80 / 130 = 61.5385%

被排除在 n >= 3 分母外的 positive:
34 / 164

heldout negative false positive:
4 / 128
```

解释如下：

- `164`：HumanEval 全量题数。
- `113`：真实单候选通过测试的题数。
- `130`：可靠窗口数至少为 3 的 positive 样本数。
- `80`：在这 130 个样本里，被检测为带水印的数量。
- `34`：窗口数不足 3 的 positive 样本，没放进 `n >= 3` 条件 TPR 分母。
- `4 / 128`：heldout negative 里按该诊断阈值误报的数量。

如果你改了 `lsh_d`、`lsh_gamma`、`minimum_reliable_windows`、`target_fpr`、gate 阈值或改写次数，这些数字都会变，不能再直接引用这次结果。

### 14. 推荐调参顺序

如果目标是“先稳定复现，再提高”，建议按这个顺序：

```text
1. 先不要改参数，完整复现一次。
2. 只改一个参数，换新 EXPERIMENT_ROOT 跑。
3. 先看 Pass@1，确认没有明显掉。
4. 再看 reliable_window_count 覆盖率。
5. 再看 hit_rate / TPR。
6. 最后看 heldout negative 的 FPR。
```

不要一次改很多参数，否则不知道结果变化是谁造成的。

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
