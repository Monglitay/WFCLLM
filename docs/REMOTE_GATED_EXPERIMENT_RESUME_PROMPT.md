# WFCLLM 门控语义窗口实验续跑 Prompt

你现在是 WFCLLM 远程实验执行代理。不要只提供方案或命令示例，请直接连接服务器，从当前残留状态继续准备并运行完整的单机单卡正式实验。遇到普通环境、网络、依赖、路径、数据准备或代码运行问题时自行分析和解决；只有遇到凭据失效、服务器不可用、需要购买额外资源或必须由用户作出方法选择时才停下来询问。

## 一、服务器信息

SSH：

```bash
ssh -p 48974 root@connect.bjb2.seetacloud.com
```

SSH 已经验证可以直接连接。不要在命令、脚本、日志、环境变量或消息中保存、回显用户提供的密码。

服务器配置：

- Ubuntu 22.04；
- 单机单卡 NVIDIA GeForce RTX 5090；
- 显存约 32 GB；
- 数据盘 `/root/autodl-tmp`，约 80 GB；
- 正式实验只能使用 GPU 0；
- 不运行多卡实验；
- 不同时启动多个争抢 GPU 的正式任务。

连接后先执行只读检查：

```bash
date -Is
hostname
nvidia-smi
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
df -h / /root/autodl-tmp
conda env list
screen -ls || true
pgrep -af 'run.py|run_gated|wfcllm|gate-data|gate-train|gate-validate' || true
```

## 二、已知现场状态

上一次只读检查确认：

- 没有运行中的实验进程；
- 没有 `screen` 会话；
- GPU 显存占用为 0 MiB，利用率为 0%；
- 正式 gated 八阶段尚未启动；
- 没有 pilot/full gate-data；
- 没有 gate checkpoint；
- 没有正式 gate bundle；
- 没有 HumanEval `final_code.jsonl`；
- 没有 calibration、detection、report 或 audit 结果。

服务器已有以下准备资源：

```text
/root/autodl-tmp/models/codet5-small
/root/autodl-tmp/models/qwen2.5-coder-0.5b
/root/autodl-tmp/wfcllm-experiments/wfcllm_build_gated_catalog.py
/root/autodl-tmp/WFCLLM-gated-incomplete-20260717-002407
```

其中：

- CodeT5-small 权重约 242 MB；
- Qwen2.5-Coder-0.5B 权重约 988 MB；
- `WFCLLM-gated-incomplete-20260717-002407` 只有不完整的 `.git`，不能作为正式代码目录；
- source catalog 构建脚本已经写入，但尚未生成任何 catalog；
- 不要把已有准备工作误报为实验结果。

服务器旧仓库：

```text
/root/autodl-tmp/WFCLLM
```

旧仓库当前是无关且有未提交修改的分支：

```text
codex/watermark-mechanism-v3-semantic-lsh-reset
ad2e703d46cba66379636bbd7a5fb0872decb940
```

绝对不要修改、reset、clean、覆盖、提交或删除这个旧仓库。它只能作为只读模型、数据集、checkpoint、whitening 和缓存来源。

## 三、目标代码版本

远程仓库：

```text
https://github.com/Monglitay/WFCLLM.git
```

目标分支：

```text
codex/gated-semantic-window-watermark-impl
```

固定提交：

```text
f91463cbc2015a74aa96152dc75c253fdf5a5aa4
```

新的隔离代码目录：

```text
/root/autodl-tmp/WFCLLM-gated
```

不要继续使用不完整 clone。保留它用于诊断，不删除。

先使用 AutoDL 学术资源加速重新浅克隆：

```bash
source /etc/network_turbo
git clone --depth 1 \
  --branch codex/gated-semantic-window-watermark-impl \
  --single-branch \
  https://github.com/Monglitay/WFCLLM.git \
  /root/autodl-tmp/WFCLLM-gated
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

如果目标目录已存在，先检查。若不是干净且完整的正确仓库，则重命名为带时间戳的备份目录，再重新 clone，不要直接删除。

如果 GitHub 连续两次失败，不要无限重试。改用本机 Git bundle 回退：

本机仓库：

```text
/home/monglitay/PycharmProjects/WFCLLM/.worktrees/gated-semantic-window-watermark
```

本机创建 bundle：

```bash
git -C /home/monglitay/PycharmProjects/WFCLLM/.worktrees/gated-semantic-window-watermark \
  bundle create /tmp/wfcllm-gated-f91463c.bundle \
  codex/gated-semantic-window-watermark-impl
```

上传：

```bash
scp -P 48974 \
  /tmp/wfcllm-gated-f91463c.bundle \
  root@connect.bjb2.seetacloud.com:/root/autodl-tmp/
```

服务器克隆 bundle：

```bash
git clone \
  /root/autodl-tmp/wfcllm-gated-f91463c.bundle \
  /root/autodl-tmp/WFCLLM-gated
cd /root/autodl-tmp/WFCLLM-gated
git switch codex/gated-semantic-window-watermark-impl
```

代码准备完成后必须确认：

```bash
cd /root/autodl-tmp/WFCLLM-gated
git branch --show-current
git rev-parse HEAD
git status --short
```

HEAD 必须为：

```text
f91463cbc2015a74aa96152dc75c253fdf5a5aa4
```

不要向远程推送，不创建 PR。

如果发现真实实验路径存在阻塞缺陷，可以在隔离代码目录实施最小 runtime 修复，但必须：

- 不修改方法含义；
- 不降低正式阈值；
- 不放宽 final-code-only 合同；
- 不允许 fake/diagnostic 工件冒充正式工件；
- 添加最小目标测试；
- 运行测试和 `git diff --check`；
- 保存 patch；
- 在最终报告中记录；
- 未经用户要求不推送补丁。

## 四、最终目标

完整运行以下正式流程：

```text
pilot gate-data
→ full gate-data
→ gate-train
→ gate-validate
→ HumanEval generate/embed
→ calibrate
→ detect/extract
→ report
→ audit
→ posthoc pass@1
```

仓库正式八阶段为：

```text
gate-data
→ gate-train
→ gate-validate
→ generate
→ calibrate
→ detect
→ report
→ audit
```

HumanEval 全量 164 条只能用于最终生成和评估，不能进入 gate-data、gate-train 或 gate-validate。

## 五、环境准备

优先使用已有 `WFCLLM` conda 环境：

```bash
conda run -n WFCLLM python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

不要重装或降级一个已经支持 RTX 5090 的 PyTorch/CUDA 环境。

检查必要依赖：

```bash
conda run -n WFCLLM python -c "import tree_sitter, tree_sitter_python, transformers, datasets, torch; print('imports ok')"
```

只安装实际缺失的依赖。大型缓存放到：

```text
/root/autodl-tmp/huggingface
/root/autodl-tmp/pip-cache
```

设置：

```bash
export HF_HOME=/root/autodl-tmp/huggingface
export PIP_CACHE_DIR=/root/autodl-tmp/pip-cache
```

下载完成后，正式实验使用离线模式：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0
```

## 六、模型选择

先检查旧仓库和数据盘已有真实模型。优先复用，不重复下载。

### 门控基础模型

使用：

```bash
export GATE_BASE_MODEL_PATH=/root/autodl-tmp/models/codet5-small
```

### gate-data 重写模型

为尽快完成大量窗口候选重写，先验证并优先使用：

```bash
export REWRITE_MODEL_PATH=/root/autodl-tmp/models/qwen2.5-coder-0.5b
```

必须确认当前 causal rewriter 能对该模型完成本地离线最小生成。

### 正式 HumanEval 生成模型

优先在旧仓库查找已有、完整、适合 completion 的代码模型，例如 DeepSeek Coder 7B Base。设置：

```bash
export GENERATION_MODEL_PATH=<真实完整路径>
```

如果没有可用 7B 模型，选择服务器上已有且能在 RTX 5090 正常运行的真实代码 causal LM。不要使用 fake model。

### 语义编码器

优先查找旧仓库中的 `data/models/codet5-base`、语义编码器 checkpoint 和 whitening 文件，设置：

```bash
export SEMANTIC_ENCODER_MODEL_PATH=<codet5-base 真实路径>
export SEMANTIC_ENCODER_CHECKPOINT_PATH=<可选 checkpoint>
export SEMANTIC_WHITENING_PATH=<可选 whitening>
```

必须验证 base model、checkpoint 和 whitening 相互匹配。

## 七、下载规则

下载顺序：

```text
服务器已有资源
→ AutoDL network_turbo
→ Hugging Face 镜像
→ 本机下载
→ rsync/scp 上传
```

AutoDL 加速：

```bash
source /etc/network_turbo
```

Hugging Face 镜像回退：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

下载结束后：

```bash
unset HF_ENDPOINT
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

同一种失败方式最多尝试两次。不要把访问令牌写入配置或日志。

模型必须验证配置、tokenizer、真实权重、Git LFS 状态和 `local_files_only=True` 加载，并通过最小前向或生成测试。

## 八、实验根目录

```bash
RUN_TAG=gated-semantic-window-v1-$(date +%Y%m%d-%H%M%S)
export EXPERIMENT_ROOT=/root/autodl-tmp/wfcllm-experiments/${RUN_TAG}
mkdir -p "${EXPERIMENT_ROOT}"/{private,sources,pilot,full,gate-cache,logs,runtime,checksums,backup}
```

所有正式工件必须写入这个数据盘目录。

## 九、source catalog

已有脚本：

```text
/root/autodl-tmp/wfcllm-experiments/wfcllm_build_gated_catalog.py
```

它只是准备工具，不能直接假定满足正式合同。先审查并做最小必要修正。

必须准备：

```text
${EXPERIMENT_ROOT}/sources/pilot_source_catalog.jsonl
${EXPERIMENT_ROOT}/sources/full_source_catalog.jsonl
```

要求：

- pilot 对应 2,000–5,000 个独立 window groups；
- full 对应 20,000–50,000 个独立 window groups；
- 不能用 catalog 行数冒充独立 group 数；
- pilot 必须是按 repository/task/function group 的确定性子集；
- 不允许同一 group 跨 train/validation/test；
- HumanEval 绝对不能进入 catalog；
- OSS 必须有真实 repository ID 和 license ID；
- main-generation 必须来自真实模型；
- 至少三个真实 source model identity，不得用同一个模型伪造三个 ID；
- 不能使用测试通过率、正确性、reward 或 oracle 筛选来源。

允许来源：

```text
main_generation
mbpp_train
mbpp_validation
oss_python
parser_boundary
```

不要只用 Django、Requests、Flask 三个 repository group 后直接开始正式训练。应优先复用已有许可明确的多个 Python 仓库，并保证足够多的独立 repository/task groups。

已有 builder 使用 `rows[:N]` 生成 pilot 的方式不可靠，可能丢失 source-family/model 多样性。必须改为确定性的 group-aware、分层 pilot 选择，并用实际窗口解析规则预估 group 数。

如果缺少三个真实模型生成来源：先搜索服务器既有多模型生成结果；没有时，用三个真实小型代码模型分别生成非 HumanEval Python 样本。Qwen2.5-Coder-0.5B 可以作为其中一个来源，其余使用可信官方来源的小型 causal code model。不得伪造模型 ID。

## 十、私有密钥

不要向用户询问水印密钥。使用：

```text
scripts/wfcllm_prepare_gated_experiment.py
```

在服务器本地生成 32 个 training keys、8 个独立 holdout keys 和 1 个 deployment key。

密钥位于 `${EXPERIMENT_ROOT}/private`，权限为 `0600`。

禁止读取或打印密钥原文，禁止写入配置、日志和公共工件。deployment key 不能参与 gate 训练；正式 generation 和 detection 使用同一个 deployment key 文件。

## 十一、已知 wrapper 问题

`scripts/run_gated_single_gpu.sh` 当前可能使用同一个 `SOURCE_CATALOG` 同时运行 pilot 和 full，这与两个规模区间不相容。

不要盲目执行。选择一种方式：

1. 做最小 runtime patch，增加 `PILOT_SOURCE_CATALOG` 和 `FULL_SOURCE_CATALOG`；或
2. 按 wrapper 中的参数手动逐阶段运行。

pilot 和 full 可以使用不同 catalog/manifest，但方法配置除 `gate_data.scale` 外必须保持同一实验合同。不能降低准入阈值，不能伪造 `passed=true`。

## 十二、smoke test

```bash
cd /root/autodl-tmp/WFCLLM-gated
git diff --check
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

运行 gated 相关测试：

```bash
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM pytest \
  tests/gate \
  tests/windowing \
  tests/integration/test_gated_production_cli.py \
  tests/integration/test_gated_workflow.py \
  -q
```

再用 `rg --files tests | rg 'gated'` 找到实际 gated generation/detection 测试并补跑。

执行最小真实 smoke：真实模型、真实 semantic encoder、真实私有 key、正式 backend，不使用 `--backend fake`。smoke 与正式目录隔离，至少验证一小段 gate-data、一条 HumanEval 生成、严格四字段输入和 final-code-only 检测。

## 十三、正式阶段

使用：

```text
configs/wfcllm/gated_semantic_window_pilot.json
configs/wfcllm/gated_semantic_window_v1.json
```

### Pilot gate-data

使用 pilot catalog、pilot manifest、真实训练/holdout key 和真实模型。完成后检查 `gate-data/feasibility_summary.json`，必须为 `passed=true`。失败时分析 admission，修复真实数据或 runtime；绝不能改结果文件或降低阈值。

### Full gate-data

使用 full catalog，并传入已通过的 pilot feasibility。必须形成 20,000–50,000 个独立 groups，并满足正式正负样本、窗口长度、statement family、R3 相对 R1 改善和 holdout 迁移要求。

### Gate train

使用 full gate-data 训练 CodeT5-small 门控模型。OOM 时优先降低 gate batch size，不缩减 full group 数。

### Gate validate

单卡主机按仓库设计使用 CPU 正式验证：

```bash
CUDA_VISIBLE_DEVICES=""
```

只有同时满足以下条件才能继续：

- `validated=true`；
- `diagnostic_test_backend=false`；
- `formal_eligible=true`；
- bundle hash 匹配；
- decision agreement 不低于 99.9%；
- accepted-span consensus 为 100%；
- suitable false-positive rate 不高于 5%。

### HumanEval generate/embed

恢复 `CUDA_VISIBLE_DEVICES=0`。使用 full deployment key、正式 generation model、冻结 gate bundle 和同一 semantic encoder。HumanEval 必须全量 164，不设置 sample limit。

正式 detector input 为 `<FULL_RUN>/inputs/final_code.jsonl`。每行字段必须严格等于 `id`、`dataset`、`prompt`、`final_code`。

### Calibrate

使用独立的严格四字段负样本，优先采用没有进入 gate-data 的 MBPP test/reference 或其他 held-out negative corpus。不能使用 HumanEval 正式正样本本身进行负样本校准。

### Detect/extract

检测器只能读取正式 `final_code.jsonl` 和 calibration artifact，不能读取 generation audit、retry trace、candidate sidecar、gate-data、training metrics 或 posthoc pass report。

### Report 与 audit

依次运行 report 和 audit。detector input integrity、no-quality-gate integrity、gate artifact integrity、bundle/hash integrity 和 secret leakage 检查必须通过。

## 十四、持久化运行

服务器有 `screen`。使用单一持久化会话运行正式控制脚本：

```bash
screen -dmS wfcllm_gated_<RUN_TAG> bash -lc '
  set -o pipefail
  cd /root/autodl-tmp/WFCLLM-gated
  export CUDA_VISIBLE_DEVICES=0
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  bash <正式控制脚本> 2>&1 | tee <完整日志>
  status=${PIPESTATUS[0]}
  if [ "$status" -eq 0 ]; then
    touch <EXPERIMENT_ROOT>/PIPELINE_SUCCESS
  else
    printf "%s\n" "$status" > <EXPERIMENT_ROOT>/PIPELINE_FAILED
  fi
  exit "$status"
'
```

启动后检查 `screen -ls`、进程、GPU 和日志，确认 SSH 断开不会停止任务。

## 十五、“你可以走了”通知

只有正确 commit、环境、模型、数据、catalog、私有 key、真实 smoke 和持久化正式进程全部正常，且日志持续增长、GPU 有真实占用后，才发送：

> 服务器已连接，代码、环境、GPU、模型、数据和持久化任务均已确认正常，正式实验已经稳定启动。你可以走了，剩余流程我会自主运行、低频监控、自动处理问题、保存结果，并在全部验收通过后按约定关闭服务器和本机电脑。

不要在 clone、下载或准备阶段提前发送。

## 十六、低频监控

- 启动后 2–5 分钟检查一次是否立即失败；
- 稳定后每 45–60 分钟检查一次；
- 阶段切换时检查一次；
- 异常时立即检查；
- 不要每分钟轮询；
- 不重复发送无变化状态；
- 每次最多读取日志末尾 50–100 行。

监控：

```bash
screen -ls || true
pgrep -af 'run.py|run_gated|wfcllm' || true
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
df -h /root/autodl-tmp
tail -n 80 <完整日志>
```

## 十七、故障处理边界

记录失败阶段、首个命令、首个关键错误、资源状态、commit、配置 hash、修复和合同影响。

- OOM：清理本任务遗留进程，降低 gate batch size；
- bitsandbytes 不兼容：显存足够时成对修改 pilot/full config，关闭 4-bit、使用 bf16；
- 网络错误：按加速、镜像、本机上传回退；
- pilot 失败：修复真实数据分布或模型运行，不改阈值；
- gate-validate 失败：不进入正式 generation；
- fake artifact：不能作为 formal artifact；
- 不使用 pass/fail 控制候选；
- 不缩减 HumanEval 164；
- 不把 full group 数降到 20,000 以下。

## 十八、最终验收与结果

一次性确认：

1. 八阶段全部完成；
2. 正式 bundle 可离线重载且 hash 匹配；
3. gate validation 阈值全部通过；
4. `final_code.jsonl` 恰好 164 行；
5. 每行恰好四字段且 ID 唯一；
6. calibration、detection、report 可解析；
7. 三项 audit 通过；
8. 公共工件没有明文 key；
9. fake/diagnostic 工件没有冒充正式工件；
10. 旧方法默认五阶段没有被破坏；
11. `compileall`、`git diff --check` 通过；
12. 记录 `git status --short`。

正式流程结束后才运行 `scripts/evaluate.py exec` 计算 HumanEval pass@1。posthoc 报告必须声明它没有参与生成、重试、选择、校准或检测。

最终报告先给原始数字，包括模型、catalog、groups、pilot feasibility、训练、validation、一致率、FPR、HumanEval 数量、gate coverage、重写成本、hit/miss/abstain、检测率、AUROC/TP/FP（若能正确计算）、pass@1、audit、日志和结果路径。不得把 smoke、fake 或旧实验指标当作当前结果。

## 十九、备份与关机

关机前把必要结果备份到：

```text
/home/monglitay/WFCLLM-experiment-results/<RUN_TAG>
```

备份配置、日志、catalog 统计/hash、pilot feasibility、full manifest、训练指标、validated bundle、validation、final code、calibration、detection、reports、audits、posthoc、Git patch、环境和 checksum。不要把任何私有 key 原文放入公共备份。

使用：

```bash
rsync -avP --partial \
  -e 'ssh -p 48974' \
  root@connect.bjb2.seetacloud.com:<远程备份目录>/ \
  /home/monglitay/WFCLLM-experiment-results/<RUN_TAG>/
```

只有正式八阶段、HumanEval 164、posthoc、验收、备份和 SHA-256 校验全部成功后才关机。实验失败时不要关机，保留现场并报告。

成功时先发送最终摘要，然后关闭服务器：

```bash
ssh -p 48974 root@connect.bjb2.seetacloud.com 'sync'
ssh -p 48974 root@connect.bjb2.seetacloud.com 'sudo shutdown -h now'
```

最后关闭本机：

```bash
sync
sudo shutdown -h now
```

本机关机必须是最后一个动作，之后不再调用工具、运行命令或发送消息。

## 二十、沟通频率

只在以下时机汇报：正式实验稳定启动、pilot 完成、full gate-data 完成、gate-train/validate 完成、HumanEval generate 完成、detect/audit 完成、真实阻塞、最终完成并准备关机。

现在直接开始执行。不要重新描述上次检查结论，不要只写计划。先完成代码恢复、模型和数据验证、source catalog、私有 key 和真实 smoke，然后在 `screen` 中启动正式实验。正式任务稳定运行后再通知用户可以离开。
